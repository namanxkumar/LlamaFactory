# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX


# ---------------------------------------------------------------------------
# Answer-token loss weighting
# ---------------------------------------------------------------------------
# Tokens inside <answer>(x,y)</answer> spans are upweighted by this factor.
# Answer actions are rare relative to navigation actions, so boosting their
# gradient prevents the model from ignoring them during SFT.
# Set via ANSWER_TOKEN_WEIGHT env var (default 1.0 = disabled).
_ANSWER_TOKEN_WEIGHT = float(os.environ.get("ANSWER_TOKEN_WEIGHT", "1.0"))
_answer_open_ids: "list[int] | None" = None
_answer_close_ids: "list[int] | None" = None
_answer_debug_logged: bool = False


def _get_answer_tag_ids(tokenizer) -> "tuple[list[int], list[int]]":
    global _answer_open_ids, _answer_close_ids
    if _answer_open_ids is None:
        # Unwrap Processor → Tokenizer (Qwen3-VL uses a Processor as processing_class)
        tok = getattr(tokenizer, "tokenizer", tokenizer)
        _answer_open_ids = tok.encode("<answer>", add_special_tokens=False)
        _answer_close_ids = tok.encode("</answer>", add_special_tokens=False)
    return _answer_open_ids, _answer_close_ids


def _build_answer_weights(labels: "torch.Tensor", tokenizer) -> "torch.Tensor":
    """Build a (batch, seq_len) float weight tensor for answer-token upweighting.

    - 0.0  for IGNORE_INDEX positions (prompt / padding, excluded from loss)
    - _ANSWER_TOKEN_WEIGHT for tokens inside/including <answer>…</answer> spans
    - 1.0  for all other response tokens

    Uses decoded text matching instead of token-ID matching to avoid BPE
    boundary issues (e.g. ``\\n<`` merging into one token).
    """
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[labels == IGNORE_INDEX] = 0.0

    OPEN_TAG = "<answer>"
    CLOSE_TAG = "</answer>"

    for b in range(labels.size(0)):
        row = labels[b].tolist()
        # Decode each token individually to map token positions to text
        token_texts: list[str] = []
        for tid in row:
            if tid == IGNORE_INDEX:
                token_texts.append("")
            else:
                token_texts.append(tok.decode([tid]))

        # Build cumulative char offsets: char_offsets[t] = start char of token t
        char_offsets: list[int] = []
        cum = 0
        for txt in token_texts:
            char_offsets.append(cum)
            cum += len(txt)

        full_text = "".join(token_texts)

        # Find all <answer>...</answer> spans in the decoded text
        search_start = 0
        while True:
            open_pos = full_text.find(OPEN_TAG, search_start)
            if open_pos == -1:
                break
            close_pos = full_text.find(CLOSE_TAG, open_pos + len(OPEN_TAG))
            if close_pos == -1:
                break
            span_end = close_pos + len(CLOSE_TAG)

            # Upweight all tokens that overlap with [open_pos, span_end)
            for t in range(len(row)):
                tok_start = char_offsets[t]
                tok_end = tok_start + len(token_texts[t])
                if tok_end <= open_pos:
                    continue
                if tok_start >= span_end:
                    break
                # Token overlaps with the answer span
                if row[t] != IGNORE_INDEX:
                    weights[b, t] = _ANSWER_TOKEN_WEIGHT

            search_start = span_end

    return weights


def answer_weighted_loss_func(
    outputs: "torch.Tensor",
    labels: "torch.Tensor",
    tokenizer,
) -> "tuple[torch.Tensor, float, float, int]":
    """Cross-entropy loss with per-token upweighting for <answer>…</answer> spans.

    Returns (weighted_loss, answer_loss_scalar, nav_loss_scalar, answer_token_count).
    The sub-losses are detached floats for logging only; only weighted_loss
    receives gradients.

    Normalization: returns a weighted token-level mean (same scale as
    CrossEntropyLoss(reduction="mean")).  The caller (training_step) then
    divides by gradient_accumulation_steps because model_accepts_loss_kwargs=False
    and compute_loss_func=None — do NOT divide by num_items_in_batch here.
    """
    global _answer_debug_logged

    logits = outputs.get("logits")
    if logits is None:
        fallback = outputs.get("loss", torch.tensor(0.0))
        return fallback, 0.0, 0.0, 0

    logits = logits.float()

    # Causal-LM shift: token t predicts token t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    weights = _build_answer_weights(shift_labels, tokenizer).to(shift_logits.device)

    # One-time diagnostic: dump answer span tokens, their labels, and per-token loss
    if not _answer_debug_logged:
        _answer_debug_logged = True
        n_answer_toks = int((weights == _ANSWER_TOKEN_WEIGHT).sum().item())
        tok = getattr(tokenizer, "tokenizer", tokenizer)
        diag_parts = [
            f"[AnswerWeight] weight={_ANSWER_TOKEN_WEIGHT} | "
            f"method=decoded-text-match | "
            f"answer-span tokens in first batch={n_answer_toks}"
        ]
        # Show per-token detail for first batch element with answer tokens
        if n_answer_toks > 0:
            # Compute per-token loss for the diagnostic (before weighting)
            _diag_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())
            for b in range(shift_labels.size(0)):
                mask_b = weights[b] == _ANSWER_TOKEN_WEIGHT
                if not mask_b.any():
                    continue
                idxs = mask_b.nonzero(as_tuple=True)[0].tolist()
                tids = shift_labels[b, idxs].tolist()
                losses = _diag_loss[b, idxs].tolist()
                decoded = [tok.decode([int(t)]) for t in tids]
                # Logit stats at answer positions
                answer_logits = shift_logits[b, idxs]  # (n_answer, vocab)
                correct_logits = [shift_logits[b, idx, int(tid)].item() for idx, tid in zip(idxs, tids)]
                max_logits = answer_logits.max(dim=-1).values.tolist()
                diag_parts.append(
                    f"[AnswerWeight] batch={b} positions={idxs} "
                    f"token_ids={tids} decoded={decoded} "
                    f"per_token_loss={[f'{l:.2e}' for l in losses]} "
                    f"correct_logit={[f'{v:.2f}' for v in correct_logits]} "
                    f"max_logit={[f'{v:.2f}' for v in max_logits]}"
                )
                break  # only first matching batch element
        for part in diag_parts:
            logger.info_rank0(part)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
    token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())  # (batch, seq_len-1)

    # Weighted mean: divide by number of active tokens (not sum of weights)
    # so the loss scale matches standard CE and the weights only control
    # relative token importance, not overall loss magnitude.
    active_count = (weights > 0).float().sum().clamp(min=1)
    weighted_loss = (token_loss * weights).sum() / active_count

    # Unweighted per-action-type means for logging (detached, no grad)
    with torch.no_grad():
        answer_mask = weights == _ANSWER_TOKEN_WEIGHT  # True for answer-span tokens
        nav_mask = (weights == 1.0)                    # True for ordinary response tokens
        answer_loss = token_loss[answer_mask].mean().item() if answer_mask.any() else 0.0
        nav_loss = token_loss[nav_mask].mean().item() if nav_mask.any() else 0.0
        answer_count = int(answer_mask.sum().item())

    return weighted_loss, answer_loss, nav_loss, answer_count
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        elif _ANSWER_TOKEN_WEIGHT != 1.0 and "labels" in inputs:
            return_outputs = kwargs.get("return_outputs", False)
            labels = inputs["labels"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            loss, answer_loss, nav_loss, answer_count = answer_weighted_loss_func(outputs, labels, self.processing_class)
            # Log per-action-type losses to W&B / TensorBoard every step
            if self.model.training:
                self.log({
                    "loss/answer_tokens": answer_loss,
                    "loss/nav_tokens": nav_loss,
                    "train/answer_token_count": answer_count,
                })
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
