# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf harness: **one 5 Hz LM decoder layer** (prefill, ``TransformerBlock`` only).

Profiles a single ``tt_transformers`` ``TransformerBlock.forward`` in ``Mode.PREFILL`` per timed
iteration — attn norm + QKV/SDPA/O-proj + MLP for one layer index (default layer 0). Does **not**
run embed, other layers, final norm, or ``LMHead``.

On the first call, ``prepare_inputs_prefill`` builds embedded activations + RoPE slices + page
table; later timed iterations reuse those device tensors so Tracy isolates one layer forward
(not embed / host upload per iter).

Run from the repository root (example):

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/demos/ace_step_v1_5/perf/test_perf_llm_decoder_layer_tracy.py::test_perf_ace_step_llm_decoder_layer_tracy_profile \\
        -v -s

Filter ``tt-perf-report`` on signpost ``LLM_DECODER_LAYER_FORWARD``.

**Important:** do **not** set ``ACE_STEP_USE_TRACE=1``.

Optional environment variables:

- ``ACE_STEP_CKPT_DIR`` / ``ACE_STEP_LLM_PERF_VARIANT``: checkpoint root + LM folder.
- ``ACE_STEP_PERF_NO_DOWNLOAD`` / ``ACE_STEP_PERF_DOWNLOAD``: Hugging Face fetch control.
- ``ACE_STEP_LLM_DECODER_LAYER_IDX`` (default ``0``): which ``tt_model.layers[i]`` to profile.
- ``ACE_STEP_LLM_DECODER_LAYER_PERF_ITERS`` (default ``32``): timed perf-pass iterations.
- ``ACE_STEP_PERF_WARMUP`` (default ``4``): warmup layer forwards before the timed pass.
- ``ACE_STEP_LLM_DECODER_LAYER_PERF_SEQ`` (default ``128``): tokenized prompt length (via padding).
- ``ACE_STEP_PERF_SEED`` (default ``0``): RNG seed for synthetic token ids.
- ``ACE_STEP_TRACY_EACH_LLM_ITER``: set to ``1`` for one Tracy signpost per perf iteration.
- ``ACE_STEP_PROFILER_FLUSH_EVERY``: flush device profiler every N perf iterations (default ``1``).
"""

from __future__ import annotations

import contextlib
import gc
import os
from typing import Any

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import Profiler
from models.demos.ace_step_v1_5.perf._llm_tracy_common import (
    assert_no_lm_trace,
    ckpt_paths_from_env,
    enable_llm_tracy_perf_env,
    ensure_lm_checkpoint,
    is_ci,
    pad_prefill_tokens,
    prefill_seq_len_from_env,
    random_prefill_token_ids,
    tracy_signpost,
)
from models.demos.ace_step_v1_5.ttnn_impl.five_hz_causal_lm_experimental import AceStepFiveHzExperimentalTtnnCausalLM
from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_enable_tracy_profiler_env,
    ace_step_flush_device_profiler,
    ace_step_lm_prefill_l1_enabled,
)
from models.demos.ace_step_v1_5.ttnn_impl.qwen_prefill_l1 import ace_step_qwen_prefill_l1_op_context
from models.tt_transformers.tt.common import Mode, num_blocks_in_seq


class _SingleDecoderLayerPrefillCtx:
    """Prepare embedded prefill activations once, then run one ``TransformerBlock`` forward per call."""

    def __init__(self, qwen: Any, *, layer_idx: int) -> None:
        self.qwen = qwen
        self.layer_idx = int(layer_idx)
        n_layers = len(qwen.tt_model.layers)
        if self.layer_idx < 0 or self.layer_idx >= n_layers:
            raise ValueError(f"layer_idx={layer_idx} out of range [0, {n_layers})")
        self._page_table_torch = qwen._page_table_torch.clone()
        self._paged_cfg = qwen._paged_cfg
        self._cached_inputs: tuple[Any, ...] | None = None

    def prepare_layer_inputs(self, padded_tokens: torch.Tensor, *, seq_len: int) -> tuple[Any, ...]:
        if self._cached_inputs is not None:
            return self._cached_inputs

        prefill_seq_len = int(padded_tokens.shape[1])
        block_size = int(self._paged_cfg.block_size)
        n_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
        page_table_for_prefill = self._page_table_torch[:, :n_blocks].contiguous()

        inputs = self.qwen.tt_model.prepare_inputs_prefill(
            padded_tokens,
            page_table=page_table_for_prefill,
            batch_size=1,
            user_id=0,
        )
        prefill_input, rot_mats_global, rot_mats_local, page_table_tt, _chunk_page_table_tt = inputs
        _ = seq_len  # used by callers for logging; padding already applied in padded_tokens
        self._cached_inputs = (prefill_input, rot_mats_global, rot_mats_local, page_table_tt)
        return self._cached_inputs

    def run_decoder_layer_forward(self, padded_tokens: torch.Tensor, *, seq_len: int) -> None:
        """Timed path: one layer forward only (inputs prepared on first call, then reused)."""
        self.qwen.reset_kv_cache()
        x, rot_global, rot_local, page_table_tt = self.prepare_layer_inputs(padded_tokens, seq_len=seq_len)
        layer = self.qwen.tt_model.layers[self.layer_idx]
        kv_slice = self.qwen.tt_kv_cache[self.layer_idx]

        prefill_ctx = (
            ace_step_qwen_prefill_l1_op_context() if ace_step_lm_prefill_l1_enabled() else contextlib.nullcontext()
        )
        with prefill_ctx:
            _ = layer(
                x,
                None,
                rot_mats_global=rot_global,
                rot_mats_local=rot_local,
                user_id=0,
                mode=Mode.PREFILL,
                page_table=page_table_tt,
                chunk_page_table=None,
                chunk_start_idx=None,
                kv_cache=kv_slice,
                batch_size=1,
            )


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_llm_decoder_layer_tracy_profile(device):
    """Profile one ``TransformerBlock`` prefill forward per iteration (``LLM_DECODER_LAYER_FORWARD``)."""
    ace_step_enable_tracy_profiler_env()
    assert_no_lm_trace()

    ckpt_root, variant = ckpt_paths_from_env()
    lm_dir = ensure_lm_checkpoint(ckpt_root, variant)
    _run_llm_decoder_layer_tracy_harness(device, lm_dir=lm_dir, variant_label=variant)


def _run_llm_decoder_layer_tracy_harness(device: ttnn.Device, *, lm_dir, variant_label: str) -> None:
    enable_llm_tracy_perf_env()

    layer_idx = int(os.environ.get("ACE_STEP_LLM_DECODER_LAYER_IDX", "0"))
    iters = max(1, int(os.environ.get("ACE_STEP_LLM_DECODER_LAYER_PERF_ITERS", "32")))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "4")))
    seed = int(os.environ.get("ACE_STEP_PERF_SEED", "0"))
    trace_each_iter = os.environ.get("ACE_STEP_TRACY_EACH_LLM_ITER", "").lower() in ("1", "true", "yes")
    try:
        flush_every = int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY", "1"))
    except ValueError:
        flush_every = 1

    prefill_seq = prefill_seq_len_from_env()

    cfg = AutoConfig.from_pretrained(str(lm_dir), trust_remote_code=True)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    if vocab <= 8:
        pytest.skip("invalid vocab_size from LM config")
    if n_layers <= 0:
        pytest.skip("invalid num_hidden_layers from LM config")
    if layer_idx >= n_layers:
        pytest.fail(f"ACE_STEP_LLM_DECODER_LAYER_IDX={layer_idx} >= num_hidden_layers={n_layers}")

    profiler = Profiler()
    profiler.clear()

    profiler.disable()
    profiler.start("ace_step_llm_decoder_layer_init", force_enable=True)
    tracy_signpost("LLM_DECODER_LAYER_INIT")

    max_seq_len = max(prefill_seq + 32, 1024)
    try:
        causal_lm = AceStepFiveHzExperimentalTtnnCausalLM(
            str(lm_dir),
            device,
            max_seq_len=int(max_seq_len),
        )
    except RuntimeError as exc:
        pytest.skip(f"TTNN causal LM init skipped: {exc}")

    layer_ctx = _SingleDecoderLayerPrefillCtx(causal_lm.qwen, layer_idx=layer_idx)
    prefill_ids = random_prefill_token_ids(vocab=vocab, seq_len=prefill_seq, seed=seed)
    padded_ids, seq_len, padded_len = pad_prefill_tokens(prefill_ids)

    profiler.end("ace_step_llm_decoder_layer_init", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    logger.info(
        "LLM decoder-layer Tracy setup (variant={}, layer={}/{}, seq={}/padded={}, iters={}, warmup={})",
        variant_label,
        layer_idx,
        n_layers,
        seq_len,
        padded_len,
        iters,
        warmup,
    )

    def _run_layer_once() -> None:
        tracy_signpost("LLM_DECODER_LAYER_FORWARD")
        layer_ctx.run_decoder_layer_forward(padded_ids, seq_len=seq_len)

    profiler.start("ace_step_llm_decoder_layer_compile", force_enable=True)
    tracy_signpost("LLM_DECODER_LAYER_COMPILE")
    _run_layer_once()
    profiler.end("ace_step_llm_decoder_layer_compile", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    profiler.start("ace_step_llm_decoder_layer_warmup", force_enable=True)
    tracy_signpost("LLM_DECODER_LAYER_WARMUP")
    for _ in range(warmup):
        _run_layer_once()
    profiler.end("ace_step_llm_decoder_layer_warmup", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    profiler.enable()
    profiler.start("ace_step_llm_decoder_layer_perf_pass")
    tracy_signpost("LLM_DECODER_LAYER_PERF_PASS")

    for iter_idx in range(iters):
        if trace_each_iter and not is_ci():
            tracy_signpost(f"LLM_DECODER_LAYER_ITER_{iter_idx}")
        _run_layer_once()
        if flush_every > 0 and (iter_idx + 1) % flush_every == 0:
            ace_step_flush_device_profiler(device)

    ttnn.synchronize_device(device)
    profiler.end("ace_step_llm_decoder_layer_perf_pass")
    ace_step_flush_device_profiler(device)

    if hasattr(causal_lm, "release_trace"):
        causal_lm.release_trace()
    del causal_lm, layer_ctx
    gc.collect()

    profiler.print()
    perf_wall = profiler.get("ace_step_llm_decoder_layer_perf_pass")
    per_iter_ms = (perf_wall * 1000.0 / max(1, iters)) if iters else 0.0
    logger.info(
        "AceStep LLM decoder-layer Tracy (variant={}, layer={}, seq={}, iters={}): "
        "init={:.3f}s compile={:.3f}s warmup({}x)={:.3f}s perf_pass={:.3f}s (~{:.2f}ms/layer)",
        variant_label,
        layer_idx,
        prefill_seq,
        iters,
        profiler.get("ace_step_llm_decoder_layer_init"),
        profiler.get("ace_step_llm_decoder_layer_compile"),
        warmup,
        profiler.get("ace_step_llm_decoder_layer_warmup"),
        perf_wall,
        per_iter_ms,
    )
    logger.info("Filter tt-perf-report on signpost LLM_DECODER_LAYER_FORWARD.")

    budget = os.environ.get("ACE_STEP_PERF_MAX_SECONDS")
    if budget:
        assert perf_wall <= float(
            budget
        ), f"ace_step_llm_decoder_layer_perf_pass {perf_wall}s exceeds ACE_STEP_PERF_MAX_SECONDS={budget}s"
