# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf harness: **one 5 Hz LM prefill forward** (full decoder stack + final norm/LMHead slice).

Profiles a single ``AceStepFiveHzExperimentalTtnnCausalLM.forward(..., past_key_values=None)``
call per timed iteration — embed + 28 decoder layers + output norm + sharded ``LMHead`` on the
last token tile (same path as production prefill, no decode loop).

Run from the repository root (example):

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/demos/ace_step_v1_5/perf/test_perf_llm_prefill_forward_tracy.py::test_perf_ace_step_llm_prefill_forward_tracy_profile \\
        -v -s

Filter ``tt-perf-report`` on signpost ``LLM_PREFILL_FORWARD`` for one prefill only.

**Important:** do **not** set ``ACE_STEP_USE_TRACE=1``.

Optional environment variables:

- ``ACE_STEP_CKPT_DIR`` / ``ACE_STEP_LLM_PERF_VARIANT``: checkpoint root + LM folder.
- ``ACE_STEP_PERF_NO_DOWNLOAD`` / ``ACE_STEP_PERF_DOWNLOAD``: Hugging Face fetch control.
- ``ACE_STEP_LLM_PREFILL_PERF_ITERS`` (default ``3``): timed perf-pass iterations.
- ``ACE_STEP_PERF_WARMUP`` (default ``1``): warmup prefill forwards before the timed pass.
- ``ACE_STEP_LLM_PREFILL_PERF_SEQ`` (default ``128``): tokenized prompt length (tile-padded prefill).
- ``ACE_STEP_PERF_SEED`` (default ``0``): RNG seed for synthetic token ids.
- ``ACE_STEP_TRACY_EACH_LLM_ITER``: set to ``1`` for one Tracy signpost per perf iteration.
- ``ACE_STEP_PROFILER_FLUSH_EVERY``: flush device profiler every N perf iterations (default ``1``).
- Tracy perf knobs: ``ACE_STEP_LM_PREFILL_L1``, ``ACE_STEP_LM_UNIFIED_DECODE_SHARD``, etc.
  (enabled by default via :func:`enable_llm_tracy_perf_env`).
"""

from __future__ import annotations

import gc
import os

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
    prefill_seq_len_from_env,
    random_prefill_token_ids,
    tracy_signpost,
)
from models.demos.ace_step_v1_5.ttnn_impl.five_hz_causal_lm_experimental import AceStepFiveHzExperimentalTtnnCausalLM
from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_enable_tracy_profiler_env,
    ace_step_flush_device_profiler,
)


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_llm_prefill_forward_tracy_profile(device):
    """Profile one TTNN 5 Hz LM prefill forward per iteration (Tracy signpost ``LLM_PREFILL_FORWARD``)."""
    ace_step_enable_tracy_profiler_env()
    assert_no_lm_trace()

    ckpt_root, variant = ckpt_paths_from_env()
    lm_dir = ensure_lm_checkpoint(ckpt_root, variant)
    _run_llm_prefill_forward_tracy_harness(device, lm_dir=lm_dir, variant_label=variant)


def _run_llm_prefill_forward_tracy_harness(device: ttnn.Device, *, lm_dir, variant_label: str) -> None:
    enable_llm_tracy_perf_env()

    iters = max(1, int(os.environ.get("ACE_STEP_LLM_PREFILL_PERF_ITERS", "3")))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "1")))
    seed = int(os.environ.get("ACE_STEP_PERF_SEED", "0"))
    trace_each_iter = os.environ.get("ACE_STEP_TRACY_EACH_LLM_ITER", "").lower() in ("1", "true", "yes")
    try:
        flush_every = int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY", "1"))
    except ValueError:
        flush_every = 1

    prefill_seq = prefill_seq_len_from_env()

    cfg = AutoConfig.from_pretrained(str(lm_dir), trust_remote_code=True)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    if vocab <= 8:
        pytest.skip("invalid vocab_size from LM config")

    profiler = Profiler()
    profiler.clear()

    profiler.disable()
    profiler.start("ace_step_llm_prefill_init", force_enable=True)
    tracy_signpost("LLM_PREFILL_INIT")

    max_seq_len = max(prefill_seq + 32, 1024)
    try:
        causal_lm = AceStepFiveHzExperimentalTtnnCausalLM(
            str(lm_dir),
            device,
            max_seq_len=int(max_seq_len),
        )
    except RuntimeError as exc:
        pytest.skip(f"TTNN causal LM init skipped: {exc}")

    prefill_ids = random_prefill_token_ids(vocab=vocab, seq_len=prefill_seq, seed=seed)

    profiler.end("ace_step_llm_prefill_init", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    logger.info(
        "LLM prefill Tracy setup (variant={}, seq_len={}, iters={}, warmup={})",
        variant_label,
        prefill_seq,
        iters,
        warmup,
    )

    def _run_prefill_once() -> None:
        tracy_signpost("LLM_PREFILL_FORWARD")
        causal_lm.reset_decode_state()
        with torch.inference_mode():
            causal_lm.forward(input_ids=prefill_ids.clone(), past_key_values=None, use_cache=True)

    profiler.start("ace_step_llm_prefill_compile", force_enable=True)
    tracy_signpost("LLM_PREFILL_COMPILE")
    _run_prefill_once()
    profiler.end("ace_step_llm_prefill_compile", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    profiler.start("ace_step_llm_prefill_warmup", force_enable=True)
    tracy_signpost("LLM_PREFILL_WARMUP")
    for _ in range(warmup):
        _run_prefill_once()
    profiler.end("ace_step_llm_prefill_warmup", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    profiler.enable()
    profiler.start("ace_step_llm_prefill_perf_pass")
    tracy_signpost("LLM_PREFILL_PERF_PASS")

    for iter_idx in range(iters):
        if trace_each_iter and not is_ci():
            tracy_signpost(f"LLM_PREFILL_ITER_{iter_idx}")
        _run_prefill_once()
        if flush_every > 0 and (iter_idx + 1) % flush_every == 0:
            ace_step_flush_device_profiler(device)

    ttnn.synchronize_device(device)
    profiler.end("ace_step_llm_prefill_perf_pass")
    ace_step_flush_device_profiler(device)

    if hasattr(causal_lm, "release_trace"):
        causal_lm.release_trace()
    del causal_lm
    gc.collect()

    profiler.print()
    perf_wall = profiler.get("ace_step_llm_prefill_perf_pass")
    per_iter_ms = (perf_wall * 1000.0 / max(1, iters)) if iters else 0.0
    logger.info(
        "AceStep LLM prefill Tracy (variant={}, seq={}, iters={}): "
        "init={:.3f}s compile={:.3f}s warmup({}x)={:.3f}s perf_pass={:.3f}s (~{:.1f}ms/prefill)",
        variant_label,
        prefill_seq,
        iters,
        profiler.get("ace_step_llm_prefill_init"),
        profiler.get("ace_step_llm_prefill_compile"),
        warmup,
        profiler.get("ace_step_llm_prefill_warmup"),
        perf_wall,
        per_iter_ms,
    )
    logger.info("Filter tt-perf-report on signpost LLM_PREFILL_FORWARD for one prefill forward.")

    budget = os.environ.get("ACE_STEP_PERF_MAX_SECONDS")
    if budget:
        assert perf_wall <= float(
            budget
        ), f"ace_step_llm_prefill_perf_pass {perf_wall}s exceeds ACE_STEP_PERF_MAX_SECONDS={budget}s"
