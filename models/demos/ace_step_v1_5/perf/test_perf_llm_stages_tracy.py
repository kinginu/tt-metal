# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf harness for ACE-Step 5 Hz LM **stages up to the transformer** (no decode loop).

Profiles three pipeline blocks in isolation or in sequence:

1. **Tokenizer** (host CPU): ``apply_chat_template`` + ``AutoTokenizer`` encode — production
   prompt formatting from ``LocalFiveHzLMHandler.build_formatted_prompt``.
2. **Embedding** (device): ``tt_transformers`` ``embed_tokens`` via ``Transformer.embd``.
3. **Transformer** (device): decoder prefill stack via ``ttnn_prefill_forward`` (all layers +
   final norm; stops before sampling / decode / LM-head-driven generation).

Does **not** profile decode steps, logits sampling, constrained decoding, or ``LMHead``-narrow
band selection as a generation loop — use ``test_perf_llm_tracy.py`` for end-to-end prefill+decode.

Run from the repository root (example):

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/demos/ace_step_v1_5/perf/test_perf_llm_stages_tracy.py::test_perf_ace_step_llm_stages_tracy_profile \\
        -v -s

**Important:** do **not** set ``ACE_STEP_USE_TRACE=1`` (Tracy device profiling and LM trace
capture are incompatible).

Optional environment variables:

- ``ACE_STEP_CKPT_DIR`` / ``ACE_STEP_LLM_PERF_VARIANT``: checkpoint root + LM folder
  (default ``acestep-5Hz-lm-1.7B``).
- ``ACE_STEP_PERF_NO_DOWNLOAD`` / ``ACE_STEP_PERF_DOWNLOAD``: Hugging Face fetch control.
- ``ACE_STEP_LLM_STAGES_PERF_MODE`` (default ``all``):
  - ``tokenizer``: host tokenization only (no device work in perf pass).
  - ``embedding``: TTNN ``embed_tokens`` only.
  - ``transformer``: prefill decoder forward only (requires init + compile).
  - ``all``: tokenizer → embedding → transformer each iteration (pipeline order).
- ``ACE_STEP_LLM_STAGES_PERF_ITERS`` (default ``3``): timed perf-pass iterations.
- ``ACE_STEP_PERF_WARMUP`` (default ``1``): warmup iterations before the timed pass.
- ``ACE_STEP_LLM_PERF_PROMPT`` / ``ACE_STEP_LLM_PERF_LYRICS``: caption / lyrics for tokenization.
- ``ACE_STEP_LLM_STAGES_GENERATION_PHASE`` (default ``cot``): ``cot`` or ``codes`` chat template.
- ``ACE_STEP_TRACY_EACH_LLM_ITER``: set to ``1`` for one Tracy signpost per perf iteration.
- ``ACE_STEP_PROFILER_FLUSH_EVERY``: flush device profiler every N perf iterations (default ``1``).
- ``ACE_STEP_LM_PREFILL_L1`` / ``ACE_STEP_LM_UNIFIED_DECODE_SHARD`` / etc.: same Tracy perf knobs
  as ``test_perf_llm_tracy.py`` (enabled by default in this harness).
"""

from __future__ import annotations

import contextlib
import gc
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.common.utility_functions import Profiler
from models.demos.ace_step_v1_5.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
from models.demos.ace_step_v1_5.ttnn_impl.five_hz_causal_lm_experimental import AceStepFiveHzExperimentalTtnnCausalLM
from models.demos.ace_step_v1_5.ttnn_impl.five_hz_lm.five_hz_lm_constants import DEFAULT_LM_INSTRUCTION
from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_enable_tracy_profiler_env,
    ace_step_flush_device_profiler,
    ace_step_lm_prefill_l1_enabled,
)
from models.demos.ace_step_v1_5.ttnn_impl.qwen_prefill_l1 import ace_step_qwen_prefill_l1_op_context
from models.tt_transformers.tt.common import get_padded_prefill_len, num_blocks_in_seq


def _is_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def _perf_download_disabled() -> bool:
    if os.environ.get("ACE_STEP_PERF_NO_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return True
    dl = os.environ.get("ACE_STEP_PERF_DOWNLOAD", "").lower()
    return dl in ("0", "false", "no")


def _resolve_lm_dir(ckpt_dir: Path, variant: str) -> Path:
    return (ckpt_dir / variant).resolve()


def _lm_checkpoint_ready(lm_dir: Path) -> bool:
    if not (lm_dir / "config.json").is_file():
        return False
    if (lm_dir / "model.safetensors").is_file():
        return True
    return bool(list(lm_dir.glob("model-*.safetensors")))


def _ensure_lm_checkpoint(ckpt_dir: Path, variant: str) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lm_dir = _resolve_lm_dir(ckpt_dir, variant)
    if _lm_checkpoint_ready(lm_dir):
        return lm_dir
    if _perf_download_disabled():
        pytest.skip(f"Missing 5 Hz LM under {lm_dir}. Unset ACE_STEP_PERF_NO_DOWNLOAD to fetch via Hugging Face.")
    pytest.importorskip("huggingface_hub")
    logger.info("ACE-Step LLM stages perf: fetching {} …", variant)
    try:
        _ensure_variant(variant, ckpt_dir)
    except Exception as exc:
        pytest.skip(f"Hugging Face download failed for {variant}: {exc}")
    if not _lm_checkpoint_ready(lm_dir):
        pytest.fail(f"Download finished but LM weights still missing under {lm_dir}")
    return lm_dir


def _tracy_signpost(label: str) -> None:
    if _is_ci():
        return
    try:
        from tracy import signpost  # type: ignore[import-untyped]
    except ImportError:
        return
    try:
        signpost(label)
    except Exception:
        pass


def _ace_step_enable_llm_tracy_perf_env() -> None:
    """Opt-in perf knobs for Tracy (PCC-safe defaults elsewhere)."""
    os.environ.setdefault("ACE_STEP_LM_PREFILL_L1", "1")
    os.environ.setdefault("ACE_STEP_LM_PREFILL_MLP_FF1_L1", "1")
    os.environ.setdefault("ACE_STEP_LM_UNIFIED_DECODE_SHARD", "1")
    os.environ.setdefault("ACE_STEP_LM_DECODE_QK_NORM_SHARDED", "1")
    os.environ.setdefault("ACE_STEP_LM_SDPA_GATHER_UNIFIED", "1")
    os.environ.setdefault("ACE_STEP_LM_NARROW_AUDIO_VOCAB", "1")


def _apply_chat_template(tokenizer: AutoTokenizer, *, caption: str, lyrics: str) -> str:
    body = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n"},
            {"role": "user", "content": body},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _tokenize_prompt(tokenizer: AutoTokenizer, formatted: str) -> torch.Tensor:
    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    return encoded["input_ids"]


def _pad_prefill_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    seq_len = int(tokens.shape[1])
    prefill_seq_len = int(get_padded_prefill_len(seq_len))
    if prefill_seq_len != seq_len:
        pad = torch.zeros(1, prefill_seq_len - seq_len, dtype=tokens.dtype, device=tokens.device)
        padded = torch.cat([tokens, pad], dim=-1)
    else:
        padded = tokens
    return padded, seq_len, prefill_seq_len


class _LMPrefillContext:
    """Cached prefill inputs for embedding / transformer stage loops."""

    def __init__(self, qwen: Any) -> None:
        self.qwen = qwen
        self._page_table_torch = qwen._page_table_torch.clone()
        self._paged_cfg = qwen._paged_cfg

    def prepare_transformer_inputs(self, padded_tokens: torch.Tensor, *, seq_len: int) -> tuple[Any, ...]:
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
        (
            prefill_input,
            rot_mats_global,
            rot_mats_local,
            page_table_tt,
            _chunk_page_table_tt,
        ) = inputs

        last_token_idx = seq_len - 1
        get_last_token = (last_token_idx // 32) * 32
        return (
            prefill_input,
            rot_mats_global,
            rot_mats_local,
            page_table_tt,
            get_last_token,
        )

    def run_embedding(self, padded_tokens: torch.Tensor) -> None:
        tt_model = self.qwen.tt_model
        flat = padded_tokens.reshape(1, 1, 1, -1)
        tokens_tt = ttnn.from_torch(
            flat,
            device=self.qwen.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.qwen.device),
        )
        embd = tt_model.embd(tokens_tt)
        _ = ttnn.unsqueeze_to_4D(embd)

    def run_transformer_prefill(self, padded_tokens: torch.Tensor, *, seq_len: int) -> None:
        self.qwen.reset_kv_cache()
        ctx = self.prepare_transformer_inputs(padded_tokens, seq_len=seq_len)
        prefill_input, rot_global, rot_local, page_table_tt, get_last_token = ctx

        prefill_ctx = (
            ace_step_qwen_prefill_l1_op_context() if ace_step_lm_prefill_l1_enabled() else contextlib.nullcontext()
        )
        with prefill_ctx:
            _ = self.qwen.tt_model.ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_global,
                rot_mats_local=rot_local,
                user_id=0,
                page_table=page_table_tt,
                chunk_page_table=None,
                chunk_start_idx=None,
                get_last_token=get_last_token,
                kv_cache=self.qwen.tt_kv_cache,
                batch_size=1,
            )


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_llm_stages_tracy_profile(device):
    """Profile tokenizer, embedding, and transformer prefill stages with Tracy signposts."""
    ace_step_enable_tracy_profiler_env()
    if os.environ.get("ACE_STEP_USE_TRACE", "").lower() in ("1", "true", "yes"):
        pytest.fail("ACE_STEP_USE_TRACE=1 is incompatible with Tracy device profiling for LLM stages.")

    ckpt_root = Path(os.environ.get("ACE_STEP_CKPT_DIR", str(_DEFAULT_CKPT_DIR))).expanduser().resolve()
    variant = os.environ.get("ACE_STEP_LLM_PERF_VARIANT", "acestep-5Hz-lm-1.7B")
    lm_dir = _ensure_lm_checkpoint(ckpt_root, variant)

    _run_llm_stages_tracy_harness(device, lm_dir=lm_dir, variant_label=variant)


def _run_llm_stages_tracy_harness(device: ttnn.Device, *, lm_dir: Path, variant_label: str) -> None:
    _ace_step_enable_llm_tracy_perf_env()

    stage_mode = os.environ.get("ACE_STEP_LLM_STAGES_PERF_MODE", "all").strip().lower()
    valid_modes = ("all", "tokenizer", "embedding", "transformer")
    if stage_mode not in valid_modes:
        pytest.fail(f"Unknown ACE_STEP_LLM_STAGES_PERF_MODE={stage_mode!r}; use one of {valid_modes}.")

    iters = max(1, int(os.environ.get("ACE_STEP_LLM_STAGES_PERF_ITERS", "3")))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "1")))
    trace_each_iter = os.environ.get("ACE_STEP_TRACY_EACH_LLM_ITER", "").lower() in ("1", "true", "yes")
    try:
        flush_every = int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY", "1"))
    except ValueError:
        flush_every = 1

    caption = os.environ.get("ACE_STEP_LLM_PERF_PROMPT", "lofi hip hop, warm vinyl, instrumental")
    lyrics = os.environ.get("ACE_STEP_LLM_PERF_LYRICS", "[Instrumental]")
    _ = os.environ.get("ACE_STEP_LLM_STAGES_GENERATION_PHASE", "cot")  # reserved for codes template

    needs_device = stage_mode in ("all", "embedding", "transformer")

    profiler = Profiler()
    profiler.clear()
    is_ci = _is_ci()

    # ------------------------------------------------------------------
    # Init: tokenizer (always) + TTNN causal LM (when device stages needed)
    # ------------------------------------------------------------------
    profiler.disable()
    profiler.start("ace_step_llm_stages_init", force_enable=True)
    _tracy_signpost("LLM_STAGES_INIT")

    tokenizer = AutoTokenizer.from_pretrained(str(lm_dir), use_fast=True)
    formatted_prompt = _apply_chat_template(tokenizer, caption=caption, lyrics=lyrics)
    sample_ids = _tokenize_prompt(tokenizer, formatted_prompt)
    padded_ids, seq_len, prefill_seq_len = _pad_prefill_tokens(sample_ids)

    causal_lm = None
    prefill_ctx_obj = None
    if needs_device:
        max_seq_len = max(prefill_seq_len + 32, 1024)
        try:
            causal_lm = AceStepFiveHzExperimentalTtnnCausalLM(
                str(lm_dir),
                device,
                max_seq_len=int(max_seq_len),
            )
        except RuntimeError as exc:
            pytest.skip(f"TTNN causal LM init skipped: {exc}")
        prefill_ctx_obj = _LMPrefillContext(causal_lm.qwen)

    profiler.end("ace_step_llm_stages_init", force_enable=True)
    if needs_device:
        ttnn.synchronize_device(device)
        ace_step_flush_device_profiler(device)

    logger.info(
        "LLM stages Tracy setup (mode={}, variant={}, seq_len={}, padded={}, phase={}): prompt_chars={}",
        stage_mode,
        variant_label,
        seq_len,
        prefill_seq_len,
        os.environ.get("ACE_STEP_LLM_STAGES_GENERATION_PHASE", "cot"),
        len(formatted_prompt),
    )

    def _run_tokenizer_once() -> tuple[torch.Tensor, int]:
        _tracy_signpost("LLM_STAGE_TOKENIZER")
        formatted = _apply_chat_template(tokenizer, caption=caption, lyrics=lyrics)
        ids = _tokenize_prompt(tokenizer, formatted)
        padded, real_len, _ = _pad_prefill_tokens(ids)
        return padded, real_len

    def _run_embedding_once(padded: torch.Tensor) -> None:
        assert prefill_ctx_obj is not None
        _tracy_signpost("LLM_STAGE_EMBEDDING")
        prefill_ctx_obj.run_embedding(padded)

    def _run_transformer_once(padded: torch.Tensor, *, real_seq_len: int) -> None:
        assert prefill_ctx_obj is not None
        _tracy_signpost("LLM_STAGE_TRANSFORMER")
        prefill_ctx_obj.run_transformer_prefill(padded, seq_len=real_seq_len)

    def _run_stages_once() -> None:
        if stage_mode == "tokenizer":
            _run_tokenizer_once()
            return

        if stage_mode == "embedding":
            if is_ci:
                padded, _ = padded_ids, seq_len
            else:
                padded, _ = _run_tokenizer_once()
            _run_embedding_once(padded)
            return

        if stage_mode == "transformer":
            if is_ci:
                padded, real_len = padded_ids, seq_len
            else:
                padded, real_len = _run_tokenizer_once()
            _run_transformer_once(padded, real_seq_len=real_len)
            return

        # all: pipeline order tokenizer → embedding → transformer
        padded, real_len = _run_tokenizer_once()
        _run_embedding_once(padded)
        _run_transformer_once(padded, real_seq_len=real_len)

    # ------------------------------------------------------------------
    # Compile pass
    # ------------------------------------------------------------------
    if needs_device:
        profiler.start("ace_step_llm_stages_compile", force_enable=True)
        _tracy_signpost("LLM_STAGES_COMPILE")
        _run_stages_once()
        profiler.end("ace_step_llm_stages_compile", force_enable=True)
        ttnn.synchronize_device(device)
        ace_step_flush_device_profiler(device)
    else:
        profiler.start("ace_step_llm_stages_compile", force_enable=True)
        _tracy_signpost("LLM_STAGES_COMPILE")
        _run_tokenizer_once()
        profiler.end("ace_step_llm_stages_compile", force_enable=True)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    profiler.start("ace_step_llm_stages_warmup", force_enable=True)
    _tracy_signpost("LLM_STAGES_WARMUP")
    for _ in range(warmup):
        _run_stages_once()
    profiler.end("ace_step_llm_stages_warmup", force_enable=True)
    if needs_device:
        ttnn.synchronize_device(device)
        ace_step_flush_device_profiler(device)

    # ------------------------------------------------------------------
    # Timed perf pass
    # ------------------------------------------------------------------
    profiler.enable()
    profiler.start("ace_step_llm_stages_perf_pass")
    _tracy_signpost("LLM_STAGES_PERF_PASS")

    for iter_idx in range(iters):
        if trace_each_iter and not is_ci:
            _tracy_signpost(f"LLM_STAGES_ITER_{iter_idx}")
        _run_stages_once()
        if needs_device and flush_every > 0 and (iter_idx + 1) % flush_every == 0:
            ace_step_flush_device_profiler(device)

    if needs_device:
        ttnn.synchronize_device(device)
    profiler.end("ace_step_llm_stages_perf_pass")
    if needs_device:
        ace_step_flush_device_profiler(device)

    if causal_lm is not None and hasattr(causal_lm, "release_trace"):
        causal_lm.release_trace()
    del causal_lm, tokenizer, prefill_ctx_obj
    gc.collect()

    _log_stages_profiler_summary(
        profiler=profiler,
        stage_mode=stage_mode,
        variant_label=variant_label,
        iters=iters,
        warmup=warmup,
        seq_len=seq_len,
        prefill_seq_len=prefill_seq_len,
    )


def _log_stages_profiler_summary(
    *,
    profiler: Profiler,
    stage_mode: str,
    variant_label: str,
    iters: int,
    warmup: int,
    seq_len: int,
    prefill_seq_len: int,
) -> None:
    profiler.print()
    init_wall = profiler.get("ace_step_llm_stages_init")
    compile_wall = profiler.get("ace_step_llm_stages_compile")
    warmup_wall = profiler.get("ace_step_llm_stages_warmup")
    perf_wall = profiler.get("ace_step_llm_stages_perf_pass")
    per_iter_ms = (perf_wall * 1000.0 / max(1, iters)) if iters else 0.0

    logger.info(
        "AceStep LLM stages Tracy (mode={}, variant={}, seq={}/padded={}, iters={}): "
        "init={:.3f}s compile={:.3f}s warmup({}x)={:.3f}s perf_pass={:.3f}s (~{:.1f}ms/iter)",
        stage_mode,
        variant_label,
        seq_len,
        prefill_seq_len,
        iters,
        init_wall,
        compile_wall,
        warmup,
        warmup_wall,
        perf_wall,
        per_iter_ms,
    )
    logger.info(
        "Tracy signposts: LLM_STAGE_TOKENIZER | LLM_STAGE_EMBEDDING | LLM_STAGE_TRANSFORMER — "
        "see generated/profiler/reports/<timestamp>/cpp_device_perf_report.csv"
    )

    budget = os.environ.get("ACE_STEP_PERF_MAX_SECONDS")
    if budget:
        assert perf_wall <= float(
            budget
        ), f"ace_step_llm_stages_perf_pass {perf_wall}s exceeds ACE_STEP_PERF_MAX_SECONDS={budget}s"
