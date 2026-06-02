# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for ACE-Step 5 Hz LM Tracy perf harnesses."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

from models.demos.ace_step_v1_5.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
from models.demos.ace_step_v1_5.ttnn_impl.five_hz_lm.five_hz_lm_constants import DEFAULT_LM_INSTRUCTION
from models.tt_transformers.tt.common import get_padded_prefill_len


def is_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def perf_download_disabled() -> bool:
    if os.environ.get("ACE_STEP_PERF_NO_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return True
    dl = os.environ.get("ACE_STEP_PERF_DOWNLOAD", "").lower()
    return dl in ("0", "false", "no")


def resolve_lm_dir(ckpt_dir: Path, variant: str) -> Path:
    return (ckpt_dir / variant).resolve()


def lm_checkpoint_ready(lm_dir: Path) -> bool:
    if not (lm_dir / "config.json").is_file():
        return False
    if (lm_dir / "model.safetensors").is_file():
        return True
    return bool(list(lm_dir.glob("model-*.safetensors")))


def ensure_lm_checkpoint(ckpt_dir: Path, variant: str) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lm_dir = resolve_lm_dir(ckpt_dir, variant)
    if lm_checkpoint_ready(lm_dir):
        return lm_dir
    if perf_download_disabled():
        pytest.skip(f"Missing 5 Hz LM under {lm_dir}. Unset ACE_STEP_PERF_NO_DOWNLOAD to fetch via Hugging Face.")
    pytest.importorskip("huggingface_hub")
    logger.info("ACE-Step LLM Tracy: fetching {} …", variant)
    try:
        _ensure_variant(variant, ckpt_dir)
    except Exception as exc:
        pytest.skip(f"Hugging Face download failed for {variant}: {exc}")
    if not lm_checkpoint_ready(lm_dir):
        pytest.fail(f"Download finished but LM weights still missing under {lm_dir}")
    return lm_dir


def tracy_signpost(label: str) -> None:
    if is_ci():
        return
    try:
        from tracy import signpost  # type: ignore[import-untyped]
    except ImportError:
        return
    try:
        signpost(label)
    except Exception:
        pass


def enable_llm_tracy_perf_env() -> None:
    """Opt-in perf knobs for Tracy (PCC-safe defaults elsewhere)."""
    os.environ.setdefault("ACE_STEP_LM_PREFILL_L1", "1")
    os.environ.setdefault("ACE_STEP_LM_PREFILL_MLP_FF1_L1", "1")
    os.environ.setdefault("ACE_STEP_LM_UNIFIED_DECODE_SHARD", "1")
    os.environ.setdefault("ACE_STEP_LM_DECODE_QK_NORM_SHARDED", "1")
    os.environ.setdefault("ACE_STEP_LM_SDPA_GATHER_UNIFIED", "1")
    os.environ.setdefault("ACE_STEP_LM_NARROW_AUDIO_VOCAB", "1")


def assert_no_lm_trace() -> None:
    if os.environ.get("ACE_STEP_USE_TRACE", "").lower() in ("1", "true", "yes"):
        pytest.fail("ACE_STEP_USE_TRACE=1 is incompatible with Tracy device profiling for LLM perf.")


def ckpt_paths_from_env() -> tuple[Path, str]:
    ckpt_root = Path(os.environ.get("ACE_STEP_CKPT_DIR", str(_DEFAULT_CKPT_DIR))).expanduser().resolve()
    variant = os.environ.get("ACE_STEP_LLM_PERF_VARIANT", "acestep-5Hz-lm-1.7B")
    return ckpt_root, variant


def apply_chat_template(tokenizer: AutoTokenizer, *, caption: str, lyrics: str) -> str:
    body = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n"},
            {"role": "user", "content": body},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_prompt(tokenizer: AutoTokenizer, formatted: str) -> torch.Tensor:
    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    return encoded["input_ids"]


def pad_prefill_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    seq_len = int(tokens.shape[1])
    prefill_seq_len = int(get_padded_prefill_len(seq_len))
    if prefill_seq_len != seq_len:
        pad = torch.zeros(1, prefill_seq_len - seq_len, dtype=tokens.dtype, device=tokens.device)
        padded = torch.cat([tokens, pad], dim=-1)
    else:
        padded = tokens
    return padded, seq_len, prefill_seq_len


def prefill_seq_len_from_env(default: int = 128) -> int:
    raw = os.environ.get("ACE_STEP_LLM_PREFILL_PERF_SEQ")
    if raw is None:
        raw = os.environ.get("ACE_STEP_LLM_DECODER_LAYER_PERF_SEQ")
    if raw is None:
        raw = os.environ.get("ACE_STEP_LLM_PERF_PREFILL_SEQ", str(default))
    return max(8, int(raw))


def random_prefill_token_ids(*, vocab: int, seq_len: int, seed: int) -> torch.Tensor:
    torch.manual_seed(int(seed))
    hi = min(8192, max(8, int(vocab) - 1))
    lo = max(4, min(8, hi - 1))
    return torch.randint(lo, hi, (1, int(seq_len)), dtype=torch.long)
