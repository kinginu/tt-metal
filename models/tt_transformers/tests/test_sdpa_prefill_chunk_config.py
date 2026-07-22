# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the L1-aware chunk cap in ``get_attn_sdpa_prefill_program_config``.

Prefill SDPA circular buffers scale with ``chunk * head_dim``, so the chunk size must
be capped to keep them within Blackhole L1. Before the cap, ``head_dim > 256`` (e.g.
MLA at ``head_dim=512``) selected a chunk of 256 that overflows L1 and throws at
program build. The boundaries here are the empirically-measured safe maxima
(``head_dim<=256`` tolerates chunk 256, ``head_dim=512`` tolerates chunk 128).

These tests exercise the real ``ModelArgs`` method with ``head_dim`` injected (the
method reads nothing else), so they need neither model weights nor a device.
"""

import pytest

from models.tt_transformers.tt.model_config import ModelArgs


def _prefill_chunks(head_dim, seq_len, chunk_start_idx=0):
    # The method only reads ``self.head_dim``; bypass the heavy __init__ to test it in isolation.
    args = ModelArgs.__new__(ModelArgs)
    args.head_dim = head_dim
    cfg = ModelArgs.get_attn_sdpa_prefill_program_config(args, seq_len=seq_len, chunk_start_idx=chunk_start_idx)
    return cfg.q_chunk_size, cfg.k_chunk_size


@pytest.mark.parametrize(
    "head_dim, seq_len, expected",
    [
        # head_dim <= 256: unchanged from the original heuristic (256 if seq_len >= 2048 else 64).
        (64, 4096, 256),
        (128, 4096, 256),
        (128, 1024, 64),
        (256, 4096, 256),
        (256, 1024, 64),
        # head_dim > 256: capped so chunk * head_dim fits L1.
        (512, 4096, 128),
        (512, 1024, 64),
        (576, 4096, 64),
    ],
)
def test_prefill_chunk_l1_cap(head_dim, seq_len, expected):
    q_chunk, k_chunk = _prefill_chunks(head_dim, seq_len)
    assert q_chunk == expected
    assert k_chunk == expected


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512, 576])
def test_prefill_chunk_within_l1_envelope(head_dim):
    max_safe = 256 if head_dim <= 256 else 128 if head_dim <= 512 else 64
    q_chunk, k_chunk = _prefill_chunks(head_dim, seq_len=4096)
    assert q_chunk <= max_safe
    assert k_chunk <= max_safe


@pytest.mark.parametrize("chunk_start_idx", [128, 256, 512, 1024])
def test_chunk_divides_chunk_start_idx(chunk_start_idx):
    # Chunked prefill requires the chunk size to divide chunk_start_idx; the L1 cap
    # must preserve this (both operands are powers of two).
    q_chunk, k_chunk = _prefill_chunks(512, seq_len=4096, chunk_start_idx=chunk_start_idx)
    assert chunk_start_idx % q_chunk == 0
    assert chunk_start_idx % k_chunk == 0
