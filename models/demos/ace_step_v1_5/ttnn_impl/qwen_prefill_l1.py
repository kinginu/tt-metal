# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE Qwen3 prefill: route activations through L1 (LoFi + ``bfloat8_b`` via :mod:`math_perf_env`).

``tt_transformers`` defaults prefill residuals and matmul ``in0`` to DRAM. Tracy reports large
``MinimalMatmulDeviceOperation`` / ``MatmulDeviceOperation (in0:dram_interleaved)`` buckets for
conditioning; this module patches the loaded Qwen stack so prefill uses ``L1_MEMORY_CONFIG``
without editing upstream ``tt_transformers`` sources.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import ttnn
from models.tt_transformers.tt.common import Mode

from .math_perf_env import (
    _LM_PREFILL_MLP_FF1_SWEEP_KN,
    ace_step_ensure_dram_activation,
    ace_step_linear_l1_memory_config,
    ace_step_lm_prefill_mlp_ff1_l1_activations_enabled,
    ace_step_lm_prefill_mlp_ff1_matmul_program_config,
    ace_step_lm_prefill_mlp_ff1_sweep_enabled,
    ace_step_lm_prefill_qkv_matmul_program_config,
    ace_step_lm_prefill_qkv_sweep_enabled,
    ace_step_lm_prefill_wo_matmul_program_config,
    ace_step_lm_prefill_wo_sweep_enabled,
)

# ModelArgs getters that return ``DRAM_MEMORY_CONFIG`` for ``Mode.PREFILL``.
_PREFILL_DRAM_MEMCFG_GETTERS: tuple[str, ...] = (
    "get_residual_mem_config",
    "get_mlp_input_mem_config",
    "get_mlp_ff1_3_mem_config",
    "get_mlp_ff2_mem_config",
    "get_mlp_output_mem_config",
    "get_attn_input_mem_config",
    "get_attn_qkv_mm_mem_config",
    "get_attn_qkv_all_reduce_output_mem_config",
    "get_attn_create_head_input_mem_config",
    "get_attn_create_head_output_mem_config",
    "get_attn_sdpa_output_mem_config",
    "get_attn_concat_heads_output_mem_config",
    "get_attn_all_gather_output_mem_config",
    "get_attn_wo_output_mem_config",
    "get_attn_dense_output_mem_config",
    "get_attn_all_reduce_output_mem_config",
    "get_attn_gather_users_mem_config",
)


def _is_dram_mc(mc: Any, dram_mc: Any) -> bool:
    return mc is not None and dram_mc is not None and mc == dram_mc


def _ensure_l1(ttnn_module: Any, tensor: Any, *, l1_mc: Any) -> Any:
    if tensor is None or l1_mc is None or not hasattr(ttnn_module, "to_memory_config"):
        return tensor
    if tensor.memory_config() == l1_mc:
        return tensor
    return ttnn_module.to_memory_config(tensor, l1_mc)


def _swap_dram_kwarg(kwargs: dict, *, dram_mc: Any, l1_mc: Any) -> None:
    mc = kwargs.get("memory_config")
    if _is_dram_mc(mc, dram_mc):
        kwargs["memory_config"] = l1_mc
    imc = kwargs.get("intermediate_memory_config")
    if _is_dram_mc(imc, dram_mc):
        kwargs["intermediate_memory_config"] = l1_mc


def _ensure_l1_arg(arg: Any, *, l1_mc: Any) -> Any:
    """Move a tensor or a list of tensors (e.g. ``ttnn.concat`` inputs) to L1."""
    if isinstance(arg, list):
        return [_ensure_l1(ttnn, t, l1_mc=l1_mc) for t in arg]
    return _ensure_l1(ttnn, arg, l1_mc=l1_mc)


def _ensure_l1_first_arg(args: tuple, *, l1_mc: Any) -> tuple:
    if not args:
        return args
    return (_ensure_l1_arg(args[0], l1_mc=l1_mc),) + args[1:]


def _ensure_dram_first_arg(args: tuple, *, dram_mc: Any) -> tuple:
    if not args:
        return args
    return (ace_step_ensure_dram_activation(ttnn, args[0], dram_mc),) + args[1:]


def _tensor_memory_layout(tensor: Any) -> Any | None:
    try:
        return tensor.memory_config().memory_layout
    except Exception:
        return None


def _promote_in0_to_l1_interleaved(tensor: Any, *, l1_mc: Any) -> Any:
    """Width-sharded SwiGLU output (FF2 in0) must be L1 interleaved for stock 2D matmul."""
    interleaved = ttnn.TensorMemoryLayout.INTERLEAVED
    if _tensor_memory_layout(tensor) == interleaved:
        return tensor
    try:
        if tensor.memory_config().is_sharded():
            return ttnn.sharded_to_interleaved(tensor, l1_mc)
    except Exception:
        pass
    return ttnn.to_memory_config(tensor, l1_mc)


def _is_mlp_ff2_down_linear_args(args: tuple) -> bool:
    """True for prefill down-proj: in0 [*,*,M,6144] x w2 [*,*,6144,2048]."""
    if len(args) < 2:
        return False
    a, b = args[0], args[1]
    if not (hasattr(a, "shape") and hasattr(b, "shape")):
        return False
    try:
        _, ff1_n = _LM_PREFILL_MLP_FF1_SWEEP_KN
        return len(a.shape) == 4 and len(b.shape) == 4 and int(a.shape[-1]) == ff1_n and int(b.shape[-2]) == ff1_n
    except Exception:
        return False


def _is_mlp_ff1_gate_up_linear_args(args: tuple) -> bool:
    """True for prefill gate/up: in0 [*,*,M,2048] x w1/w3 [*,*,2048,6144]."""
    if len(args) < 2:
        return False
    a, b = args[0], args[1]
    if not (hasattr(a, "shape") and hasattr(b, "shape")):
        return False
    try:
        k_dim, n_dim = _LM_PREFILL_MLP_FF1_SWEEP_KN
        return (
            len(a.shape) == 4
            and len(b.shape) >= 2
            and int(a.shape[-1]) == k_dim
            and int(b.shape[-2]) == k_dim
            and int(b.shape[-1]) == n_dim
        )
    except Exception:
        return False


def _is_stock_2d_mcast_program_config(program_config: Any) -> bool:
    cfg_cls = getattr(ttnn, "MatmulMultiCoreReuseMultiCastProgramConfig", None)
    return program_config is not None and cfg_cls is not None and isinstance(program_config, cfg_cls)


def _is_dram_sharded_linear_program_config(program_config: Any) -> bool:
    """LM head / decode DRAM-sharded matmul — must keep width-sharded in0 and caller memcfg."""
    for name in (
        "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
        "MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig",
    ):
        cfg_cls = getattr(ttnn, name, None)
        if cfg_cls is not None and isinstance(program_config, cfg_cls):
            return True
    return False


def _ensure_interleaved_if_sharded_in0(args: tuple, *, l1_mc: Any, program_config: Any) -> tuple:
    """Stock 2D prefill matmuls (e.g. FF2 after ws gate/up) require interleaved in0."""
    if _is_dram_sharded_linear_program_config(program_config):
        return args
    if not args:
        return args
    t = args[0]
    if not hasattr(t, "memory_config"):
        return args

    cfg_1d = getattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", None)
    if program_config is not None and cfg_1d is not None and isinstance(program_config, cfg_1d):
        if getattr(program_config, "mcast_in0", False) and getattr(program_config, "fuse_batch", False):
            return args

    interleaved = ttnn.TensorMemoryLayout.INTERLEAVED
    layout = _tensor_memory_layout(t)
    is_ff2 = _is_mlp_ff2_down_linear_args(args)
    if layout == interleaved and not is_ff2:
        return args

    if is_ff2 or _is_stock_2d_mcast_program_config(program_config) or layout != interleaved:
        try:
            if layout != interleaved or is_ff2:
                return (_promote_in0_to_l1_interleaved(t, l1_mc=l1_mc),) + args[1:]
        except Exception:
            pass
    return args


def _patch_lru_cached_getter(model_args: Any, name: str, *, dram_mc: Any, l1_mc: Any) -> None:
    original: Callable = getattr(model_args, name)

    if hasattr(original, "cache_clear"):
        original.cache_clear()

    @functools.wraps(original)
    def patched(*args, **kwargs):
        mode = args[0] if args else kwargs.get("mode")
        out = original(*args, **kwargs)
        if mode == Mode.PREFILL and _is_dram_mc(out, dram_mc):
            return l1_mc
        return out

    if hasattr(original, "cache_clear"):
        patched = functools.lru_cache(maxsize=None)(patched)  # type: ignore[assignment]

    setattr(model_args, name, patched)


def _patch_mlp_ff2_all_reduce_getter(model_args: Any, *, dram_mc: Any, l1_mc: Any) -> None:
    name = "get_mlp_ff2_all_reduce_mem_config"
    original: Callable = getattr(model_args, name)

    @functools.wraps(original)
    def patched(mode, tensor):
        out = original(mode, tensor)
        if mode == Mode.PREFILL and _is_dram_mc(out, dram_mc):
            return l1_mc
        return out

    setattr(model_args, name, patched)


def ace_step_patch_model_args_lm_prefill_qkv_matmul(model_args: Any, device: Any) -> None:
    """Replace ``get_attn_qkv_program_config`` prefill path with the swept 128×2048×4096 pin."""
    if not ace_step_lm_prefill_qkv_sweep_enabled():
        return
    name = "get_attn_qkv_program_config"
    if not hasattr(model_args, name):
        return

    original: Callable = getattr(model_args, name)
    if hasattr(original, "cache_clear"):
        original.cache_clear()

    hidden_dim = int(getattr(model_args, "dim", 0))
    qkv_dim = int(getattr(model_args, "qkv_size", 0))

    @functools.wraps(original)
    def patched(mode, seq_len: int = 1, prefetcher=None):
        if mode == Mode.PREFILL and prefetcher is None and int(seq_len) <= 128:
            pc = ace_step_lm_prefill_qkv_matmul_program_config(
                device,
                seq_len=int(seq_len),
                hidden_dim=hidden_dim,
                qkv_dim=qkv_dim,
            )
            if pc is not None:
                return pc
        return original(mode, seq_len, prefetcher)

    if hasattr(original, "cache_clear"):
        patched = functools.lru_cache(maxsize=None)(patched)  # type: ignore[assignment]

    setattr(model_args, name, patched)


def ace_step_patch_model_args_lm_prefill_wo_matmul(model_args: Any, device: Any) -> None:
    """Replace ``get_attn_wo_program_config`` prefill path with the swept 128×2048×2048 pin."""
    if not ace_step_lm_prefill_wo_sweep_enabled():
        return
    name = "get_attn_wo_program_config"
    if not hasattr(model_args, name):
        return

    original: Callable = getattr(model_args, name)
    if hasattr(original, "cache_clear"):
        original.cache_clear()

    num_devices = max(1, int(getattr(model_args, "num_devices", 1)))
    k_dim = int(getattr(model_args, "n_heads", 0)) * int(getattr(model_args, "head_dim", 0)) // num_devices
    n_dim = int(getattr(model_args, "dim", 0))

    @functools.wraps(original)
    def patched(mode, seq_len: int = 1, prefetcher=None):
        if mode == Mode.PREFILL and prefetcher is None and int(seq_len) <= 128:
            pc = ace_step_lm_prefill_wo_matmul_program_config(
                device,
                seq_len=int(seq_len),
                k_dim=k_dim,
                n_dim=n_dim,
            )
            if pc is not None:
                return pc
        return original(mode, seq_len, prefetcher)

    if hasattr(original, "cache_clear"):
        patched = functools.lru_cache(maxsize=None)(patched)  # type: ignore[assignment]

    setattr(model_args, name, patched)


def ace_step_patch_model_args_lm_prefill_mlp_ff1_matmul(model_args: Any, device: Any) -> None:
    """Replace ``get_mlp_ff1_3_prg_config`` prefill path with the swept 128×2048×6144 pin."""
    if not ace_step_lm_prefill_mlp_ff1_sweep_enabled():
        return
    name = "get_mlp_ff1_3_prg_config"
    if not hasattr(model_args, name):
        return

    original: Callable = getattr(model_args, name)
    if hasattr(original, "cache_clear"):
        original.cache_clear()

    cluster_shape = getattr(model_args, "cluster_shape", (1, 1))
    k_dim = int(getattr(model_args, "dim", 0)) // int(cluster_shape[0])
    n_dim = int(getattr(model_args, "hidden_dim", 0)) // int(cluster_shape[1])

    @functools.wraps(original)
    def patched(mode, seq_len: int = 1, prefetcher=None):
        if not ace_step_lm_prefill_mlp_ff1_l1_activations_enabled():
            return original(mode, seq_len, prefetcher)
        if mode == Mode.PREFILL and prefetcher is None and int(seq_len) <= 128:
            pc = ace_step_lm_prefill_mlp_ff1_matmul_program_config(
                device,
                seq_len=int(seq_len),
                k_dim=k_dim,
                n_dim=n_dim,
            )
            if pc is not None:
                return pc
        return original(mode, seq_len, prefetcher)

    if hasattr(original, "cache_clear"):
        patched = functools.lru_cache(maxsize=None)(patched)  # type: ignore[assignment]

    setattr(model_args, name, patched)


def _promote_attention_weight_to_dram_interleaved(weight: Any):
    dram = ttnn.DRAM_MEMORY_CONFIG
    interleaved = ttnn.TensorMemoryLayout.INTERLEAVED
    mc = weight.memory_config()
    if mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == interleaved:
        return weight
    device = weight.device()
    torch_w = ttnn.to_torch(weight)
    return ttnn.from_torch(
        torch_w,
        dtype=weight.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _layer_mlp_module(layer: Any) -> Any | None:
    """Return the decoder MLP module (``feed_forward`` in ``tt_transformers``)."""
    return getattr(layer, "feed_forward", None) or getattr(layer, "mlp", None)


def ace_step_promote_attention_wqkv_to_dram_interleaved(tt_model: Any) -> None:
    """Install DRAM-interleaved copies for swept prefill pins; keep sharded weights for decode."""
    if not ace_step_lm_prefill_qkv_sweep_enabled():
        return
    for layer in getattr(tt_model, "layers", ()):
        attn = getattr(layer, "attention", None)
        if attn is not None:
            if hasattr(attn, "wqkv") and getattr(attn, "wqkv_prefill_interleaved", None) is None:
                attn.wqkv_prefill_interleaved = _promote_attention_weight_to_dram_interleaved(attn.wqkv)
            if ace_step_lm_prefill_wo_sweep_enabled() and hasattr(attn, "wo"):
                if getattr(attn, "wo_prefill_interleaved", None) is None:
                    attn.wo_prefill_interleaved = _promote_attention_weight_to_dram_interleaved(attn.wo)
            _patch_attention_prefill_sweep_weights(attn)

        if ace_step_lm_prefill_mlp_ff1_sweep_enabled():
            mlp = _layer_mlp_module(layer)
            if mlp is None:
                continue
            if hasattr(mlp, "w1") and getattr(mlp, "w1_prefill_interleaved", None) is None:
                mlp.w1_prefill_interleaved = _promote_attention_weight_to_dram_interleaved(mlp.w1)
            if hasattr(mlp, "w3") and getattr(mlp, "w3_prefill_interleaved", None) is None:
                mlp.w3_prefill_interleaved = _promote_attention_weight_to_dram_interleaved(mlp.w3)
            _patch_mlp_prefill_sweep_weights(mlp)


def _patch_attention_prefill_sweep_weights(attn: Any) -> None:
    """Swap ``wqkv`` / ``wo`` to interleaved DRAM copies for prefill (seq_len <= 128); decode keeps sharded."""
    if getattr(attn, "_ace_step_prefill_sweep_patched", False):
        return
    wqkv_prefill = getattr(attn, "wqkv_prefill_interleaved", None)
    wqkv_decode = getattr(attn, "wqkv", None)
    wo_prefill = getattr(attn, "wo_prefill_interleaved", None)
    wo_decode = getattr(attn, "wo", None)
    orig_forward_prefill = attn.forward_prefill

    def forward_prefill(x_11SH, *args, **kwargs):
        seq_len = int(x_11SH.shape[-2])
        use_prefill_weights = seq_len <= 128
        if use_prefill_weights and wqkv_prefill is not None:
            attn.wqkv = wqkv_prefill
        if use_prefill_weights and wo_prefill is not None:
            attn.wo = wo_prefill
        try:
            return orig_forward_prefill(x_11SH, *args, **kwargs)
        finally:
            if use_prefill_weights and wqkv_decode is not None:
                attn.wqkv = wqkv_decode
            if use_prefill_weights and wo_decode is not None:
                attn.wo = wo_decode

    attn.forward_prefill = forward_prefill  # type: ignore[method-assign]
    attn._ace_step_prefill_sweep_patched = True


def _patch_mlp_prefill_sweep_weights(mlp: Any) -> None:
    """Swap ``w1`` / ``w3`` to interleaved DRAM copies for prefill (seq_len <= 128); decode keeps sharded."""
    if getattr(mlp, "_ace_step_prefill_sweep_patched", False):
        return
    w1_prefill = getattr(mlp, "w1_prefill_interleaved", None)
    w1_decode = getattr(mlp, "w1", None)
    w3_prefill = getattr(mlp, "w3_prefill_interleaved", None)
    w3_decode = getattr(mlp, "w3", None)
    orig_forward = mlp.forward

    def forward(x, mode, *args, **kwargs):
        seq_len = int(x.shape[-2])
        use_prefill_weights = mode == Mode.PREFILL and seq_len <= 128
        if use_prefill_weights and w1_prefill is not None:
            mlp.w1 = w1_prefill
        if use_prefill_weights and w3_prefill is not None:
            mlp.w3 = w3_prefill
        try:
            return orig_forward(x, mode, *args, **kwargs)
        finally:
            if use_prefill_weights and w1_decode is not None:
                mlp.w1 = w1_decode
            if use_prefill_weights and w3_decode is not None:
                mlp.w3 = w3_decode

    mlp.forward = forward  # type: ignore[method-assign]
    mlp._ace_step_prefill_sweep_patched = True


def _patch_attention_prefill_interleaved_wqkv(attn: Any) -> None:
    """Deprecated alias — use :func:`_patch_attention_prefill_sweep_weights`."""
    _patch_attention_prefill_sweep_weights(attn)


def _is_lm_prefill_wo_1d_mcast_program_config(program_config: Any) -> bool:
    """True for the swept WO pin (1D mcast, ``per_core_N=2``, ``out_subblock_h=1``)."""
    if not _is_lm_prefill_1d_mcast_program_config(program_config):
        return False
    return (
        int(getattr(program_config, "per_core_N", -1)) == 2 and int(getattr(program_config, "out_subblock_h", -1)) == 1
    )


def _is_lm_prefill_mlp_ff1_1d_mcast_program_config(program_config: Any) -> bool:
    """True for the swept MLP gate/up pin (1D mcast, ``in0_block_w=4``, ``per_core_N=6``, subblock 1×3)."""
    if not _is_lm_prefill_1d_mcast_program_config(program_config):
        return False
    return (
        int(getattr(program_config, "in0_block_w", -1)) == 4
        and int(getattr(program_config, "per_core_N", -1)) == 6
        and int(getattr(program_config, "out_subblock_h", -1)) == 1
        and int(getattr(program_config, "out_subblock_w", -1)) == 3
    )


def _is_lm_prefill_qkv_1d_mcast_program_config(program_config: Any) -> bool:
    """True for the swept QKV pin (1D mcast, ``per_core_N=4``, ``out_subblock_h=2``)."""
    if not _is_lm_prefill_1d_mcast_program_config(program_config):
        return False
    return int(getattr(program_config, "per_core_N", -1)) == 4


def _is_lm_prefill_1d_mcast_program_config(program_config: Any) -> bool:
    cfg_cls = getattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", None)
    if program_config is None or cfg_cls is None or not isinstance(program_config, cfg_cls):
        return False
    return bool(getattr(program_config, "mcast_in0", False))


def ace_step_patch_model_args_prefill_l1(model_args: Any) -> None:
    """Force L1 interleaved for all ``ModelArgs`` prefill activation memory getters."""
    l1_mc = ace_step_linear_l1_memory_config(ttnn)
    dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if l1_mc is None or dram_mc is None:
        return

    for name in _PREFILL_DRAM_MEMCFG_GETTERS:
        if hasattr(model_args, name):
            _patch_lru_cached_getter(model_args, name, dram_mc=dram_mc, l1_mc=l1_mc)

    if hasattr(model_args, "get_mlp_ff2_all_reduce_mem_config"):
        _patch_mlp_ff2_all_reduce_getter(model_args, dram_mc=dram_mc, l1_mc=l1_mc)


def _patch_distributed_norm_prefill_l1(norm_mod: Any, *, l1_mc: Any) -> None:
    """Prefill ``DistributedNorm`` uses DRAM for ``to_memory_config`` / gather — switch to L1."""
    orig_forward = norm_mod.forward

    def forward(x, mode: Mode, norm_config=None):
        if getattr(norm_mod, "TG", False) or mode != Mode.PREFILL:
            return orig_forward(x, mode, norm_config)

        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None
        input_mem_cfg = l1_mc

        if norm_mod.args.is_multichip and not norm_mod.args.is_distributed_norm(mode):
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=norm_mod.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=norm_mod.tt_ccl.get_num_links(1),
                topology=norm_mod.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=norm_mod.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
                subdevice_id=norm_mod.prefetcher.worker_sub_device_id if norm_mod.prefetcher is not None else None,
            )
        else:
            x = ttnn.to_memory_config(x, input_mem_cfg)

        x = norm_mod.norm(x, mode=mode, in_sharded=False, out_sharded=False, norm_config=norm_config)

        if norm_mod.args.is_distributed_norm(mode) and norm_mod.enable_all_gather:
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=norm_mod.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=norm_mod.tt_ccl.get_num_links(1),
                topology=norm_mod.args.ccl_topology(),
                memory_config=x.memory_config(),
                barrier_semaphore=norm_mod.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
        return x

    norm_mod.forward = forward  # type: ignore[method-assign]


def _patch_embd_l1_output(embd_module: Any, *, l1_mc: Any) -> None:
    """Prefill-only L1 embedding output; decode passes an explicit ``memory_config`` — leave it."""

    orig_forward = embd_module.forward

    def forward(x, memory_config=None):
        if memory_config is not None:
            return orig_forward(x, memory_config=memory_config)
        out = orig_forward(x, memory_config=l1_mc)
        return _ensure_l1(ttnn, out, l1_mc=l1_mc)

    embd_module.forward = forward  # type: ignore[method-assign]


def _patch_decoder_l1_residual(decoder_layer: Any, *, l1_mc: Any) -> None:
    orig_forward = decoder_layer.forward

    def forward(x, *args, mode="decode", **kwargs):
        if mode == Mode.PREFILL:
            x = _ensure_l1(ttnn, x, l1_mc=l1_mc)
        return orig_forward(x, *args, mode=mode, **kwargs)

    decoder_layer.forward = forward  # type: ignore[method-assign]


def ace_step_apply_qwen_prefill_l1(tt_model: Any, model_args: Any) -> None:
    """Patch a loaded ``tt_transformers`` Qwen model for L1 prefill activations."""
    l1_mc = ace_step_linear_l1_memory_config(ttnn)
    if l1_mc is None:
        return

    ace_step_patch_model_args_prefill_l1(model_args)

    _patch_embd_l1_output(tt_model.embd, l1_mc=l1_mc)
    for layer in tt_model.layers:
        _patch_decoder_l1_residual(layer, l1_mc=l1_mc)
        if hasattr(layer, "attention_norm"):
            _patch_distributed_norm_prefill_l1(layer.attention_norm, l1_mc=l1_mc)
        if hasattr(layer, "ff_norm"):
            _patch_distributed_norm_prefill_l1(layer.ff_norm, l1_mc=l1_mc)

    if hasattr(tt_model, "norm"):
        _patch_distributed_norm_prefill_l1(tt_model.norm, l1_mc=l1_mc)


@contextmanager
def ace_step_qwen_prefill_l1_op_context() -> Iterator[None]:
    """During Qwen prefill, redirect hard-coded ``DRAM_MEMORY_CONFIG`` TTNN calls to L1."""
    l1_mc = ace_step_linear_l1_memory_config(ttnn)
    dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if l1_mc is None or dram_mc is None:
        yield
        return

    saved: dict[str, Any] = {}
    experimental = getattr(ttnn, "experimental", None)
    tile_layout = getattr(ttnn, "TILE_LAYOUT", None)

    def _wrap_l1_compute(name: str, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            if name.endswith("minimal_matmul") and kwargs.get("memory_config") is None:
                kwargs["memory_config"] = l1_mc
            return fn(*args, **kwargs)

        return wrapper

    def _wrap_l1_unary(name: str, fn: Callable) -> Callable:
        """Matmul/SDPA/concat paths: L1 ``in0`` + L1 output when callers pass DRAM defaults."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            args = _ensure_l1_first_arg(args, l1_mc=l1_mc)
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            if kwargs.get("memory_config") is None:
                kwargs["memory_config"] = l1_mc
            out = fn(*args, **kwargs)
            return _ensure_l1(ttnn, out, l1_mc=l1_mc)

        return wrapper

    def _wrap_to_layout(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(tensor, layout, *args, **kwargs):
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            out = fn(tensor, layout, *args, **kwargs)
            if tile_layout is not None and layout == tile_layout:
                out = _ensure_l1(ttnn, out, l1_mc=l1_mc)
            return out

        return wrapper

    def _wrap_l1_linear(fn: Callable) -> Callable:
        """QKV: L1 in0 + L1 interleaved out; WO/MLP ff1: L1 in0 + L1 width-sharded out (WO then s2i)."""
        mlp_ff1_ws_mc = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            pc = kwargs.get("program_config")
            ws_mc = getattr(ttnn, "L1_WIDTH_SHARDED_MEMORY_CONFIG", None)
            if _is_lm_prefill_qkv_1d_mcast_program_config(pc):
                args = _ensure_l1_first_arg(args, l1_mc=l1_mc)
                _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            elif _is_lm_prefill_mlp_ff1_1d_mcast_program_config(pc) or _is_mlp_ff1_gate_up_linear_args(args):
                if ace_step_lm_prefill_mlp_ff1_l1_activations_enabled():
                    args = _ensure_l1_first_arg(args, l1_mc=l1_mc)
                    kwargs["memory_config"] = mlp_ff1_ws_mc
                else:
                    args = _ensure_dram_first_arg(args, dram_mc=dram_mc)
                    if dram_mc is not None:
                        kwargs["memory_config"] = dram_mc
            elif _is_lm_prefill_wo_1d_mcast_program_config(pc):
                args = _ensure_l1_first_arg(args, l1_mc=l1_mc)
                if ws_mc is not None:
                    kwargs["memory_config"] = ws_mc
            elif _is_dram_sharded_linear_program_config(pc):
                pass
            else:
                args = _ensure_interleaved_if_sharded_in0(args, l1_mc=l1_mc, program_config=pc)
                _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            out = fn(*args, **kwargs)
            if _is_lm_prefill_wo_1d_mcast_program_config(pc) and hasattr(out, "memory_config"):
                try:
                    if out.memory_config().is_sharded():
                        out = ttnn.sharded_to_interleaved(out, l1_mc)
                except Exception:
                    pass
            return out

        return wrapper

    if hasattr(ttnn, "linear"):
        saved["ttnn.linear"] = ttnn.linear
        ttnn.linear = _wrap_l1_linear(ttnn.linear)

    for op_name in ("add", "mul", "multiply", "pad", "reshape", "permute", "typecast"):
        if hasattr(ttnn, op_name):
            saved[f"ttnn.{op_name}"] = getattr(ttnn, op_name)
            setattr(ttnn, op_name, _wrap_l1_compute(op_name, getattr(ttnn, op_name)))

    if hasattr(ttnn, "embedding"):
        _orig_embedding = ttnn.embedding
        saved["ttnn.embedding"] = _orig_embedding

        @functools.wraps(_orig_embedding)
        def _embedding(*args, **kwargs):
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            if kwargs.get("memory_config") is None:
                kwargs["memory_config"] = l1_mc
            return _orig_embedding(*args, **kwargs)

        ttnn.embedding = _embedding

    if hasattr(ttnn, "to_layout"):
        saved["ttnn.to_layout"] = ttnn.to_layout
        ttnn.to_layout = _wrap_to_layout(ttnn.to_layout)

    if hasattr(ttnn, "concat"):
        saved["ttnn.concat"] = ttnn.concat
        ttnn.concat = _wrap_l1_unary("concat", ttnn.concat)

    transformer = getattr(ttnn, "transformer", None)
    if transformer is not None:
        for op_name in ("scaled_dot_product_attention", "chunked_scaled_dot_product_attention"):
            if hasattr(transformer, op_name):
                key = f"transformer.{op_name}"
                saved[key] = getattr(transformer, op_name)
                setattr(transformer, op_name, _wrap_l1_unary(op_name, getattr(transformer, op_name)))

    if experimental is not None:
        for op_name in ("minimal_matmul", "nlp_create_qkv_heads", "all_gather_async"):
            if hasattr(experimental, op_name):
                key = f"experimental.{op_name}"
                saved[key] = getattr(experimental, op_name)
                setattr(experimental, op_name, _wrap_l1_compute(op_name, getattr(experimental, op_name)))
        if hasattr(experimental, "nlp_concat_heads"):
            saved["experimental.nlp_concat_heads"] = experimental.nlp_concat_heads
            experimental.nlp_concat_heads = _wrap_l1_unary("nlp_concat_heads", experimental.nlp_concat_heads)

    tt_ccl_mod = None
    try:
        from models.tt_transformers.tt import ccl as tt_ccl_mod

        if hasattr(tt_ccl_mod, "tt_all_reduce"):
            saved["tt_all_reduce"] = tt_ccl_mod.tt_all_reduce
            tt_ccl_mod.tt_all_reduce = _wrap_l1_compute("tt_all_reduce", tt_ccl_mod.tt_all_reduce)
    except ImportError:
        tt_ccl_mod = None

    try:
        yield
    finally:
        for key, fn in saved.items():
            if key.startswith("ttnn."):
                setattr(ttnn, key.split(".", 1)[1], fn)
            elif key.startswith("experimental.") and experimental is not None:
                setattr(experimental, key.split(".", 1)[1], fn)
            elif key.startswith("transformer.") and transformer is not None:
                setattr(transformer, key.split(".", 1)[1], fn)
            elif key == "tt_all_reduce" and tt_ccl_mod is not None:
                tt_ccl_mod.tt_all_reduce = fn


__all__ = [
    "ace_step_apply_qwen_prefill_l1",
    "ace_step_patch_model_args_lm_prefill_mlp_ff1_matmul",
    "ace_step_patch_model_args_lm_prefill_qkv_matmul",
    "ace_step_patch_model_args_lm_prefill_wo_matmul",
    "ace_step_patch_model_args_prefill_l1",
    "ace_step_promote_attention_wqkv_to_dram_interleaved",
    "ace_step_qwen_prefill_l1_op_context",
]
