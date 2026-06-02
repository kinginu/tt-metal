# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Program-config + memory-layout + fidelity sweep for a SINGLE matmul: M=128, K=2048, N=4096.

This is the 5 Hz LM **prefill QKV projection** (fused q+k+v).  Tracy reports it at HiFi4 / BF16
running ~109 us on 32 cores (8x4), with the auto-tuner flagging:

    128 x 2048 x 4096   HiFi4 BF16 x BF16 => BF16   109 us   32c   30.0% DRAM   19.7 TFLOPs (44%)
      - in0_block_w=1 is small, try in0_block_w=2 or above
      - out_subblock 1x1 is small, try out_subblock_h * out_subblock_w >= 2 if possible
      - HiFi2 may also work (discards the lowest activation bit; 2x the throughput of HiFi4)

So this sweep fixes the shape and sweeps the three levers those hints point at — **math fidelity /
weight dtype**, **in0_block_w (>=2)**, and **out_subblock (h*w>=2 where the layout allows)** — plus
program-config family, grid, and memory layout, and reads off the fastest variant that still
passes PCC vs an fp32 reference.

M=128 ⇒ Mt=4 — this changes the family set vs the M=32 (Mt=1) decode siblings:
  - The DRAM-bank-sharded weights family ("DS") is DROPPED — the kernel requires M==1 tile
    (matmul_device_operation.cpp:757), so it is decode-only.
  - 2D block-sharding ("2D") is ADDED — gy may divide Mt=4 (gy in {2,4}), so a real 2D grid
    exists (in0 height-sharded across rows, in1 mcast across cols).

Swept axes:
  - fidelity/dtype : ``(HiFi4, bf16)`` is the current default (the baseline).  ``(HiFi2, bf16)``
    keeps bf16 weights but halves the fidelity phases (the auto-tuner's suggestion); ``(HiFi2,
    bfp8)`` and ``(LoFi, bfp8)`` additionally drop weights to ``bfloat8_b`` (halves weight DRAM BW).
    Activations (in0) and output stay bf16 to match the model's prefill activation path.
  - family : "1D" = MatmulMultiCoreReuseMultiCast1DProgramConfig (mcast_in0).
             "2D" = MatmulMultiCoreReuseMultiCastProgramConfig (block-sharded).
  - grid (gx, gy), capped to the device ``compute_with_storage_grid_size()``.
  - in0_block_w in {2, 4, 8} (never 1 — the auto-tuner hint).
  - memory layout in0/out.  Codes: l1 = L1 interleaved, dram = DRAM interleaved,
    ws = L1 WIDTH_SHARDED.  Weights (in1) always stream from DRAM interleaved.

Note on width-sharded output: the QKV result feeds ``NlpCreateHeads`` (a TM op that wants
interleaved input), so a ``ws`` output would force a ``ShardedToInterleaved`` in the real model.
The ``ws`` rows are kept as exploration (matmul-only time); pin an ``l1``-out winner unless you
also rework the head-split to consume sharded.

Baseline (must-beat): the production config — ``(HiFi4, bf16)``, 1D ``l1/dram/l1``, grid 8x4
(32 cores), in0_block_w=8.  ``_BASELINE_US`` left None uses that row's own measured time; set it
to 109.0 to gate against the perf-report number directly.

Measurement: true device kernel time from the device profiler (``ttnn.ReadDeviceProfiler`` +
tracy ``import_log_run_stats``, ``device_kernel_duration`` averaged over the single timed op).
REQUIRES a profiler build — the test skips if ``TT_METAL_DEVICE_PROFILER`` is unset.  Each row:
one warmup (program-cache miss + JIT), clear the log, one timed run; each config in try/except so
an infeasible layout is recorded as an ERROR row instead of aborting the sweep.

Output: one aligned line per config (device us, TFLOPs, PCC, PASS/FAIL/ERROR), then a summary
line naming the fastest PCC-passing variant and its speedup vs the baseline.  The test FAILS if
no config both passes PCC and beats the baseline.

Run from the repository root **as plain pytest** (NOT under ``python -m tracy``)::

    TT_METAL_DEVICE_PROFILER=1 pytest \\
        models/demos/ace_step_v1_5/perf/test_perf_matmul_128x2048x4096_sweep_tracy.py::test_matmul_128x2048x4096_sweep \\
        -v -s

The test sets ``TT_METAL_PROFILER_MID_RUN_DUMP=1`` and ``TT_METAL_PROFILER_CPP_POST_PROCESS=1``
at import time (before ``ttnn`` loads) so ``ttnn.ReadDeviceProfiler`` writes
``profile_log_device.csv`` between configs.  Export them yourself only if another module imports
``ttnn`` before this file.

IMPORTANT — do **not** wrap this in ``python -m tracy -p -r``.  Tracy defers the device-CSV dump
to process exit, so mid-run ``ReadDeviceProfiler`` reads fail under the wrapper.
"""

import os
from dataclasses import dataclass

# Before MetalContext reads rtoptions (first ttnn import).
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
if os.getenv("TT_METAL_DEVICE_PROFILER"):
    os.environ.setdefault("TTNN_OP_PROFILER", "1")

import pytest
import torch
import tracy.device_post_proc_config as device_post_proc_config
from loguru import logger
from tracy.common import PROFILER_DEVICE_SIDE_LOG, PROFILER_LOGS_DIR, rm
from tracy.process_device_log import import_log_run_stats

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole

# --- fixed shape -----------------------------------------------------------
M = 128
K = 2048
N = 4096
TILE = 32
MT = M // TILE  # 4
KT = K // TILE  # 64
NT = N // TILE  # 128

# Must-beat device time [us].  None → use the measured baseline row below.
# (Perf report shows ~109 us for the HiFi4/bf16 32-core config.)
_BASELINE_US = None


# --- fidelity / dtype chain (the primary lever) ----------------------------
@dataclass(frozen=True)
class FidCase:
    label: str
    fidelity: "ttnn.MathFidelity"
    in1_dtype: "ttnn.DataType"  # weight dtype; activations (in0) + out stay bf16
    pcc_target: float
    is_baseline_fid: bool = False


def _fid_cases():
    return [
        FidCase("HiFi4/bf16", ttnn.MathFidelity.HiFi4, ttnn.bfloat16, 0.999, is_baseline_fid=True),
        FidCase("HiFi2/bf16", ttnn.MathFidelity.HiFi2, ttnn.bfloat16, 0.99),
        FidCase("HiFi2/bfp8", ttnn.MathFidelity.HiFi2, ttnn.bfloat8_b, 0.99),
        FidCase("LoFi/bfp8", ttnn.MathFidelity.LoFi, ttnn.bfloat8_b, 0.98),
    ]


# fp32 dest reg holds <=4 tiles when fp32_dest_acc_en=True, so out_subblock h*w<=4.
# Ordered by descending h*w so we pick the largest valid block (>=2 where possible).
_SUBBLOCK_CHOICES = [(2, 2), (4, 1), (1, 4), (3, 1), (1, 3), (2, 1), (1, 2), (1, 1)]


def _subblock(per_core_m, per_core_n, out_sharded):
    """Largest valid (out_subblock_h, out_subblock_w), h*w<=4 (fp32 dest)."""
    if out_sharded:  # sharded out requires out_subblock_h==1 — widen along N only
        for w in (4, 3, 2, 1):
            if per_core_n % w == 0:
                return 1, w
        return 1, 1
    for h, w in _SUBBLOCK_CHOICES:
        if per_core_m % h == 0 and per_core_n % w == 0:
            return h, w
    return 1, 1


@dataclass
class MatmulCase:
    fid: FidCase
    family: str  # "1D" | "2D"
    grid_x: int
    grid_y: int
    in0_block_w: int
    in0: str  # "l1" | "dram" | "ws"
    out: str  # "l1" | "dram" | "ws"
    is_baseline: bool = False

    @property
    def num_cores(self):
        return self.grid_x * self.grid_y

    @property
    def out_sharded(self):
        return self.out == "ws"

    @property
    def per_core_M(self):
        return MT if self.family == "1D" else MT // self.grid_y

    @property
    def per_core_N(self):
        cols = self.num_cores if self.family == "1D" else self.grid_x
        return NT // cols

    @property
    def kt_per_core(self):
        if self.in0 == "ws" and self.family == "1D":  # width-sharded in0 splits K across cores
            return KT // self.num_cores
        return KT

    @property
    def subblock(self):
        return _subblock(self.per_core_M, self.per_core_N, self.out_sharded)

    @property
    def layout(self):
        return f"{self.in0}/dram/{self.out}"

    @property
    def label(self):
        return f"{self.fid.label} {self.family} {self.layout} {self.grid_x}x{self.grid_y} w{self.in0_block_w}"

    def validate(self, gx_max, gy_max):
        """Raise AssertionError for an infeasible config (recorded as a skipped row)."""
        assert (
            self.grid_x <= gx_max and self.grid_y <= gy_max
        ), f"grid {self.grid_x}x{self.grid_y} > device {gx_max}x{gy_max}"
        assert KT % self.in0_block_w == 0, f"in0_block_w must divide Kt={KT}"
        assert self.kt_per_core % self.in0_block_w == 0, f"in0_block_w must divide Kt/core={self.kt_per_core}"
        if self.family == "1D":
            assert NT % self.num_cores == 0, f"{self.num_cores} cores must divide Nt={NT}"
            if self.in0 == "ws":
                assert KT % self.num_cores == 0, f"ws in0 needs cores | Kt={KT}"
            assert self.in0 in ("l1", "dram", "ws") and self.out in ("l1", "dram", "ws")
        else:  # 2D block-shard
            assert MT % self.grid_y == 0, f"gy={self.grid_y} must divide Mt={MT}"
            assert NT % self.grid_x == 0, f"gx={self.grid_x} must divide Nt={NT}"
            assert self.in0 in ("l1", "dram") and self.out in ("l1", "dram"), "2D uses interleaved in0/out here"


# 1D grids whose core count divides Nt=128, fitting an 8x8 grid (safe on WH + BH).
_GRIDS_1D = [(8, 8), (8, 4), (8, 2), (4, 4), (4, 2), (2, 2)]
# Blackhole L1 (~1.5 MiB/worker): 64-core l1/l1 for this shape always overflows CB budget.
_GRIDS_1D_BLACKHOLE = [(8, 4), (8, 2), (4, 4), (4, 2), (2, 2)]
# 2D grids: gy divides Mt=4 (gy in {2,4}), gx divides Nt=128.
_GRIDS_2D = [(8, 4), (8, 2), (4, 4), (4, 2), (2, 2)]
# Contrast grids for the exploratory in0=DRAM rows.
_GRIDS_CONTRAST = [(8, 4), (4, 2), (2, 2)]
# ws/ws rows: (grid, w) where kt_per_core = Kt/num_cores >= 2 and w | kt_per_core.
_WSWS = [(4, 4, 4), (4, 4, 2), (8, 2, 4), (8, 2, 2), (4, 2, 8), (8, 1, 8), (2, 2, 8)]


def _grids_1d():
    return _GRIDS_1D_BLACKHOLE if is_blackhole() else _GRIDS_1D


def _make_cases():
    cases = []
    fids = _fid_cases()
    grids_1d = _grids_1d()
    for fid in fids:
        # --- 1D, in0 L1-interleaved, out L1 (the production family) --------
        for gx, gy in grids_1d:
            for w in (2, 4, 8):
                is_base = fid.is_baseline_fid and gx == 8 and gy == 4 and w == 8
                cases.append(MatmulCase(fid, "1D", gx, gy, w, "l1", "l1", is_baseline=is_base))
        # --- 2D block-shard, interleaved in0/out ---------------------------
        for gx, gy in _GRIDS_2D:
            for w in (2, 4, 8):
                cases.append(MatmulCase(fid, "2D", gx, gy, w, "l1", "l1"))

    # Exploratory layouts only on the two bf16 fidelities (limit the matrix).
    for fid in fids[:2]:
        # 1D, out L1 width-sharded (matmul-only; real consumer would pay s2i)
        for gx, gy in grids_1d:
            for w in (2, 4, 8):
                cases.append(MatmulCase(fid, "1D", gx, gy, w, "l1", "ws"))
        # 1D, in0 streamed from DRAM instead of L1
        for gx, gy in _GRIDS_CONTRAST:
            cases.append(MatmulCase(fid, "1D", gx, gy, 8, "dram", "l1"))
        # 1D, in0 + out L1 width-sharded (splits K across cores)
        for gx, gy, w in _WSWS:
            cases.append(MatmulCase(fid, "1D", gx, gy, w, "ws", "ws"))
    return cases


_CASES = _make_cases()


@dataclass
class CaseResult:
    case: MatmulCase
    dev_us: float
    tflops: float
    pcc: float
    pcc_pass: bool
    err: str = ""


def _mem_in0(case):
    if case.in0 == "l1":
        return ttnn.L1_MEMORY_CONFIG
    if case.in0 == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.create_sharded_memory_config(
        (1, 1, M, K),
        core_grid=ttnn.CoreGrid(y=case.grid_y, x=case.grid_x),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def _mem_out(case):
    if case.out == "l1":
        return ttnn.L1_MEMORY_CONFIG
    if case.out == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.MemoryConfig(memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type=ttnn.BufferType.L1)


def _program_config(case):
    sh, sw = case.subblock
    if case.family == "2D":
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(case.grid_x, case.grid_y),
            in0_block_w=case.in0_block_w,
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=case.per_core_M,
            per_core_N=case.per_core_N,
            transpose_mcast=False,
            fused_activation=None,
        )
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(case.grid_x, case.grid_y),
        in0_block_w=case.in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=case.per_core_M,
        per_core_N=case.per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _compute_kernel_config(device, fid):
    init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
    if callable(init_ck):
        return init_ck(
            device.arch(),
            math_fidelity=fid.fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid.fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _read_device_kernel_us():
    """Device kernel duration [us] of the single op currently in the profiler log."""
    log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG
    if not log_path.exists():
        raise RuntimeError(
            f"device profiler log not found at {log_path}. "
            "Run as plain pytest with TT_METAL_DEVICE_PROFILER=1 — NOT under `python -m tracy`, "
            "which defers the device-CSV dump and breaks per-op ReadDeviceProfiler reads."
        )
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = log_path
    data = import_log_run_stats(setup)
    freq_mhz = data["deviceInfo"]["freq"]
    cycles = data["devices"][0]["cores"]["DEVICE"]["analysis"]["device_kernel_duration"]["stats"]["Average"]
    return cycles / freq_mhz  # cycles / MHz == us


def _require_profiler_env():
    missing = [
        name
        for name, val in (
            ("TT_METAL_DEVICE_PROFILER", "1"),
            ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
        )
        if os.environ.get(name) != val
    ]
    if missing:
        pytest.fail(
            "Device-time sweep needs a profiler build with mid-run CSV dumps. Set before pytest:\n"
            "  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 "
            "TT_METAL_PROFILER_CPP_POST_PROCESS=1"
        )


def test_matmul_128x2048x4096_sweep(device):
    if os.getenv("TT_METAL_DEVICE_PROFILER") is None:
        pytest.skip("device-time sweep needs a profiler build: set TT_METAL_DEVICE_PROFILER=1")
    _require_profiler_env()

    grid = device.compute_with_storage_grid_size()
    gx_max, gy_max = int(grid.x), int(grid.y)

    torch.manual_seed(0)
    profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG
    PROFILER_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    torch_a = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
    torch_out = torch.matmul(torch_a.float(), torch_b.float()).to(torch.bfloat16)

    results = []
    for case in _CASES:
        try:
            case.validate(gx_max, gy_max)
        except AssertionError as e:
            results.append(CaseResult(case, float("inf"), 0.0, 0.0, False, f"skip: {e}"))
            continue

        in0 = in1 = warm = timed = None
        try:
            in0 = ttnn.from_torch(
                torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_mem_in0(case)
            )
            in1 = ttnn.from_torch(
                torch_b,
                dtype=case.fid.in1_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            pc = _program_config(case)
            out_mc = _mem_out(case)
            ck = _compute_kernel_config(device, case.fid)

            # Warmup absorbs program-cache miss + JIT; clear the log so the timed run is alone.
            warm = ttnn.matmul(
                in0, in1, program_config=pc, memory_config=out_mc, compute_kernel_config=ck, dtype=ttnn.bfloat16
            )
            ttnn.synchronize_device(device)
            ttnn.deallocate(warm)
            warm = None
            ttnn.ReadDeviceProfiler(device)
            rm(profiler_log_path)
            ttnn.synchronize_device(device)

            timed = ttnn.matmul(
                in0, in1, program_config=pc, memory_config=out_mc, compute_kernel_config=ck, dtype=ttnn.bfloat16
            )
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            dev_us = _read_device_kernel_us()

            out = ttnn.to_torch(timed)
            assert out.shape == torch_out.shape, f"{out.shape} != {torch_out.shape}"
            pcc_pass, pcc = comp_pcc(torch_out, out, case.fid.pcc_target)
            tflops = 2 * M * K * N / 1e6 / dev_us
            results.append(CaseResult(case, dev_us, tflops, float(pcc), bool(pcc_pass)))
        except Exception as e:  # OOM / FATAL / shape mismatch — record, keep sweeping
            msg = str(e).strip().splitlines()[0][:90] if str(e).strip() else type(e).__name__
            results.append(CaseResult(case, float("inf"), 0.0, 0.0, False, msg))
        finally:
            for t in (timed, warm, in1, in0):
                if t is not None:
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
            ttnn.synchronize_device(device)

    # --- pick winners ------------------------------------------------------
    baseline = next(r for r in results if r.case.is_baseline)
    assert not baseline.err, f"baseline config {baseline.case.label} failed: {baseline.err}"
    must_beat_us = _BASELINE_US if _BASELINE_US is not None else baseline.dev_us
    passing = [r for r in results if r.pcc_pass]
    winners = [r for r in passing if r.dev_us < must_beat_us]
    best = min(passing, key=lambda r: r.dev_us) if passing else None

    # --- report ------------------------------------------------------------
    logger.info(f"matmul sweep M={M} K={K} N={N} (device grid {gx_max}x{gy_max}; in0=bf16, out=bf16)")
    logger.info(
        f"{'fidelity':>11} {'fam':>3} {'in0/in1/out':>12} {'grid':>5} {'cores':>5} {'ibw':>4} {'pcM':>3} {'pcN':>4} "
        f"{'sub':>5} {'dev_us':>9} {'TFLOPs':>7} {'PCC':>8} {'result':>7}  note"
    )
    for r in sorted(results, key=lambda r: (r.case.fid.label, r.dev_us)):
        c = r.case
        sh, sw = c.subblock
        if r.err:
            metrics = f"{'-':>9} {'-':>7} {'-':>8} {'ERROR':>7}"
            note = r.err
        else:
            metrics = f"{r.dev_us:>9.2f} {r.tflops:>7.2f} {r.pcc:>8.4f} {('PASS' if r.pcc_pass else 'FAIL'):>7}"
            note = "baseline" if c.is_baseline else ("best" if r is best else "")
        logger.info(
            f"{c.fid.label:>11} {c.family:>3} {c.layout:>12} {f'{c.grid_x}x{c.grid_y}':>5} {c.num_cores:>5} "
            f"{c.in0_block_w:>4} {c.per_core_M:>3} {c.per_core_N:>4} {f'{sh}x{sw}':>5} {metrics}  {note}"
        )

    if best is not None:
        logger.info(
            f"FASTEST PCC-PASS: {best.case.label} → {best.dev_us:.2f}us, {best.tflops:.2f} TFLOPs, PCC={best.pcc:.4f} "
            f"| must-beat={must_beat_us:.2f}us | speedup={must_beat_us / best.dev_us:.2f}x"
        )
    else:
        logger.info(f"FASTEST PCC-PASS: none reached its PCC target | must-beat={must_beat_us:.2f}us")

    assert winners, (
        f"No config both passed PCC and beat the baseline {must_beat_us:.2f}us "
        f"(baseline row {baseline.case.label}={baseline.dev_us:.2f}us). Baseline is already optimal among the swept configs."
    )
