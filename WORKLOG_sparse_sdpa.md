# Work log — `sparse_sdpa` implementation

Goal: **Stage 1 (H=32, full `{32,32}` tiles) working end-to-end, PCC ≥ 0.99 vs `sparse_mla` golden.**
Branch: `pjosipovic/sparse_mla_prefill_ref`. Plan: `PLAN_sparse_sdpa.md`. Golden: `models/demos/deepseek_v32/reference_cpu/sparse_sdpa_prefill.py::sparse_mla`.

## Strategy decision (v0)
Implement the **`compute_common` materialized path first** (not the streaming `kt_inplace_v` path) to reach a passing result fastest, then optimize:
- tilize K → `k_tiles[Skt,DHt]`; **transpose** → `kt_tiles[DHt,Skt]` (Kᵀ, for QK); **copy compact V** → `v_tiles[Skt,vDHt]` (rope cols dropped, for PV).
- QK = `matmul_blocks(q_tiles, kt_tiles, transpose=false)` → `qk` (kt is already Kᵀ, standard NN).
- mask add = plain `add_block_inplace(qk, mask)` — **valid here** because `qk` is a normal reserved/pushed CB (the hold-wr-ptr problem is specific to the streaming `cb_qkt_im`; we're not on that path).
- softmax = `reduce_c<MAX,REDUCE_ROW>` → `sub_exp_block_bcast_cols_inplace` → `matmul_reduce(col_identity,sum)` → `recip_block_inplace` → `mul_block_bcast_cols`.
- PV = `matmul_blocks(probs, v_tiles, transpose=false)` → `out`.
- untilize `out` → row-major; writer writes H rows.

Rationale: separate Kᵀ/V buffers ⇒ standard `matmul_blocks` (pops its in1 fine), no custom MOPs, no hold-wr-ptr/mask-stamp subtleties. The streaming overlap path (`PLAN §6.3`) is the perf follow-up. (Heavier L1: `k_tiles` + `kt_tiles` + `v_tiles` — may force small `k_chunk_size`.)

## Phases (per PLAN §8)
1. Scaffold + identity (op callable, compiles, registers)
2. Reader gather (debug dump K)
3. Tilize/untilize round-trip
4. QK + masked softmax (intermediate check)
5. PV + normalize → full op, PCC
6. Sweep k_chunk + edge cases

## Log
- (start) Plan committed `788593b564f`. Reference impl committed `d48f5da128e`.
- **Phase 1 DONE ✅** — scaffold builds + registers + runs e2e on BH, `test_sparse_sdpa_phase1_runs` PASS.
  - Files: `device/sparse_sdpa_device_operation_types.hpp`, `.../sparse_sdpa_device_operation.{hpp,cpp}`, `.../sparse_sdpa_program_factory.cpp`, `sparse_sdpa.{hpp,cpp}`, kernels `dataflow/sparse_sdpa_{reader,writer}.cpp` + `compute/sparse_sdpa_compute.cpp`; wired `sources.cmake`, `CMakeLists.txt` (FILE_SET api), `sdpa_nanobind.cpp`.
  - Validated: ProgramDescriptor new-infra (all-static, `prim::sparse_sdpa`→`launch<>`); validate (BH gate, H==32, ROW_MAJOR/DRAM, padded==logical, TOPK≤2048, TOPK%k_chunk==0, fp32_dest_acc_en==false via `get_compute_kernel_config_args`); ROW_MAJOR output spec; per-core token split; RM stick writer (`TensorAccessor(args,addr).get_noc_addr(h*S+t)`, page=row).
  - Gotchas hit: BH config accessor is `get_compute_kernel_config_args` (no `BlackholeComputeKernelConfig`); BH-gate decorator is `models.common.utility_functions.run_for_blackhole` (NOT `models.utility_functions`). Two unused-var warnings in program_factory (`q_row_bytes`/`idx_row_bytes`) — consumed in phase 2.
- **Phases 2–5 (single-chunk) DONE ✅** — reader gather+mask, compute tilize/Kᵀ-reposition/QK/masked-softmax/PV/normalize/untilize, writer-from-compute all wired; `test_sparse_sdpa_pcc_single_chunk` **9/9 PASS** (PCC ≥ 0.99) vs `sparse_mla`, across (S,T,TOPK)∈{(64,256,32),(64,512,64),(32,1024,128)} × {all_valid,few_valid,boundary}.
  - Compile fix: compute kernel needs `compute_common.hpp` defines — added `STATS/SUB_EXP/MUL_BCAST/DHT/REDUCE_GRANULARITY=1` + `EXP_APPROX_MODE` to `compute_desc.defines` (granularity=1 is always-correct: it divides any dim).
  - **Root-cause bug (Skt≥2 only): `pack_tile` out-of-order index ignored without `<true>`.** The Kᵀ grid-reposition packed with bare `pack_tile(0, CB_KT, d*Skt+sk)`; the explicit `output_tile_index` is honored ONLY with `pack_tile<true>(...)` (template `out_of_order_output`, default false → sequential pack). Bare pack wrote sequentially in `sk`-outer/`d`-inner order = natural K `[Skt,DHt]`, but the matmul reads in1 as `[DHt,Skt]` (`in1_index += N`). Coincides at Skt=1 (KT==K), scrambles at Skt≥2. Symptom: Skt=1 perfect, Skt≥2 PCC 0.39–0.85, output norm correct (denominator fine) but values scrambled, and even tile-0-only corrupted. Fix: `pack_tile<true>` for both CB_KT and CB_V packs. (CB_V happened to be correct anyway — its intended index `sk*vDHt+d` matched the sequential order.) Diagnosis via numeric probes (`/tmp/dbg*.py`): norm match + tile-isolation + the matmul `in1_index += N` convention. **Lesson: any out-of-order `pack_tile(dst, cb, idx)` MUST be `pack_tile<true>`.**
  - Known latent (not yet hit): CB_QK_IM/CB_MAX/CB_SUM/CB_OUT_IM are never popped, so >1 token/core would overflow — fine for S≤num_cores (≈1 tok/core) but must be addressed for production S. And the chunk loop does NOT flash-accumulate yet (overwrites max/sum/out per chunk) → multi-chunk (`test_sparse_sdpa_pcc`) still TODO.
- **Multi-chunk flash + multi-token DONE ✅ — Stage 1 COMPLETE.** Full suite **22/22 PASS** (phase1×1, single_chunk×9, multi-chunk×9, multitoken×3), all PCC ≥ 0.99 vs `sparse_mla`.
  - **Flash (online softmax)** rewrote the compute chunk loop into the canonical running-(max,sum,output) form. Per chunk: QK→mask→`reduce_c<MAX>`(do_eltwise_max=c>0, running max)→`sub_exp` in-place→`reduce_c<SUM>`(chunk sum)→PV; then for c>0 combine via `correction = sub_exp_block(prev_max,cur_max)` (=exp((prev−cur)·scale)), `prev_sum*=corr` (`mul_tiles_bcast_cols_inplace`)+`cur_sum+=prev_sum`, and `cur_out += prev_out*corr` via `mul_block_bcast_cols<...,pack_accumulate=true>` (L1 acc). Ping-pong (prev/cur) CB handles as runtime uint32 + `std::swap`; max needs 2 buffers (reduce_c reads prev_cb≠out_cb), sum/out also ping-pong. SUM done with explicit `reduce_c<SUM>` (not SDPA's sub_exp-partial + `matmul_reduce`+col_identity) → no col_identity CB needed. n_chunks==1 degenerates to plain single-chunk (same end state), so it subsumes the earlier path.
  - **untilize takes a compile-time CB id** but the running out lives in a swapped (runtime) handle → `copy_block(out_prev, CB_OUT_IM)` into a fixed CB, then `untilize<vDHt,CB_OUT_IM,CB_OUT_RM>`.
  - **Per-token CB hygiene** (matters for >1 token/core, i.e. S>num_cores; num_cores=110 here): everything must be empty at token end. `matmul_blocks` only pops in1, so `CB_QK_IM` popped after PV and **`CB_Q_IN` popped after the chunk loop** (Q is reused across chunks). The final running max is never consumed by a combine → explicit `cb_pop_front(max_prev)` at token end. `CB_SCALE` is intentionally persistent (reduce scaler, never popped). Verified by `test_sparse_sdpa_multitoken` (S=256>110 ⇒ 2–3 tok/core) at 1/2/4 chunks.
  - CB set (20): +ping-pong `CB_{MAX,SUM,OUT}_{A,B}`, `CB_CORR`; `CB_OUT_{A,B}` single-buffered (=vDHt tiles) for L1 acc. Writer `CB_OUT_RM` 13→18, reader `CB_IDX` 14→19.
- **Stage 1 acceptance met.** Remaining (future): Stage 2 = H=16 / `{16,32}` tiles (LLK uplift per PLAN); perf path (fuse sub_exp partial-sum, streaming overlap); larger TOPK sweep up to 2048.
