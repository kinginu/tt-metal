// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_tilize.h"
#include "llk_unpack_tilize_operands.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK TILIZE
 *************************************************************************/

/**
 * @brief Initializes the unpacker for tilize operations on Quasar.
 *
 * Configures UNP_A stride registers and programs the MOP for tilizing
 * block_ct_dim tiles from row-major L1 data into face format in SrcA.
 *
 * @param operand       The input dataflow buffer identifier.
 * @param full_ct_dim   Number of tiles in a full row of the input tensor.
 * @param block_ct_dim  Number of tiles per MOP invocation (defaults to 1).
 */
inline void llk_unpack_tilize_init(
    const std::uint32_t operand, const std::uint32_t full_ct_dim, const std::uint32_t block_ct_dim = 1) {
    const std::uint32_t operand_id = get_operand_id(operand);

    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    _llk_unpack_tilize_init_<p_unpacr::UNP_A, DST_ACCUM_MODE>(operand_id, full_ct_dim, block_ct_dim, tensor_shape);
}

/**
 * @brief Tilizes a block of tiles from L1 row-major layout into SrcA.
 *
 * Computes the L1 face index from the DFB read position and the input
 * tile index, then runs the MOP configured by llk_unpack_tilize_init.
 *
 * @param operand          The input dataflow buffer identifier.
 * @param block_c_tiles    Number of tiles in one block row (must match BLOCK_CT_DIM from init).
 * @param input_tile_index Starting tile index (encodes row offset via block_c_tiles stride).
 */
inline void llk_unpack_tilize_block(
    const std::uint32_t operand, const std::uint32_t block_c_tiles, const std::uint32_t input_tile_index = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);

    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    const std::uint32_t faces_per_entry = tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;

    const LocalDFBInterface& local_dfb = g_dfb_interface[operand_id];
    const std::uint32_t rd_entry_idx = local_dfb.tc_slots[local_dfb.tc_idx].rd_entry_idx;

    // TODO (SK) #42757: Remove ct_dim loop when block_ct_dim unpacking optimization implemented.
    // BLOCK_CT_DIM is currently hardcoded to 1 in tilize_init (see compute/tilize.h), so the MOP
    // emits one SrcA dvalid per invocation. Loop to match the per-tile math consumption same
    // structural pattern as BH/WH llk_unpack_tilize_block
    const std::uint32_t l1_base_idx = (rd_entry_idx + input_tile_index) * faces_per_entry;
    for (std::uint32_t t = 0; t < block_c_tiles; t++) {
        _llk_unpack_tilize_<p_unpacr::UNP_A>(l1_base_idx + t);
    }
}

/*************************************************************************
 * LLK UNPACK TILIZE SRC A, UNPACK SRC B
 *************************************************************************/

/**
 * Initialize the unpacker for the combined tilize-A / unpack-B operation.
 *
 * Operand A and B face geometry (face_r_dim, num_faces) is derived from circular-buffer unpack
 * metadata (see set_unpack_face_geometry). In debug builds, validates that both unpackers are
 * configured consistently before programming the init sequence.
 *
 * @tparam neginf_srcA      Initialize srcA padding with negative infinity (for reduce-max).
 * @tparam reload_srcB      Whether srcB is reloaded each iteration.
 * @tparam zero_srcA        Zero out srcA.
 * @tparam zero_srcA_reduce Zero out srcA for the reduce path.
 * @param  operandA         Input operand index for tilize source A.
 * @param  operandB         Input operand index for unpack source B.
 * @param  ct_dim           Number of tiles along the column (tilize block width).
 */
template <
    [[maybe_unused]] bool neginf_srcA = false,
    [[maybe_unused]] std::uint32_t reload_srcB = false,
    [[maybe_unused]] bool zero_srcA = false,
    [[maybe_unused]] bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t ct_dim) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    const ckernel::TensorShape tensor_shape_A = get_operand_tensor_shape(operandA_id);

    // LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly<UnpackerProgramType::ProgramByFace>(
    //     unpack_src_format[operandA_id],
    //     unpack_dst_format[operandA_id],
    //     unpack_src_format[operandB_id],
    //     unpack_dst_format[operandB_id],
    //     unpA_face_r_dim,
    //     unpB_face_r_dim,
    //     num_faces,
    //     get_operand_num_faces(operandB_id)));

    _llk_unpack_tilize_operands_init_<TilizeUnpackerSel::UnpA>(operandA_id, operandB_id, ct_dim, tensor_shape_A);
}

/**
 * Unpack and tilize one srcA tile while unpacking the corresponding srcB tile.
 *
 * Operand A face geometry and narrow-tile flag are derived from CB unpack metadata; source base
 * addresses are read from the CB fifo state.
 *
 * @tparam neginf_srcA      Initialize srcA padding with negative infinity (for reduce-max).
 * @tparam reload_srcB      Whether srcB is reloaded each iteration.
 * @tparam zero_srcA        Zero out srcA.
 * @tparam zero_srcA_reduce Zero out srcA for the reduce path.
 * @param  operandA     Input operand index for tilize source A.
 * @param  operandB     Input operand index for unpack source B.
 * @param  tile_index_a Tile index within operand A.
 * @param  tile_index_b Tile index within operand B.
 * @param  block_ct_dim Number of column tiles in the block.
 */
template <
    [[maybe_unused]] bool neginf_srcA = false,
    [[maybe_unused]] std::uint32_t reload_srcB = false,
    [[maybe_unused]] bool zero_srcA = false,
    [[maybe_unused]] bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    [[maybe_unused]] const std::uint32_t block_ct_dim) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    const LocalDFBInterface& local_dfb_interface_a = get_local_dfb_interface(operandA_id);
    const LocalDFBInterface& local_dfb_interface_b = get_local_dfb_interface(operandB_id);

    const std::uint32_t l1_index_a =
        local_dfb_interface_a.tc_slots[local_dfb_interface_a.tc_idx].rd_entry_idx + tile_index_a;  // revisit
    const std::uint32_t l1_index_b =
        local_dfb_interface_b.tc_slots[local_dfb_interface_b.tc_idx].rd_entry_idx + tile_index_b;

    // LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly<UnpackerProgramType::ProgramByFace>(
    //     unpack_src_format[operandA_id],
    //     unpack_dst_format[operandA_id],
    //     unpack_src_format[operandB_id],
    //     unpack_dst_format[operandB_id],
    //     face_r_dim,
    //     get_operand_face_r_dim(operandB_id),
    //     num_faces,
    //     get_operand_num_faces(operandB_id)));

    WAYPOINT("UPTW");

    _llk_unpack_tilize_operands_<TilizeUnpackerSel::UnpA>(l1_index_a, l1_index_b);

    WAYPOINT("UPTD");
}

/**
 * Unpack and tilize a block of srcA column tiles against srcB by repeatedly calling
 * llk_unpack_tilizeA_B.
 *
 * @tparam neginf_srcA      Initialize srcA padding with negative infinity (for reduce-max).
 * @tparam reload_srcB      Whether srcB is reloaded each iteration.
 * @tparam zero_srcA        Zero out srcA.
 * @tparam zero_srcA_reduce Zero out srcA for the reduce path.
 * @param  operandA        Input operand index for tilize source A.
 * @param  operandB        Input operand index for unpack source B.
 * @param  block_c_tiles_a Number of column tiles in operand A's block.
 * @param  tile_idx_b      Tile index within operand B.
 */
template <
    [[maybe_unused]] bool neginf_srcA = false,
    [[maybe_unused]] std::uint32_t reload_srcB = false,
    [[maybe_unused]] bool zero_srcA = false,
    [[maybe_unused]] bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_block(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t block_c_tiles_a,
    const std::uint32_t tile_idx_b) {
    for (std::uint32_t tile_idx_a = 0; tile_idx_a < block_c_tiles_a; tile_idx_a++) {
        llk_unpack_tilizeA_B<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
            operandA, operandB, tile_idx_a, tile_idx_b, block_c_tiles_a);
    }
}

/**
 * Tear down the combined tilize-A / unpack-B configuration so a subsequent operation can reprogram
 * the unpacker. -> No-op for Quasar.
 *
 * @param operand Input circular buffer / operand index.
 */
inline void llk_unpack_tilizeA_B_uninit([[maybe_unused]] const std::uint32_t operand) {}
