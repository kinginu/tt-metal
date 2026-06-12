// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_unary_broadcast_operands.h"
#include "llk_unpack_unary_operand.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

/**
 *
 * @brief Initialize selected unpacker to unpack a single tile
 *
 * @tparam TRANSPOSE_EN: Enables transpose of a tile, supported for SrcA and SrcB
 * @tparam IS_32b_DEST_EN: Enable using Math destination Register in 32-bit mode
 * @param operand: The input operand circular buffer
 *
 * This function initializes unpacker0 to unpack a single tile
 * from the input circular buffer to srcA/dest register.
 */
template <bool TRANSPOSE_EN, bool IS_32b_DEST_EN>
inline void llk_unpack_A_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, TRANSPOSE_EN, IS_32b_DEST_EN>(operand_id, NUM_TILES, num_faces);
}

/**
 *
 * @brief Initialize unpacker for unary / unary-broadcast / binary-dest-reuse paths.
 *
 * Overload matching Blackhole/Wormhole API signature `(transpose_of_faces, within_face_16x16_transpose, operand)`.
 *
 * When `binary_reuse_dest != NONE`, uses the eltwise-binary dest-reuse init path (UNP_A, default tile/face counts).
 * Otherwise uses the unary / unary-broadcast path. For the non-broadcast path the UNP_DEST routing decision is
 * made inside the primitive, which gates on the operand's BD-table format (operand_id == buf_desc_id) and routes
 * only 32-bit operands to dest — so `unpack_to_dest` can be passed unconditionally.
 *
 * @tparam BType: Broadcast type; BroadcastType::NONE selects the plain unary path
 * @tparam acc_to_dest: Unused on Quasar in dest-reuse path; kept for API parity
 * @tparam binary_reuse_dest: Dest reuse mode; when not NONE, selects the dest-reuse sub-path
 * @tparam unpack_to_dest: When true, the (non-broadcast) primitive routes 32-bit operands through UNP_DEST
 * @param transpose_of_faces: Non-zero enables transpose of 16x16 faces (unary/broadcast NONE path only)
 * @param within_face_16x16_transpose: Unused on Quasar; kept for API parity with Blackhole / other arches
 * @param operand: The input operand logical dataflow buffer / CB id
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    [[maybe_unused]] bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(
    const std::uint32_t transpose_of_faces = 0,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {
        static_assert(unpack_to_dest == false, "unpack_to_dest is not yet supported on Quasar");
        static_assert(acc_to_dest == false, "acc_to_dest is not yet supported on Quasar");
        static_assert(BType == BroadcastType::NONE, "On Quasar, only BroadcastType::NONE is supported for dest reuse");

        // For Quasar, the unp_sel field is ignored if binary_reuse_dest != EltwiseBinaryReuseDestType::NONE
        _llk_unpack_unary_operand_init_<
            p_unpacr::UNP_A,
            false /* TRANSPOSE_EN */,
            false /* IS_32b_DEST_EN */,
            binary_reuse_dest>(operand_id);
    } else {
        if constexpr (BType == BroadcastType::NONE) {
            LLK_ASSERT(
                transpose_of_faces == within_face_16x16_transpose,
                "Quasar unpack unary supports only full transpose (transpose_of_faces and within_face_16x16_transpose "
                "must match)");
            const std::uint32_t num_faces = get_operand_num_faces(operand_id);
            // Pass UNP_A + unpack_to_dest; the primitive redirects 32-bit operands to UNP_DEST internally
            // (format-gated via the BD table), so this is safe even when unpack_to_dest is set for a 16-bit operand.
            if (transpose_of_faces && within_face_16x16_transpose) {
                _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, true, DST_ACCUM_MODE, binary_reuse_dest, unpack_to_dest>(
                    operand_id, 1, num_faces);
            } else {
                _llk_unpack_unary_operand_init_<
                    p_unpacr::UNP_A,
                    false,
                    DST_ACCUM_MODE,
                    binary_reuse_dest,
                    unpack_to_dest>(operand_id, 1, num_faces);
            }
        } else {
            constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
            constexpr bool is_fp32_dest_acc_en = unpack_to_dest ? false : DST_ACCUM_MODE;
            _llk_unpack_unary_broadcast_operands_init_<unp_sel, BType, unpack_to_dest, is_fp32_dest_acc_en>(
                operand_id, 1);
        }
    }
}

/**
 *
 * @brief Unpacks a single operand for unary and unary-broadcast paths.
 *
 * For the non-broadcast path the UNP_DEST routing and the UNPACK_MATH / MATH_PACK semaphore handshake live
 * inside the primitive; this wrapper forwards `unpack_to_dest` and the operand id (used as buf_desc_id for
 * the BD-table format lookup).
 *
 * @tparam BType: Broadcast type; BroadcastType::NONE selects the plain unary path
 * @tparam acc_to_dest: Unused on Quasar; kept for API parity with Blackhole / other arches
 * @tparam binary_reuse_dest: Dest reuse mode (unary path only)
 * @tparam unpack_to_dest: when true, the (non-broadcast) primitive routes 32-bit operands through UNP_DEST
 * @param operand: The logical dataflow buffer id
 * @param tile_index: The index in the input CB to read from
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    [[maybe_unused]] bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    WAYPOINT("UPAW");
    const std::uint32_t operand_id = get_operand_id(operand);
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(operand_id);
    const std::uint32_t l1_tile_idx =
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_entry_idx + tile_index;
    if constexpr (BType == BroadcastType::NONE) {
        _llk_unpack_unary_operand_<p_unpacr::UNP_A, binary_reuse_dest, unpack_to_dest, DST_SYNC_MODE>(
            l1_tile_idx, operand_id /*buf_desc_id*/);
    } else {
        constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
        _llk_unpack_unary_broadcast_operands_<unp_sel, unpack_to_dest>(l1_tile_idx);
    }
    WAYPOINT("UPAD");
}

/**
 * @brief Unpacks a contiguous block of tiles for unary and unary-broadcast paths.
 *
 * @tparam BType: Broadcast type; BroadcastType::NONE selects the plain unary path
 * @tparam acc_to_dest: Unused on Quasar; kept for API parity with Blackhole / other arches
 * @tparam binary_reuse_dest: Dest reuse mode (unary path only)
 * @tparam unpack_to_dest: when true, the (non-broadcast) primitive routes 32-bit operands through UNP_DEST
 * @param operand: The logical dataflow buffer id
 * @param start_tile_index: The starting tile index within the input buffer
 * @param ntiles: The number of consecutive tiles to unpack
 */
// TODO: AM; Optimize block calls by using ntiles per unpack, issue #40798
template <
    BroadcastType BType = BroadcastType::NONE,
    [[maybe_unused]] bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_block(
    const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(operand_id);
    const std::uint32_t rd_entry_idx = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_entry_idx;
    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        WAYPOINT("UPAW");
        if constexpr (BType == BroadcastType::NONE) {
            _llk_unpack_unary_operand_<p_unpacr::UNP_A, binary_reuse_dest, unpack_to_dest, DST_SYNC_MODE>(
                rd_entry_idx + tile_index, operand_id /*buf_desc_id*/);
        } else {
            constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
            _llk_unpack_unary_broadcast_operands_<unp_sel, unpack_to_dest>(rd_entry_idx + tile_index);
        }
        WAYPOINT("UPAD");
    }
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_A_uninit(const std::uint32_t operand) {}
