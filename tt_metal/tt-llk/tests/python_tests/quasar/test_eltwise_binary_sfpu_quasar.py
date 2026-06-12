# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Consolidated Quasar binary-SFPU test.

Replaces the three standalone Quasar binary-SFPU test pairs (sfpu_binary [int],
sfpu_binary_float [mul/div], sfpu_binary_max_min) with a single python driver +
single cpp source (`sources/quasar/eltwise_binary_sfpu_quasar_test.cpp`) +
dispatcher header (`sfpu_operations_quasar.h`, shared with the unary helpers).
The op is selected at
compile time via the `SFPU_BINARY_OP = ckernel::BinaryOp::<op>` constant (the LLK
BinaryOp enum, extended with the comparison and max/min ops).

The three op families differ structurally on Quasar (operand addressing, dispatch
wrapper, init, dual unpack path, golden, stimuli), so they are kept as three
parametrized test functions here — all targeting the one cpp — rather than a
single homogeneous driver. Each function preserves its standalone file's coverage.
"""

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    BinarySFPUGolden,
    get_golden_generator,
    quantize_mx_stimuli,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    InputOutputFormat,
    generate_sfpu_format_dest_acc_combinations,
    input_output_formats,
    is_invalid_quasar_sfpu_format_combination,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import (
    StimuliSpec,
    apply_log_uniform_magnitudes,
    compute_safe_input_magnitude_range,
    format_elem_max,
    generate_stimuli,
)
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    SFPU_BINARY_OP,
    SFPU_TILE_INDICES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test

_CPP_SOURCE = "sources/quasar/eltwise_binary_sfpu_quasar_test.cpp"
_ELEMENTS_PER_TILE = 1024  # 4 faces * 16 rows * 16 cols


# ===========================================================================
# Family 1 — integer ops (add, mul, gt, lt, le, ge), Int32 only.
# Ported from test_sfpu_binary_quasar.py. Operands at tiles (0, 1) -> tile 0.
# ===========================================================================
_INT_SRC0_IDX, _INT_SRC1_IDX, _INT_DST_IDX = 0, 1, 0


def _run_sfpu_binary_int_quasar(
    data_format,
    dest_acc,
    mathop,
    binary_op,
    clamp_inputs=None,
):
    """Shared driver for the Int32 binary SFPU ops (add/mul/comparisons)."""
    src0_idx, src1_idx, dst_idx = _INT_SRC0_IDX, _INT_SRC1_IDX, _INT_DST_IDX
    num_tiles_needed = max(src0_idx, src1_idx, dst_idx) + 1
    formats = InputOutputFormat(input_format=data_format, output_format=data_format)
    input_dimensions = [num_tiles_needed * 32, 32]

    iinfo = torch.iinfo(format_dict[data_format])
    spec = StimuliSpec.uniform(low=float(iinfo.min), high=float(iinfo.max - 1))
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=data_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=data_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    if clamp_inputs is not None:
        src_A = torch.clamp(src_A, -clamp_inputs, clamp_inputs)
        src_B = torch.clamp(src_B, -clamp_inputs, clamp_inputs)

    num_faces = 4

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_full = generate_golden(
        mathop, src_A, src0_idx, src1_idx, dst_idx, 32, input_dimensions, data_format
    ).flatten()
    dst_start = dst_idx * _ELEMENTS_PER_TILE
    golden_tensor = golden_full[dst_start : dst_start + _ELEMENTS_PER_TILE]

    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            SFPU_BINARY_OP(binary_op),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            SFPU_TILE_INDICES(src0_idx, src1_idx, dst_idx),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            data_format,
            src_B,
            data_format,
            data_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=1,
            num_faces=num_faces,
            twos_complement=data_format.is_integer(),
        ),
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[data_format])
    assert passed_test(golden_tensor, res_tensor, data_format)


@pytest.mark.quasar
@pytest.mark.parametrize(
    "data_format, dest_acc", [(DataFormat.Int32, DestAccumulation.Yes)]
)
def test_eltwise_binary_sfpu_add_int_quasar(data_format, dest_acc):
    """Binary SFPU ADD (Int32)."""
    _run_sfpu_binary_int_quasar(data_format, dest_acc, MathOperation.SfpuElwadd, "ADD")


@pytest.mark.quasar
@pytest.mark.parametrize(
    "data_format, dest_acc", [(DataFormat.Int32, DestAccumulation.Yes)]
)
def test_eltwise_binary_sfpu_mul_int_quasar(data_format, dest_acc):
    """Binary SFPU MUL (Int32)."""
    _run_sfpu_binary_int_quasar(
        data_format, dest_acc, MathOperation.SfpuElwmulInt, "MUL", clamp_inputs=1000
    )


_INT_COMP_OPS = [
    ("GT", MathOperation.SfpuGtInt),
    ("LT", MathOperation.SfpuLtInt),
    ("LE", MathOperation.SfpuLeInt),
    ("GE", MathOperation.SfpuGeInt),
]


@pytest.mark.quasar
@pytest.mark.parametrize(
    "binary_op, mathop", _INT_COMP_OPS, ids=[op for op, _ in _INT_COMP_OPS]
)
@pytest.mark.parametrize(
    "data_format, dest_acc", [(DataFormat.Int32, DestAccumulation.Yes)]
)
def test_eltwise_binary_sfpu_comp_int_quasar(binary_op, mathop, data_format, dest_acc):
    """Binary SFPU integer comparison (GT/LT/LE/GE)."""
    _run_sfpu_binary_int_quasar(data_format, dest_acc, mathop, binary_op)


# ===========================================================================
# Family 2 — float ops (mul, div). Ported from test_sfpu_binary_float_quasar.py.
# Operand/result tile-index variants exercise result-over-operand aliasing.
# ===========================================================================
_FLOAT_TILE_INDEX_VARIANTS = [(0, 1, 0), (0, 1, 1), (2, 3, 0)]

# Crafted lanes in face 0 exercising the div special-case branches.
_DIV_SPECIAL_CASE_LANES = [
    (0, 0.0, 0.0, "nan"),
    (1, 1.5, 0.0, "pos_inf"),
    (2, -1.5, 0.0, "neg_inf"),
    (3, 2.7, 2.7, "one"),
    (4, -3.3, -3.3, "one"),
]


def _get_valid_float_formats_dest_acc():
    """Float16 + DestAccumulation.Yes is not supported."""
    formats = input_output_formats(
        [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
    )
    return [
        (fmt, dest_acc)
        for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(formats)
        if not (
            fmt.input_format == DataFormat.Float16 and dest_acc == DestAccumulation.Yes
        )
    ]


def _get_valid_implied_math_formats(fmt: FormatConfig):
    if fmt.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def _prepare_float_inputs(src_A, data_format, src0_idx, src1_idx, mathop):
    """Map [0,1) uniform stimuli into op-appropriate ranges (div: ±[0.25,4] + special
    lanes; mul: ±250)."""
    torch_format = format_dict[data_format]
    if mathop == MathOperation.SfpuElwdiv:
        scaled = (src_A.to(torch.float32) - 0.5) * 8.0
        sign = torch.where(scaled >= 0, torch.tensor(1.0), torch.tensor(-1.0))
        abs_scaled = torch.maximum(scaled.abs(), torch.tensor(0.25))
        scaled = (sign * abs_scaled).to(torch_format)
        flat = scaled.flatten()
        for lane, dividend, divisor, _ in _DIV_SPECIAL_CASE_LANES:
            flat[src0_idx * _ELEMENTS_PER_TILE + lane] = dividend
            flat[src1_idx * _ELEMENTS_PER_TILE + lane] = divisor
        return flat.reshape(scaled.shape)
    # SfpuElwmul
    scaled = ((src_A.to(torch.float32) - 0.5) * 500.0).to(torch_format)
    return scaled.flatten().reshape(scaled.shape)


def _run_sfpu_binary_float_quasar(
    formats_dest_acc, implied_math_format, tile_indices, mathop, binary_op
):
    formats, dest_acc = formats_dest_acc
    src0_idx, src1_idx, dst_idx = tile_indices
    num_tiles_needed = max(src0_idx, src1_idx, dst_idx) + 1
    input_dimensions = [num_tiles_needed * 32, 32]

    torch.manual_seed(42)
    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    src_A = _prepare_float_inputs(
        src_A, formats.input_format, src0_idx, src1_idx, mathop
    )

    num_faces = 4

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_full = generate_golden(
        mathop,
        src_A,
        src0_idx,
        src1_idx,
        dst_idx,
        32,
        input_dimensions,
        formats.input_format,
    ).flatten()
    dst_start = dst_idx * _ELEMENTS_PER_TILE
    golden_tensor = golden_full[dst_start : dst_start + _ELEMENTS_PER_TILE]
    torch_format_out = format_dict[formats.output_format]
    golden_tensor = golden_tensor.to(torch_format_out)

    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            SFPU_BINARY_OP(binary_op),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            SFPU_TILE_INDICES(src0_idx, src1_idx, dst_idx),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=1,
            num_faces=num_faces,
        ),
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format_out)
    assert passed_test(golden_tensor, res_tensor, formats.output_format)

    # The kernel's x/x branch forces an exact 1.0 regardless of reciprocal rounding.
    if mathop == MathOperation.SfpuElwdiv:
        for lane, _, _, kind in _DIV_SPECIAL_CASE_LANES:
            if kind != "one":
                continue
            actual = res_tensor[lane].item()
            assert (
                actual == 1.0
            ), f"x/x special case at lane {lane}: expected 1.0, got {actual}"


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_valid_float_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_valid_implied_math_formats(
        formats_dest_acc[0]
    ),
    tile_indices=_FLOAT_TILE_INDEX_VARIANTS,
)
def test_eltwise_binary_sfpu_div_quasar(
    formats_dest_acc, implied_math_format, tile_indices
):
    """Binary SFPU DIV (float)."""
    _run_sfpu_binary_float_quasar(
        formats_dest_acc,
        implied_math_format,
        tile_indices,
        MathOperation.SfpuElwdiv,
        "DIV",
    )


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_valid_float_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_valid_implied_math_formats(
        formats_dest_acc[0]
    ),
    tile_indices=_FLOAT_TILE_INDEX_VARIANTS,
)
def test_eltwise_binary_sfpu_mul_quasar(
    formats_dest_acc, implied_math_format, tile_indices
):
    """Binary SFPU MUL (float)."""
    _run_sfpu_binary_float_quasar(
        formats_dest_acc,
        implied_math_format,
        tile_indices,
        MathOperation.SfpuElwmul,
        "MUL",
    )


# ===========================================================================
# Family 3 — max / min (float + Int32). Ported from test_binary_max_min_quasar.py.
# Layout in0=Dest[0], in1=Dest[1], out=Dest[2]; dual unpack path; torch golden.
# ===========================================================================
SFPU_BINARY_MAX_MIN_FLOAT_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
    ],
)
SFPU_BINARY_MAX_MIN_INT32_FORMATS = input_output_formats([DataFormat.Int32], same=True)


def prepare_binary_max_min_inputs(src_A, src_B, input_format, output_format):
    """Two safe-range inputs for max/min (result == one operand verbatim)."""
    torch_fmt = format_dict[input_format]

    if not torch_fmt.is_floating_point:
        iinfo = torch.iinfo(torch_fmt)
        max_val = iinfo.max // 8
        in0 = (
            torch.clamp(src_A * max_val, iinfo.min, iinfo.max)
            .to(torch_fmt)
            .to(torch.float32)
        )
        in1 = (
            torch.clamp(src_B * max_val, iinfo.min, iinfo.max)
            .to(torch_fmt)
            .to(torch.float32)
        )
        return in0, in1

    cap = min(format_elem_max(input_format), format_elem_max(output_format))
    min_mag, max_mag = compute_safe_input_magnitude_range(
        input_format, output_format, input_magnitude_cap=cap, output_magnitude_cap=cap
    )
    in0 = apply_log_uniform_magnitudes(
        src_A,
        min_magnitude=min_mag,
        max_magnitude=max_mag,
        cast_to_format=input_format,
        sign_source=src_A,
    )
    in1 = apply_log_uniform_magnitudes(
        src_B,
        min_magnitude=min_mag,
        max_magnitude=max_mag,
        cast_to_format=input_format,
        sign_source=src_B,
    )
    return in0, in1


def _generate_max_min_float_combinations(formats_list: List[FormatConfig]):
    combinations = []
    for fmt in formats_list:
        in_fmt = fmt.input_format
        dest_acc_modes = (
            (DestAccumulation.Yes,) if in_fmt.is_32_bit() else (DestAccumulation.No,)
        )
        for dest_acc in dest_acc_modes:
            if is_invalid_quasar_sfpu_format_combination(fmt, dest_acc):
                continue
            for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
                if (
                    in_fmt.is_mx_format()
                    and implied_math_format == ImpliedMathFormat.No
                ):
                    continue
                for is_max_op in [True, False]:
                    for input_dimensions in [[32, 32]]:
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                implied_math_format,
                                is_max_op,
                                input_dimensions,
                            )
                        )
    return combinations


def _generate_max_min_int32_combinations(formats_list: List[FormatConfig]):
    combinations = []
    for fmt in formats_list:
        for dest_acc in (DestAccumulation.Yes,):
            if is_invalid_quasar_sfpu_format_combination(fmt, dest_acc):
                continue
            for is_max_op in [True, False]:
                for input_dimensions in [[32, 32]]:
                    combinations.append((fmt, dest_acc, is_max_op, input_dimensions))
    return combinations


def _run_max_min(
    formats, dest_acc, implied_math_format, is_max_op, input_dimensions, spec, is_int
):
    binary_op = "MAX" if is_max_op else "MIN"
    num_faces = 4
    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    in0, in1 = prepare_binary_max_min_inputs(
        src_A, src_B, formats.input_format, formats.output_format
    )
    output_torch_fmt = format_dict[formats.output_format]
    max_tile_elements = 1024

    if is_int:
        in0_int = in0.to(torch.int32)
        in1_int = in1.to(torch.int32)
        golden_int = (
            torch.maximum(in0_int, in1_int)
            if is_max_op
            else torch.minimum(in0_int, in1_int)
        )
        golden_tensor = golden_int.to(torch.float32)
        buffer_A_combined = torch.cat([in0_int.flatten(), in1_int.flatten()])
        if len(in0_int.flatten()) < max_tile_elements:
            pad_len = max_tile_elements - len(in0_int.flatten())
            buffer_A_combined = torch.cat(
                [
                    in0_int.flatten(),
                    torch.zeros(pad_len, dtype=torch.int32),
                    in1_int.flatten(),
                    torch.zeros(pad_len, dtype=torch.int32),
                ]
            )
        buffer_B_dummy = in1_int
        disable_format_inference = False
    else:
        torch_fmt = format_dict[formats.input_format]
        in0 = in0.to(torch_fmt)
        in1 = in1.to(torch_fmt)
        if formats.input_format.is_mx_format():
            in0_g = quantize_mx_stimuli(
                in0.flatten(), formats.input_format, num_faces
            ).reshape(in0.shape)
            in1_g = quantize_mx_stimuli(
                in1.flatten(), formats.input_format, num_faces
            ).reshape(in1.shape)
        else:
            in0_g, in1_g = in0, in1
        in0_f32 = in0_g.to(torch.float32)
        in1_f32 = in1_g.to(torch.float32)
        golden_f32 = (
            torch.maximum(in0_f32, in1_f32)
            if is_max_op
            else torch.minimum(in0_f32, in1_f32)
        )
        golden_tensor = golden_f32.to(output_torch_fmt)
        if formats.output_format.is_mx_format():
            golden_tensor = quantize_mx_stimuli(
                golden_tensor.flatten(), formats.output_format, num_faces
            ).reshape(golden_tensor.shape)
        buffer_A_combined = torch.cat([in0.flatten(), in1.flatten()])
        if len(in0.flatten()) < max_tile_elements:
            pad_len = max_tile_elements - len(in0.flatten())
            buffer_A_combined = torch.cat(
                [
                    in0.flatten(),
                    torch.zeros(pad_len, dtype=in0.dtype),
                    in1.flatten(),
                    torch.zeros(pad_len, dtype=in1.dtype),
                ]
            )
        buffer_B_dummy = in1
        disable_format_inference = formats.input_format.is_mx_format()

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            SFPU_BINARY_OP(binary_op),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(2),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            SFPU_TILE_INDICES(0, 1, 2),  # in0=Dest[0], in1=Dest[1], out=Dest[2]
        ],
        variant_stimuli=StimuliConfig(
            buffer_A_combined,
            formats.input_format,
            buffer_B_dummy,  # dummy buffer_B (unused by kernel)
            formats.input_format,
            formats.output_format,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=num_faces,
            sfpu=True,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        disable_format_inference=disable_format_inference,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=output_torch_fmt)
    assert passed_test(golden_tensor, res_tensor, formats.output_format), (
        f"max/min failed for is_max_op={is_max_op}, "
        f"format={formats.input_format}->{formats.output_format}, dest_acc={dest_acc}"
    )


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_is_max_input_dims=_generate_max_min_float_combinations(
        SFPU_BINARY_MAX_MIN_FLOAT_FORMATS
    ),
)
def test_eltwise_binary_sfpu_max_min_float_quasar(
    formats_dest_acc_implied_math_is_max_input_dims,
):
    """Binary SFPU max/min (float + MX)."""
    (formats, dest_acc, implied_math_format, is_max_op, input_dimensions) = (
        formats_dest_acc_implied_math_is_max_input_dims[0]
    )
    spec = StimuliSpec.uniform(low=-0.9, high=1.1)
    _run_max_min(
        formats,
        dest_acc,
        implied_math_format,
        is_max_op,
        input_dimensions,
        spec,
        is_int=False,
    )


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_is_max_input_dims=_generate_max_min_int32_combinations(
        SFPU_BINARY_MAX_MIN_INT32_FORMATS
    ),
)
def test_eltwise_binary_sfpu_max_min_int32_quasar(formats_dest_acc_is_max_input_dims):
    """Binary SFPU max/min (Int32)."""
    (formats, dest_acc, is_max_op, input_dimensions) = (
        formats_dest_acc_is_max_input_dims[0]
    )
    iinfo = torch.iinfo(format_dict[formats.input_format])
    half_max = iinfo.max // 2
    spec = StimuliSpec.uniform(low=float(-half_max + 1), high=float(half_max - 1))
    _run_max_min(
        formats,
        dest_acc,
        ImpliedMathFormat.No,
        is_max_op,
        input_dimensions,
        spec,
        is_int=True,
    )
