# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# AI-generated

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    is_invalid_quasar_sfpu_format_combination,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test

# The six comparison-to-zero modes under test.
COMP_OPS = [
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
    MathOperation.LessThanZero,
    MathOperation.GreaterThanZero,
    MathOperation.LessThanEqualZero,
    MathOperation.GreaterThanEqualZero,
]


def prepare_comp_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """
    Prepare input tensor for comparison-to-zero operations.

    Mixes positive, negative, exact +0.0/-0.0, and small-magnitude values so the
    sign-vs-magnitude split (ltz/gtz are sign tests; eqz/nez are magnitude tests)
    is exercised. Avoids NaN/subnormal stimuli, which SFPSETCC does not special-case
    and on which Quasar and an IEEE golden could disagree.
    """
    input_torch_format = format_dict[input_format]

    src_A_float = src_A.to(torch.float32)
    src_B_float = src_B.to(torch.float32)

    # Magnitudes in a comfortably-representable range, signed by src_B.
    magnitudes = torch.clamp(torch.abs(src_A_float) * 0.5 + 0.5, 0.1, 100.0)
    signs = torch.where(src_B_float < 0.0, -1.0, 1.0)
    values = signs * magnitudes

    flat = values.flatten()
    # Seed exact zeros of both signs and a few small-magnitude values to pin
    # down the sign-vs-magnitude behaviour at the origin.
    if flat.numel() >= 8:
        flat[0] = 0.0  # +0.0
        flat[1] = -0.0  # -0.0
        flat[2] = 1.0
        flat[3] = -1.0
        flat[4] = 0.5
        flat[5] = -0.5
        flat[6] = 2.0
        flat[7] = -2.0
    values = flat.reshape(values.shape)

    return values.to(input_torch_format)


def generate_sfpu_comp_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate SFPU comp test combinations across the six comparison-to-zero modes.

    Returns: list of (op, format, dest_acc, implied_math_format, input_dimensions) tuples.
    """
    combinations = []

    for op in COMP_OPS:
        for fmt in formats_list:
            in_fmt = fmt.input_format

            # SFPU tests use unpack_to_dest=True: only bit-width-matched Dest modes.
            dest_acc_modes = (
                (DestAccumulation.Yes,)
                if in_fmt.is_32_bit()
                else (DestAccumulation.No,)
            )
            for dest_acc in dest_acc_modes:
                if is_invalid_quasar_sfpu_format_combination(fmt, dest_acc):
                    continue

                for implied_math_format in [ImpliedMathFormat.No]:
                    for input_dimensions in [[32, 32], [64, 64]]:
                        combinations.append(
                            (op, fmt, dest_acc, implied_math_format, input_dimensions)
                        )

    return combinations


SFPU_COMP_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Float16_b,
    ]
)


@pytest.mark.quasar
@parametrize(
    op_formats_dest_acc_implied_math_input_dims=generate_sfpu_comp_combinations(
        SFPU_COMP_FORMATS
    ),
)
def test_sfpu_comp_quasar(op_formats_dest_acc_implied_math_input_dims):
    """
    Test the six comparison-to-zero SFPU ops on Quasar.

    Output is a boolean 1.0/0.0 float per element. The golden matches Quasar's
    SFPSETCC sign-vs-magnitude semantics (see UnarySFPUGolden comparison methods).
    """
    (op, formats, dest_acc, implied_math_format, input_dimensions) = (
        op_formats_dest_acc_implied_math_input_dims[0]
    )

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    src_A = prepare_comp_inputs(
        src_A, src_B, formats.input_format, formats.output_format
    )

    num_faces = 4

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        op,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    # SFPU comp always unpacks directly to Dest; matrix pre-filtered to matched bit-widths.
    configuration = TestConfig(
        "sources/quasar/sfpu_comp_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=op),
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
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
