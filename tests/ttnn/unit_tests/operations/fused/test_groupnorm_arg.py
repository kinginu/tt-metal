# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #26594: weight, bias and input_mask are optional in ttnn.group_norm.

On the interleaved (DRAM) path, omitting weight/bias must skip the affine transform (previously
segfaulted) and omitting input_mask must auto-generate the mask (previously hung the device). Every
combination is validated against the torch reference, so a skipped affine step is checked for
numerical correctness, not merely for the absence of a crash.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

DEVICE_PARAMS_L1_SMALL_SIZE = [{"l1_small_size": 0}]

# (use_weight, use_bias, use_input_mask) -> readable test id
OPTIONAL_ARG_CASES = {
    "all_provided": (True, True, True),
    "no_weight": (False, True, True),
    "no_bias": (True, False, True),
    "no_weight_no_bias": (False, False, True),
    "no_input_mask": (True, True, False),
    "none_provided": (False, False, False),
}


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize(
    "use_weight, use_bias, use_input_mask",
    OPTIONAL_ARG_CASES.values(),
    ids=OPTIONAL_ARG_CASES.keys(),
)
def test_group_norm_optional_args(device, use_weight, use_bias, use_input_mask):
    torch.manual_seed(0)

    N, C, H, W = 1, 480, 1, 64
    num_groups = 8
    grid_size = ttnn.CoreGrid(y=1, x=1)

    # Torch reference (gamma/beta dropped from the reference when the op omits them).
    torch_input = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output = torch.nn.functional.group_norm(
        torch_input,
        num_groups,
        weight=torch_weight if use_weight else None,
        bias=torch_bias if use_bias else None,
        eps=1e-12,
    )
    torch_output = torch_output.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Device input: tilized, interleaved DRAM.
    input_tensor = torch_input.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor = ttnn.tilize_with_zero_padding(input_tensor, use_multicore=True)

    # Build correctly-shaped gamma/beta/mask, then pass None below to omit them.
    # (Passing None is equivalent to omitting the argument entirely.)
    [gamma, beta], input_mask = ttnn.dram_group_norm_params_from_torch(
        [torch_weight, torch_bias], C, num_groups, device, core_grid=grid_size, return_mask=True
    )

    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask if use_input_mask else None,
        weight=gamma if use_weight else None,
        bias=beta if use_bias else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
        core_grid=grid_size,
        inplace=False,
        num_out_blocks=1,
    )
    ttnn.synchronize_device(device)

    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor))
    assert_with_pcc(torch_output, output_tensor, 0.999)
