import math
import ttnn


def test_gn(device, *, use_weight=True, use_bias=True, use_input_mask=True):
    N, C, H, W = 1, 480, 1, 64
    grid_size = ttnn.CoreGrid(y=1, x=1)
    num_out_blocks = 1
    tile_size = 32
    num_groups = 8

    input_tensor_row_major = ttnn.rand(
        [N, 1, H * W, C], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    input_tensor_tilized = ttnn.tilize_with_zero_padding(input_tensor_row_major, use_multicore=True)

    # input mask
    width_per_group = C // num_groups
    max_tiles_group_can_span = 1 + math.ceil((width_per_group - 1) / tile_size)
    input_mask_tensor = ttnn.zeros(
        [1, num_groups, tile_size, max_tiles_group_can_span * tile_size],
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # gamma/beta
    gamma_beta = ttnn.rand([1, 1, 15, 32], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # groupnorm — pass None for any optional we want to omit
    output_tensor = ttnn.group_norm(
        input_tensor_tilized,
        num_groups=num_groups,
        input_mask=input_mask_tensor if use_input_mask else None,
        weight=gamma_beta if use_weight else None,
        bias=gamma_beta if use_bias else None,
        output_layout=ttnn.TILE_LAYOUT,
        core_grid=grid_size,
        inplace=False,
        num_out_blocks=num_out_blocks,
    )

    ttnn.synchronize_device(device)
    output_tensor = ttnn.from_device(output_tensor)
    return output_tensor


if __name__ == "__main__":
    cases = [
        ("baseline (all provided)", dict(use_weight=True, use_bias=True, use_input_mask=True)),
        ("no weight", dict(use_weight=False, use_bias=True, use_input_mask=True)),
        ("no bias", dict(use_weight=True, use_bias=False, use_input_mask=True)),
        ("no weight + no bias", dict(use_weight=False, use_bias=False, use_input_mask=True)),
        ("no input_mask", dict(use_weight=True, use_bias=True, use_input_mask=False)),
        ("no weight + no bias + no input_mask", dict(use_weight=False, use_bias=False, use_input_mask=False)),
    ]

    device = ttnn.open_device(device_id=0)
    results = []
    try:
        for name, kw in cases:
            try:
                out = test_gn(device, **kw)
                results.append((name, f"OK  shape={out.shape}"))
            except Exception as e:
                results.append((name, f"FAIL {type(e).__name__}: {str(e).splitlines()[0]}"))
    finally:
        ttnn.close_device(device)

    print("\n==================== group_norm optional-args check ====================")
    for name, status in results:
        print(f"  [{status.split()[0]:4}] {name:38} -> {status}")
    print("========================================================================")
