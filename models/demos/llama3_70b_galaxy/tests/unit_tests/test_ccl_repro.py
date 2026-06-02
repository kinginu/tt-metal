# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger
import ttnn


def _fabric():
    fabric_map = {
        "1d": ttnn.FabricConfig.FABRIC_1D,
        "1d_ring": ttnn.FabricConfig.FABRIC_1D_RING,
        "2d": ttnn.FabricConfig.FABRIC_2D,
        "2d_torus_x": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        "2d_torus_y": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        "2d_torus_xy": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    }
    return fabric_map[os.getenv("QWEN_FABRIC", "2d_torus_xy").strip().lower()]


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = int(os.getenv("QWEN_FABRIC_PAYLOAD", "15232"))
    return cfg


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": _fabric(),
            "fabric_router_config": _router_config(),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_ccl_repro(mesh_device, reset_seeds, ensure_gc):
    topo_name = os.getenv("CCL_TOPO", "Ring")
    topology = getattr(ttnn.Topology, topo_name)

    submesh_env = os.getenv("CCL_SUBMESH", "")  # e.g. "4,2"
    if submesh_env:
        r, c = (int(v) for v in submesh_env.split(","))
        mesh_device = mesh_device.create_submesh(ttnn.MeshShape((r, c)))
        logger.info(f"Using submesh {tuple(mesh_device.shape)}")
    logger.info(f"mesh shape={tuple(mesh_device.shape)} fabric={_fabric()} topology={topology}")

    # replicated [1,1,32,32] tensor across the whole mesh
    full = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)

    use_experimental = os.getenv("CCL_EXPERIMENTAL", "0") == "1"

    # Global semaphores for the experimental async path (mirrors TT_CCL setup).
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    sems = [[ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)] for _ in range(2)]
    barrier = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)]

    for cluster_axis in (0, 1):
        t = ttnn.from_torch(
            full,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        try:
            if use_experimental:
                out = ttnn.experimental.all_gather_async(
                    t,
                    dim=3,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    topology=topology,
                    multi_device_global_semaphore=sems[cluster_axis],
                    barrier_semaphore=barrier[cluster_axis],
                    num_links=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    subdevice_id=ttnn.SubDeviceId(0),
                )
            else:
                out = ttnn.all_gather(
                    t,
                    dim=3,
                    cluster_axis=cluster_axis,
                    topology=topology,
                    num_links=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            ttnn.synchronize_device(mesh_device)
            logger.info(f"[OK] experimental={use_experimental} cluster_axis={cluster_axis} -> out shape {out.shape}")
        except Exception as e:
            logger.error(
                f"[FAIL] experimental={use_experimental} cluster_axis={cluster_axis} -> {type(e).__name__}: {e}"
            )
