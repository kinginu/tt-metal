# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Architecture dispatcher: Wormhole runs the original (main) test code path; Blackhole runs the
# Qwen3 Blackhole-Galaxy bring-up test code path. This keeps the Wormhole behavior identical to
# main while letting Blackhole exercise the no-prefetcher bring-up path.
import os
import importlib.util

import ttnn


def _is_blackhole_galaxy():
    # Optional explicit override (set to "blackhole"/"bh" or "wormhole"/"wh").
    forced = os.environ.get("QWEN_TEST_FORCE_ARCH", "").lower()
    if forced in ("blackhole", "bh"):
        return True
    if forced in ("wormhole", "wormhole_b0", "wh"):
        return False
    try:
        cluster_type = ttnn.cluster.get_cluster_type()
        if cluster_type == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
            return True
        if cluster_type in (ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG):
            return False
    except Exception:
        pass
    arch = os.environ.get("ARCH_NAME", "")
    if not arch:
        try:
            arch = ttnn.get_arch_name()
        except Exception:
            arch = ""
    return "blackhole" in arch.lower()


_impl_file = "_qwen_mlp_bh_impl.py" if _is_blackhole_galaxy() else "_qwen_mlp_wh_impl.py"
_spec = importlib.util.spec_from_file_location(
    "qwen_mlp_active_impl", os.path.join(os.path.dirname(__file__), _impl_file)
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

test_qwen_mlp_inference = _impl.test_qwen_mlp_inference
