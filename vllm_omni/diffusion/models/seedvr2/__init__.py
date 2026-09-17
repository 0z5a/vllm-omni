# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SeedVR2 (``NaDiT``) video super-resolution support.

The model ports ByteDance's SeedVR2 diffusion transformer together with the
window-aligned sequence-parallel path described in RFC #7723:

* :mod:`~vllm_omni.diffusion.models.seedvr2.window_geometry` -- reference-faithful
  3D window geometry (regular / shifted, 720p-normalised window sizes).
* :mod:`~vllm_omni.diffusion.models.seedvr2.window_sp` -- token-balanced window
  planner and the Plan A variable-split redistribution runtime.
"""

from vllm_omni.diffusion.models.seedvr2.window_geometry import (
    DEFAULT_WINDOW,
    DEFAULT_WINDOW_METHODS,
    GEOMETRY_VERSION,
    layer_window_method,
    make_720p_shifted_windows,
    make_720p_windows,
    window_layout_geometry,
)
from vllm_omni.diffusion.models.seedvr2.window_sp import (
    PLANNER_VERSION,
    RankWindowPlan,
    WindowAssignment,
    WindowLayout,
    WindowLayoutKey,
    WindowLayoutManager,
    WindowPlanCache,
    WindowRedistributionPlan,
    build_rank_window_plan,
    build_redistribution_plans,
    build_window_assignment,
    build_window_layout,
    global_window_mean,
    joint_cu_seqlens,
    materialize_rank_window_plan,
    materialize_redistribution_plan,
    redistribute_window_rows,
)

__all__ = [
    "DEFAULT_WINDOW",
    "DEFAULT_WINDOW_METHODS",
    "GEOMETRY_VERSION",
    "PLANNER_VERSION",
    "RankWindowPlan",
    "WindowAssignment",
    "WindowLayout",
    "WindowLayoutKey",
    "WindowLayoutManager",
    "WindowPlanCache",
    "WindowRedistributionPlan",
    "build_rank_window_plan",
    "build_redistribution_plans",
    "build_window_assignment",
    "build_window_layout",
    "global_window_mean",
    "joint_cu_seqlens",
    "layer_window_method",
    "make_720p_shifted_windows",
    "make_720p_windows",
    "materialize_rank_window_plan",
    "materialize_redistribution_plan",
    "redistribute_window_rows",
    "window_layout_geometry",
]
