#!/usr/bin/env python3
"""Shared spring and ablation configuration for the K_lat hardware workflow.

Theory-backed parameters from IEEE paper:
"Detecting Force-Induced Instability in Bimanual Manipulation of Deformable Objects:
Contact Rotation and Buckling as Competing Stability Margins"

Key relationships:
  - B_eff = EI / K^2 (K=1/2 for clamped-clamped)
  - P_b* = pi^2 * B_eff / L_0^2 (critical buckling load)
  - K_lat* = P_b*^2 / k_theta (crossover lateral stiffness)
  - m_b = pi^2 * B_eff / L(P)^2 - P (buckling margin)
  - m_c = k_theta^0 / L(P) - P * chi * L(P) - P^2 / K_lat (contact rotation margin)
"""

from __future__ import annotations

import math
from typing import Dict, List


# ── Spring Parameters (from paper Table 1) ────────────────────────────────────

SPRING_SPECIMENS = {
    "s1": {
        "label": "Spring S1 (measured)",
        "L_0": 0.128,              # Free length [m]
        "k_a": 49.4,               # Axial stiffness [N/m]
        "EI": 4.36e-4,             # Bending stiffness [N·m²]
        "B_eff": 1.74e-3,          # Effective bending stiffness [N·m²] (B_eff = 4*EI for clamped-clamped)
        "P_b_star": 1.051,         # Critical buckling load at free length [N]
        "K_lat_star": 25.3,        # Crossover lateral stiffness [N/m]
        "k_theta_0": 0.00558,      # Length-scaled rotational stiffness [N·m²]
        "chi": 0.019,              # Moment-arm ratio (dimensionless)
        "width_m": 0.024,          # Spring width [m] (for reference)
        "thickness_m": 0.0017,     # Spring thickness [m] (for reference)
    },
    "s2": {
        "label": "Spring S2 (measured)",
        "L_0": 0.170,              # Free length [m]
        "k_a": 87.1,               # Axial stiffness [N/m]
        "EI": 1.47e-3,             # Bending stiffness [N·m²]
        "B_eff": 5.86e-3,          # Effective bending stiffness [N·m²]
        "P_b_star": 2.002,         # Critical buckling load at free length [N]
        "K_lat_star": 122.1,       # Crossover lateral stiffness [N/m]
        "k_theta_0": 0.00558,      # Length-scaled rotational stiffness [N·m²]
        "chi": 0.019,              # Moment-arm ratio (dimensionless)
        "width_m": 0.024,          # Spring width [m] (for reference)
        "thickness_m": 0.0017,     # Spring thickness [m] (for reference)
    },
}

# Legacy parameters (kept for backward compatibility)
SPRING_NOMINAL_LENGTH_M = 0.31
SPRING_EI_NM2 = 0.0085
SPRING_END_CONSTRAINT_FACTOR = 0.5
SPRING_EFFECTIVE_BENDING_STIFFNESS_NM2 = SPRING_EI_NM2 / (SPRING_END_CONSTRAINT_FACTOR ** 2)
SPRING_CONTACT_ROTATIONAL_STIFFNESS_NM_PER_RAD = 0.018
SPRING_CENTER_HEIGHT_M = 0.05

SPRING_THEORETICAL_BUCKLING_LOAD_N = (
    (math.pi ** 2) * SPRING_EFFECTIVE_BENDING_STIFFNESS_NM2 / (SPRING_NOMINAL_LENGTH_M ** 2)
)
SPRING_THEORETICAL_CRITICAL_K_LAT_NPM = (
    SPRING_THEORETICAL_BUCKLING_LOAD_N ** 2
) / SPRING_CONTACT_ROTATIONAL_STIFFNESS_NM_PER_RAD

DEFAULT_DEPTH_CAMERA_HEIGHT_M = 0.61
DEFAULT_DEPTH_CAMERA_HORIZONTAL_OFFSET_M = 0.46
DEFAULT_DEPTH_CAMERA_SOURCE = "docstring_default_lookat"

DEFAULT_ARUCO_DICTIONARY_NAME = "DICT_4X4_50"
DEFAULT_ARUCO_MARKER_LENGTH_M = 0.02
DEFAULT_ARUCO_ROBOT1_MARKER_ID = 1
DEFAULT_ARUCO_ROBOT2_MARKER_ID = 2
DEFAULT_ARUCO_METAL_PLATFORM_MARKER_ID = 0
DEFAULT_ARUCO_TARGET_ROW_FRACTION = 0.50

# Spring-cap geometry from ws/src/open_manipulator/open_manipulator_x_description/models/spring_assembly_wall/model.sdf.
# The robot1 cap is centered at x=-0.0865, y=0.0, z=0.08 and measures 0.045 m x 0.060 m x 0.060 m.
SPRING_CAP_ROBOT1_CENTER_X_M = -0.0865
SPRING_CAP_ROBOT1_CENTER_Y_M = 0.0
SPRING_CAP_ROBOT1_CENTER_Z_M = 0.08
SPRING_CAP_ROBOT1_SIZE_X_M = 0.045
SPRING_CAP_ROBOT1_SIZE_Y_M = 0.060
SPRING_CAP_ROBOT1_SIZE_Z_M = 0.060
SPRING_CAP_ROBOT1_HALF_SIZE_Y_M = SPRING_CAP_ROBOT1_SIZE_Y_M / 2.0
SPRING_CAP_ROBOT1_HALF_SIZE_Z_M = SPRING_CAP_ROBOT1_SIZE_Z_M / 2.0

DEFAULT_ALIGNMENT_SOURCE = "auto"
DEFAULT_ALIGNMENT_POLICY = "auto_then_manual"
DEFAULT_ALIGNMENT_MAX_AUTO_Z_TRIM_M = 0.006
DEFAULT_ALIGNMENT_FALLBACK_W_TOL_M = 0.002
DEFAULT_ALIGNMENT_FALLBACK_YAW_TOL_DEG = 2.0
DEFAULT_ALIGNMENT_FALLBACK_PSI_DIFF_TOL_DEG = 2.0

CARTESIAN_STIFFNESS_LIMIT_NPM = 65.0
K_LAT_MID_BAND_START_FRACTION = 0.40

_BASE_K_LAT_GRID_NPM: List[float] = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def build_hw_k_lat_cases(limit_npm: float = CARTESIAN_STIFFNESS_LIMIT_NPM) -> Dict[str, List[float]]:
    limit_npm = float(limit_npm)
    low_band_threshold = limit_npm * K_LAT_MID_BAND_START_FRACTION
    filtered_values = [value for value in _BASE_K_LAT_GRID_NPM if value < limit_npm]
    cases: Dict[str, List[float]] = {}

    low_values = [value for value in filtered_values if value <= low_band_threshold]
    mid_values = [value for value in filtered_values if value > low_band_threshold]

    if low_values:
        cases["K_low"] = low_values
    if mid_values:
        cases["K_mid"] = mid_values
    cases["K_ref"] = [limit_npm]
    return cases


HW_K_LAT_CASES = build_hw_k_lat_cases()