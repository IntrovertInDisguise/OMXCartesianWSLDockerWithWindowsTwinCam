#!/usr/bin/env python3
"""
Unit tests for tools/hardware_harness_single_arm_ablation.py

These tests verify the structure, argument parsing, and configuration logic
of the single-arm ablation harness WITHOUT requiring ROS2 or hardware.
Tests run with plain pytest in < 1s.

Run:
  python3 -m pytest tools/test_single_arm_ablation_unit.py -v
  bash scripts/test_single_arm_ablation.sh
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import types
import unittest

# Ensure the workspace root is on the path so tools.* imports work
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


def _ensure_ros_mocks():
    """Install mock modules for ROS2 dependencies if not already present.

    Keeps mocks installed for the entire test session so that the
    launch_testing pytest plugin does not encounter KeyErrors when
    introspecting the module.
    """
    mock_modules = [
        "rclpy", "rclpy.node", "rclpy.parameter", "rclpy.qos",
        "geometry_msgs", "geometry_msgs.msg",
        "sensor_msgs", "sensor_msgs.msg",
        "std_msgs", "std_msgs.msg",
    ]
    for mod_name in mock_modules:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    rclpy = sys.modules["rclpy"]
    if not hasattr(rclpy, "init"):
        rclpy.init = lambda *a, **kw: None
        rclpy.shutdown = lambda *a, **kw: None
        rclpy.spin_once = lambda *a, **kw: None

    node_mod = sys.modules["rclpy.node"]
    if not hasattr(node_mod, "Node"):
        class MockNode:
            def __init__(self, *a, **kw): pass
            def create_publisher(self, *a, **kw): return None
            def create_subscription(self, *a, **kw): return None
            def create_timer(self, *a, **kw): return None
            def get_clock(self):
                class C:
                    def now(self_):
                        class T:
                            def to_msg(self__): return None
                        return T()
                return C()
            def get_logger(self):
                class L:
                    def info(self_, *a, **kw): pass
                    def warn(self_, *a, **kw): pass
                return L()
            def destroy_node(self): pass
        node_mod.Node = MockNode

    param_mod = sys.modules["rclpy.parameter"]
    if not hasattr(param_mod, "Parameter"):
        class MockParameter:
            class Type:
                BOOL = 0
            def __init__(self, *a, **kw): pass
        param_mod.Parameter = MockParameter

    qos_mod = sys.modules["rclpy.qos"]
    if not hasattr(qos_mod, "QoSProfile"):
        class MockQoSProfile:
            def __init__(self, *a, **kw): pass
        qos_mod.QoSProfile = MockQoSProfile
    if not hasattr(qos_mod, "ReliabilityPolicy"):
        class MockReliabilityPolicy:
            BEST_EFFORT = 0
        qos_mod.ReliabilityPolicy = MockReliabilityPolicy

    class MockMsg:
        def __init__(self, *a, **kw): self.data = 0.0

    for msg_mod_name in ["geometry_msgs.msg", "sensor_msgs.msg", "std_msgs.msg"]:
        mod = sys.modules[msg_mod_name]
        for attr in ["Point", "Pose", "PoseStamped", "WrenchStamped",
                      "JointState", "Bool", "Float64", "Float64MultiArray"]:
            if not hasattr(mod, attr):
                setattr(mod, attr, MockMsg)

    # Ensure WrenchStamped has nested force/torque
    ws = sys.modules["geometry_msgs.msg"].WrenchStamped
    if not hasattr(ws, "_has_nested"):
        class _Sub:
            def __init__(self):
                self.x = 0.0
                self.y = 0.0
                self.z = 0.0
        _orig_init = ws.__init__ if hasattr(ws, "__init__") else None
        def _new_init(self, *a, **kw):
            self.force = _Sub()
            self.torque = _Sub()
        ws.__init__ = _new_init
        ws._has_nested = True


# Install mocks before any import of the harness module
_ensure_ros_mocks()

# Now import the ablation module
try:
    import tools.hardware_harness_single_arm_ablation as ablation_mod
except ImportError:
    sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "tools"))
    import hardware_harness_single_arm_ablation as ablation_mod


class TestSingleArmAblationImport(unittest.TestCase):
    """Test that the module imports and has required components."""

    def test_module_imports(self):
        """Module can be imported without ROS2 runtime."""
        self.assertIsNotNone(ablation_mod)

    def test_has_main_function(self):
        """Module exposes a main() function."""
        self.assertTrue(hasattr(ablation_mod, "main"))
        self.assertTrue(callable(ablation_mod.main))

    def test_has_parse_args(self):
        """Module exposes parse_args()."""
        self.assertTrue(hasattr(ablation_mod, "parse_args"))
        self.assertTrue(callable(ablation_mod.parse_args))

    def test_has_build_springs_klat(self):
        """Module exposes build_springs_klat()."""
        self.assertTrue(hasattr(ablation_mod, "build_springs_klat"))
        self.assertTrue(callable(ablation_mod.build_springs_klat))

    def test_has_harness_class(self):
        """Module exposes SingleArmAblationHarness class."""
        self.assertTrue(hasattr(ablation_mod, "SingleArmAblationHarness"))

    def test_has_run_result_class(self):
        """Module exposes RunResult dataclass."""
        self.assertTrue(hasattr(ablation_mod, "RunResult"))


class TestRunResult(unittest.TestCase):
    """Test the RunResult dataclass."""

    def test_default_values(self):
        """RunResult has sensible defaults."""
        r = ablation_mod.RunResult()
        self.assertEqual(r.spring, "")
        self.assertEqual(r.case, "")
        self.assertEqual(r.K_lat_Npm, 0.0)
        self.assertEqual(r.rep, 0)
        self.assertFalse(r.passed)
        self.assertEqual(r.ablation_index, 0)
        self.assertEqual(r.abort_reason, "")

    def test_has_required_fields(self):
        """RunResult has all required fields for per-spring logging."""
        r = ablation_mod.RunResult()
        required_fields = [
            "spring", "case", "K_lat_Npm", "rep", "ts", "passed",
            "ablation_index", "abort_reason",
            "t_contact_s", "P_s_max_N", "P_s_at_contact_N",
            "ee_x_at_contact", "ee_x_at_max", "max_press_depth",
            "P_c_th_N", "K_lat_meas_Npm",
            "csv_path", "json_path",
        ]
        for field_name in required_fields:
            self.assertTrue(
                hasattr(r, field_name),
                f"RunResult missing field: {field_name}"
            )

    def test_nan_defaults_for_floats(self):
        """Float fields default to NaN."""
        r = ablation_mod.RunResult()
        nan_fields = [
            "t_contact_s", "P_s_max_N", "P_s_at_contact_N",
            "ee_x_at_contact", "ee_x_at_max", "max_press_depth",
            "P_c_th_N", "K_lat_meas_Npm",
        ]
        for field_name in nan_fields:
            val = getattr(r, field_name)
            self.assertTrue(
                math.isnan(val),
                f"RunResult.{field_name} should be NaN by default, got {val}"
            )


class TestBuildSpringsKlat(unittest.TestCase):
    """Test the build_springs_klat() function with various arg combinations."""

    def _make_args(self, **kwargs):
        """Create a mock args namespace."""
        import argparse
        defaults = {
            "reps": 20,
            "spring": None,
            "k_lat": None,
            "cases": None,
            "dry_run": False,
            "gazebo_smoke": False,
            "robot_namespace": "robot1",
            "abort_disable_torque": False,
            "robot1_port": None,
            "dxl_baud": 1000000,
            "use_sim_time": False,
            "log_dir": "/tmp/test_logs",
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_gazebo_smoke_default(self):
        """Gazebo smoke mode produces a single run at the limit."""
        args = self._make_args(gazebo_smoke=True)
        result = ablation_mod.build_springs_klat(args)
        self.assertIn("gazebo_smoke", result)
        self.assertEqual(len(result["gazebo_smoke"]), 1)
        case_label, k_val = result["gazebo_smoke"][0]
        self.assertEqual(case_label, "gazebo_smoke")
        self.assertEqual(k_val, ablation_mod.CARTESIAN_STIFFNESS_LIMIT_NPM)

    def test_gazebo_smoke_with_k_lat(self):
        """Gazebo smoke mode with explicit K_lat value."""
        args = self._make_args(gazebo_smoke=True, k_lat=[30.0])
        result = ablation_mod.build_springs_klat(args)
        self.assertIn("gazebo_smoke", result)
        case_label, k_val = result["gazebo_smoke"][0]
        self.assertEqual(k_val, 30.0)

    def test_gazebo_smoke_multiple_klat_rejected(self):
        """Gazebo smoke mode rejects multiple K_lat values."""
        args = self._make_args(gazebo_smoke=True, k_lat=[10.0, 20.0])
        with self.assertRaises(SystemExit):
            ablation_mod.build_springs_klat(args)

    def test_explicit_k_lat_values(self):
        """Explicit --k-lat values are passed through."""
        args = self._make_args(k_lat=[10.0, 25.0, 50.0])
        result = ablation_mod.build_springs_klat(args)
        self.assertIn("single_arm", result)
        k_values = [k for _, k in result["single_arm"]]
        self.assertEqual(k_values, [10.0, 25.0, 50.0])

    def test_explicit_k_lat_over_limit_rejected(self):
        """K_lat values above the limit are rejected."""
        args = self._make_args(k_lat=[10.0, 999.0])
        with self.assertRaises(SystemExit):
            ablation_mod.build_springs_klat(args)

    def test_default_hw_cases(self):
        """Default (no args) uses HW_K_LAT_CASES."""
        args = self._make_args()
        result = ablation_mod.build_springs_klat(args)
        self.assertIn("single_arm", result)
        # Should have multiple K_lat values from HW_K_LAT_CASES
        self.assertGreater(len(result["single_arm"]), 1)

    def test_case_filter(self):
        """--cases filters which case names are included."""
        args = self._make_args(cases=["K_ref"])
        result = ablation_mod.build_springs_klat(args)
        self.assertIn("single_arm", result)
        # All entries should have case label "K_ref"
        for case_label, _ in result["single_arm"]:
            self.assertEqual(case_label, "K_ref")

    def test_unknown_cases_rejected(self):
        """Unknown case names are rejected."""
        args = self._make_args(cases=["nonexistent_case"])
        with self.assertRaises(SystemExit):
            ablation_mod.build_springs_klat(args)

    def test_per_spring_with_specimens(self):
        """--spring with known specimen names creates per-spring entries."""
        args = self._make_args(spring=["s1", "s2"])
        result = ablation_mod.build_springs_klat(args)
        self.assertIn("s1", result)
        self.assertIn("s2", result)
        for sp_name in ["s1", "s2"]:
            self.assertGreater(len(result[sp_name]), 0)

    def test_unknown_spring_rejected(self):
        """Unknown spring names are rejected."""
        args = self._make_args(spring=["nonexistent_spring"])
        with self.assertRaises(SystemExit):
            ablation_mod.build_springs_klat(args)


class TestConstants(unittest.TestCase):
    """Test that key constants are defined and reasonable."""

    def test_k_lat_limit_positive(self):
        """CARTESIAN_STIFFNESS_LIMIT_NPM is positive."""
        self.assertGreater(ablation_mod.CARTESIAN_STIFFNESS_LIMIT_NPM, 0)

    def test_default_reps_positive(self):
        """DEFAULT_REPS is a positive integer."""
        self.assertGreater(ablation_mod.DEFAULT_REPS, 0)

    def test_rest_times_nonnegative(self):
        """Rest times between reps and cases are non-negative."""
        self.assertGreaterEqual(ablation_mod.REST_BETWEEN_REPS_S, 0)
        self.assertGreaterEqual(ablation_mod.REST_BETWEEN_CASES_S, 0)

    def test_contact_force_thresholds_positive(self):
        """Contact force thresholds are positive."""
        self.assertGreater(ablation_mod.CONTACT_FORCE_ENTER, 0)
        self.assertGreater(ablation_mod.CONTACT_FORCE_DELTA, 0)

    def test_press_step_positive(self):
        """Press step is positive."""
        self.assertGreater(ablation_mod.PRESS_STEP, 0)

    def test_klat_topic_defined(self):
        """K_lat topic for single arm is defined."""
        self.assertIn("robot1", ablation_mod.KLAT_TOPIC_R1)
        self.assertIn("set_k_lateral", ablation_mod.KLAT_TOPIC_R1)


class TestHarnessClassStructure(unittest.TestCase):
    """Test the SingleArmAblationHarness class structure (without instantiation)."""

    def test_has_run_ablation_method(self):
        """Harness class has run_ablation method."""
        self.assertTrue(hasattr(ablation_mod.SingleArmAblationHarness, "run_ablation"))

    def test_has_set_k_lateral_method(self):
        """Harness class has set_k_lateral method."""
        self.assertTrue(hasattr(ablation_mod.SingleArmAblationHarness, "set_k_lateral"))

    def test_has_stage_methods(self):
        """Harness class has all required stage methods."""
        for stage_name in ["_stage_liveness", "_stage_idle", "_stage_precontact", "_stage_hold", "_stage_ramp"]:
            self.assertTrue(
                hasattr(ablation_mod.SingleArmAblationHarness, stage_name),
                f"Missing stage method: {stage_name}"
            )

    def test_has_logging_methods(self):
        """Harness class has per-run logging methods."""
        for method_name in ["_start_run_log", "_log_tick", "_stop_run_log"]:
            self.assertTrue(
                hasattr(ablation_mod.SingleArmAblationHarness, method_name),
                f"Missing logging method: {method_name}"
            )

    def test_has_safe_abort(self):
        """Harness class has safe_abort method."""
        self.assertTrue(hasattr(ablation_mod.SingleArmAblationHarness, "safe_abort"))


if __name__ == "__main__":
    unittest.main()
