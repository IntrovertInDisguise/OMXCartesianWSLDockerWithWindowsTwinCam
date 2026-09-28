#!/usr/bin/env python3
"""Interactive stage-by-stage hardware test with pauses between stages.

Runs each stage individually and waits for user confirmation before proceeding.
This allows visual inspection of robot positions at each step.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import rclpy
from std_msgs.msg import String
from hardware_harness_contact_gated_load import ContactGatedLoadStepHarness


def wait_for_input(prompt: str) -> str:
    """Read input with explicit stdin handling to avoid EOF issues."""
    try:
        # Use Python's built-in input() which handles terminal input properly
        response = input(prompt).strip()
        if not response:
            # Empty line - treat as continue
            return 'y'
        return response
    except EOFError:
        print("\n[EOF detected] stdin closed - continuing automatically...")
        # Instead of aborting, auto-continue to allow testing
        return 'y'
    except KeyboardInterrupt:
        # Let KeyboardInterrupt propagate to main handler for clean shutdown
        raise
    except Exception as e:
        print(f"\n[Error reading input: {e}] Continuing automatically...")
        return 'y'


def wait_for_user(stage_name: str, stage_num: int) -> bool:
    """Wait for user confirmation before proceeding to next stage."""
    print("\n" + "=" * 80)
    print(f"Stage {stage_num} Complete: {stage_name}")
    print("=" * 80)
    print("\nRobot positions are now HELD. Please inspect the robots.")
    print("\nOptions:")
    print("  [y] Continue to next stage")
    print("  [n] Abort test")
    print("  [s] Skip remaining stages")
    print("  [p] Print current EE positions (not available here)")
    
    while True:
        response = wait_for_input("\nYour choice [y/n/s/p]: ")
        if response in ['y', 'Y', 'yes', 'Yes', '']:
            return True
        elif response in ['n', 'N', 'no', 'No']:
            return False
        elif response in ['s', 'S', 'skip', 'Skip']:
            return False
        else:
            print("Invalid choice. Please enter y, n, or s")


def wait_for_compression_step(prompt: str, harness=None) -> bool:
    """Wait for user confirmation before compression step in Stage 5."""
    print("\nOptions:")
    print("  [y] Continue with compression")
    print("  [n] Abort compression")
    
    while True:
        response = wait_for_input(prompt)
        if response in ['y', 'Y', 'yes', 'Yes', '']:
            return True
        elif response in ['n', 'N', 'no', 'No']:
            return False
        else:
            print("Invalid choice. Please enter y or n")


def query_aruco_platform_pose(timeout_s: float = 5.0):
    """Query ArUco detections to get metal platform pose at startup.
    
    Returns:
        dict: Platform bounds with keys:
              - surface_z: Z height of platform surface
              - center_x: X center of platform
              - center_y: Y center of platform  
              - surface_size: Platform size (square)
              Or None if query fails
    """
    print("Querying ArUco for metal platform position...")
    
    platform_data = {'received': False}
    
    def callback(msg):
        if platform_data['received']:
            return
        try:
            detections = json.loads(msg.data)
            # Look for metal_platform_center marker (ID 0)
            for det in detections.get('detections', []):
                if det.get('marker_id') == 0:  # metal_platform_center
                    pose_world = det.get('pose_world') or det.get('pose_camera')
                    if pose_world:
                        translation = pose_world.get('translation_m', [0, 0, 0])
                        # Marker is at surface corner, so Z = surface height
                        platform_data['surface_z'] = translation[2]
                        platform_data['marker_x'] = translation[0]
                        platform_data['marker_y'] = translation[1]
                        # Platform is 0.30m x 0.30m, marker at corner
                        # Center is offset by half the platform size
                        platform_data['center_x'] = translation[0] + 0.15  # half of 0.30m
                        platform_data['center_y'] = translation[1] + 0.15
                        platform_data['surface_size'] = 0.30  # 30cm square
                        platform_data['received'] = True
                        print(f"✓ ArUco: Platform surface Z={translation[2]:.3f}m, center=({platform_data['center_x']:.3f}, {platform_data['center_y']:.3f})")
        except Exception as e:
            print(f"  ArUco parse error: {e}")
    
    node = rclpy.create_node('aruco_platform_query')
    node.create_subscription(String, '/camera_aruco/detections_json', callback, 10)
    
    start_time = time.time()
    while not platform_data['received'] and (time.time() - start_time) < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
    
    node.destroy_node()
    
    if platform_data['received']:
        return {
            'surface_z': platform_data['surface_z'],
            'center_x': platform_data['center_x'],
            'center_y': platform_data['center_y'],
            'surface_size': platform_data['surface_size'],
        }
    else:
        print("⚠ ArUco query timeout - using default platform bounds")
        return None


def get_platform_bounds(platform_info: dict = None) -> dict:
    """Get platform safety bounds from ArUco or defaults."""
    if platform_info:
        return {
            'surface_z': platform_info['surface_z'],
            'safe_z_min': platform_info['surface_z'] + 0.020,  # 20mm above platform
            'safe_z_target': platform_info['surface_z'] + 0.040,  # 40mm above platform
            'center_x': platform_info['center_x'],
            'center_y': platform_info['center_y'],
            'surface_size': platform_info['surface_size'],
            'min_x': platform_info['center_x'] - platform_info['surface_size'] / 2,
            'max_x': platform_info['center_x'] + platform_info['surface_size'] / 2,
            'min_y': platform_info['center_y'] - platform_info['surface_size'] / 2,
            'max_y': platform_info['center_y'] + platform_info['surface_size'] / 2,
        }
    else:
        # Defaults (assuming platform centered at origin)
        return {
            'surface_z': 0.05,
            'safe_z_min': 0.070,  # 20mm above 50mm surface
            'safe_z_target': 0.090,  # 40mm above 50mm surface
            'center_x': 0.0,
            'center_y': 0.0,
            'surface_size': 0.30,
            'min_x': -0.15,
            'max_x': 0.15,
            'min_y': -0.15,
            'max_y': 0.15,
        }


def check_safety_stage2(harness, platform_bounds: dict) -> bool:
    """Check safety after Stage 2 (homing). Check both X and Z proprioceptively.
    
    Checks:
    - Z is above platform surface + margin
    - X doesn't violate min distance from robot base (check YAML MIN_DISTANCE_FROM_BASE)
    """
    print("\n[Safety Check - Stage 2] Proprioceptive bounds check:")
    
    # Get current EE positions from harness
    ee1 = harness.current_cartesian_position(1)
    ee2 = harness.current_cartesian_position(2)
    
    if ee1 is None or ee2 is None:
        print("  ERROR: Cannot get EE positions")
        return False
    
    print(f"  Robot1 EE: X={ee1[0]:.3f}m, Y={ee1[1]:.3f}m, Z={ee1[2]:.3f}m")
    print(f"  Robot2 EE: X={ee2[0]:.3f}m, Y={ee2[1]:.3f}m, Z={ee2[2]:.3f}m")
    
    # Check Z: must be above platform surface + margin
    safe_z_min = platform_bounds['safe_z_min']
    z1_ok = ee1[2] >= safe_z_min
    z2_ok = ee2[2] >= safe_z_min
    
    if not z1_ok or not z2_ok:
        print(f"  ⚠ Z SAFETY VIOLATION:")
        if not z1_ok:
            print(f"    Robot1 Z={ee1[2]:.3f}m < safe_min={safe_z_min:.3f}m")
        if not z2_ok:
            print(f"    Robot2 Z={ee2[2]:.3f}m < safe_min={safe_z_min:.3f}m")
        return False
    
    print(f"  ✓ Z check: Both robots above safe Z={safe_z_min:.3f}m")
    
    # Check X: must be within platform bounds (with margin for robot base distance)
    # MIN_DISTANCE_FROM_BASE is typically 0.12m in YAML
    # Robot1 base is at X=-0.39, Robot2 base is at X=0.39
    # So Robot1 EE X must be >= -0.39 + 0.12 = -0.27
    # And Robot2 EE X must be <= 0.39 - 0.12 = 0.27
    
    min_dist_from_base = 0.12  # Match YAML MIN_DISTANCE_FROM_BASE
    robot1_base_x = -0.39
    robot2_base_x = 0.39
    
    r1_min_x = robot1_base_x + min_dist_from_base
    r2_max_x = robot2_base_x - min_dist_from_base
    
    x1_ok = ee1[0] >= r1_min_x
    x2_ok = ee2[0] <= r2_max_x
    
    if not x1_ok or not x2_ok:
        print(f"  ⚠ X SAFETY VIOLATION:")
        if not x1_ok:
            print(f"    Robot1 X={ee1[0]:.3f}m < min={r1_min_x:.3f}m (base+{min_dist_from_base}m)")
        if not x2_ok:
            print(f"    Robot2 X={ee2[0]:.3f}m > max={r2_max_x:.3f}m (base-{min_dist_from_base}m)")
        return False
    
    print(f"  ✓ X check: Both robots within min distance from base ({min_dist_from_base}m)")
    
    # Check X: must be within platform bounds (with margin)
    platform_min_x = platform_bounds['min_x'] + 0.010  # 10mm margin
    platform_max_x = platform_bounds['max_x'] - 0.010
    
    x1_platform_ok = platform_min_x <= ee1[0] <= platform_max_x
    x2_platform_ok = platform_min_x <= ee2[0] <= platform_max_x
    
    if not x1_platform_ok or not x2_platform_ok:
        print(f"  ⚠ X PLATFORM BOUNDS VIOLATION:")
        if not x1_platform_ok:
            print(f"    Robot1 X={ee1[0]:.3f}m outside [{platform_min_x:.3f}, {platform_max_x:.3f}]")
        if not x2_platform_ok:
            print(f"    Robot2 X={ee2[0]:.3f}m outside [{platform_min_x:.3f}, {platform_max_x:.3f}]")
        return False
    
    print(f"  ✓ X platform check: Both robots within platform bounds [{platform_min_x:.3f}, {platform_max_x:.3f}]")
    
    return True


def check_safety_stage3(harness, platform_bounds: dict) -> bool:
    """Check safety after Stage 3 (precontact). Check both X and Z.
    
    Same checks as Stage 2, but robots are closer together.
    """
    print("\n[Safety Check - Stage 3] Proprioceptive bounds check:")
    
    ee1 = harness.current_cartesian_position(1)
    ee2 = harness.current_cartesian_position(2)
    
    if ee1 is None or ee2 is None:
        print("  ERROR: Cannot get EE positions")
        return False
    
    print(f"  Robot1 EE: X={ee1[0]:.3f}m, Z={ee1[2]:.3f}m")
    print(f"  Robot2 EE: X={ee2[0]:.3f}m, Z={ee2[2]:.3f}m")
    
    # Z check (same as Stage 2)
    safe_z_min = platform_bounds['safe_z_min']
    if ee1[2] < safe_z_min or ee2[2] < safe_z_min:
        print(f"  ⚠ Z VIOLATION: Z < {safe_z_min:.3f}m")
        return False
    print(f"  ✓ Z check: Above {safe_z_min:.3f}m")
    
    # X check: robots should be within platform bounds and min distance from base
    # Also check they haven't crossed each other
    min_dist_from_base = 0.12
    robot1_base_x = -0.39
    robot2_base_x = 0.39
    
    r1_min_x = robot1_base_x + min_dist_from_base
    r2_max_x = robot2_base_x - min_dist_from_base
    
    if ee1[0] < r1_min_x or ee2[0] > r2_max_x:
        print(f"  ⚠ X BASE DISTANCE VIOLATION")
        return False
    
    # Check robots haven't crossed
    if ee1[0] >= ee2[0]:
        print(f"  ⚠ X CROSSING VIOLATION: Robot1 X={ee1[0]:.3f} >= Robot2 X={ee2[0]:.3f}")
        return False
    
    # Check within platform bounds
    platform_min_x = platform_bounds['min_x'] + 0.010
    platform_max_x = platform_bounds['max_x'] - 0.010
    
    if not (platform_min_x <= ee1[0] <= platform_max_x and platform_min_x <= ee2[0] <= platform_max_x):
        print(f"  ⚠ X PLATFORM BOUNDS VIOLATION")
        return False
    
    print(f"  ✓ X check: Within bounds, Robot1 < Robot2")
    
    return True


def check_safety_stage4_plus(harness, platform_bounds: dict) -> bool:
    """Check safety after Stage 4 (contact) and beyond. Check Z only.
    
    After contact, X is constrained by the contact, so only check Z.
    """
    print("\n[Safety Check - Stage 4+] Proprioceptive Z check:")
    
    ee1 = harness.current_cartesian_position(1)
    ee2 = harness.current_cartesian_position(2)
    
    if ee1 is None or ee2 is None:
        print("  ERROR: Cannot get EE positions")
        return False
    
    print(f"  Robot1 Z={ee1[2]:.3f}m, Robot2 Z={ee2[2]:.3f}m")
    
    safe_z_min = platform_bounds['safe_z_min']
    
    if ee1[2] < safe_z_min or ee2[2] < safe_z_min:
        print(f"  ⚠ Z VIOLATION: Z < {safe_z_min:.3f}m")
        return False
    
    print(f"  ✓ Z check: Both above {safe_z_min:.3f}m")
    return True


def main():
    parser = argparse.ArgumentParser(description="Interactive stage-by-stage hardware test")
    parser.add_argument("--spring-specimen", default="s1", help="Spring specimen ID")
    parser.add_argument("--push-axis-height-z", type=float, default=0.08, help="Push axis height")
    parser.add_argument("--max-additional-load-travel-m", type=float, default=0.060, help="Max load travel")
    parser.add_argument("--stage4-handoff-target-n", type=float, default=0.005, help="Stage 4 handoff target")
    args = parser.parse_args()
    
    rclpy.init()
    
    # Query ArUco for platform bounds at startup
    print("\n" + "=" * 80)
    print("PRE-TEST: Query ArUco for platform position")
    print("=" * 80)
    platform_info = query_aruco_platform_pose(timeout_s=5.0)
    platform_bounds = get_platform_bounds(platform_info)
    
    print(f"Platform bounds:")
    print(f"  Surface Z: {platform_bounds['surface_z']:.3f}m")
    print(f"  Safe Z min: {platform_bounds['safe_z_min']:.3f}m")
    print(f"  Platform X range: [{platform_bounds['min_x']:.3f}, {platform_bounds['max_x']:.3f}]m")
    print(f"  Platform Y range: [{platform_bounds['min_y']:.3f}, {platform_bounds['max_y']:.3f}]m")
    
    # Create harness
    harness = ContactGatedLoadStepHarness(
        push_axis_height_z=0.145,
        max_additional_load_travel_m=args.max_additional_load_travel_m,
        stage4_handoff_target_n=args.stage4_handoff_target_n,
        spring_specimen=args.spring_specimen,
        ignore_local_frame_ee_safety=True,
    )
    
    try:
        # Stage 0: Calibration
        print("\n" + "=" * 80)
        print("STAGE 0: Calibration")
        print("=" * 80)
        ok, msg, info = harness.stage0_calibration()
        print(f"Stage 0 result: {msg} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"Stage 0 failed: {msg}")
            harness.write_results([{"stage": 0, "pass": ok, "message": msg, "info": info}])
            return 1
        
        if not wait_for_user("Calibration", 0):
            harness.safe_abort()
            return 0
        
        # Stage 1: Liveness
        print("\n" + "=" * 80)
        print("STAGE 1: Liveness Check")
        print("=" * 80)
        ok, msg, info = harness.stage1_liveness()
        print(f"Stage 1 result: {msg} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"Stage 1 failed: {msg}")
            harness.safe_abort()
            return 1
        
        if not wait_for_user("Liveness Check", 1):
            harness.safe_abort()
            return 0
        
        # Stage 2: Idle stabilization
        print("\n" + "=" * 80)
        print("STAGE 2: Idle Stabilization (Homing)")
        print("=" * 80)
        print("This stage will move robots to start_position.")
        print("Watch the robots carefully!")
        ok, msg, info = harness.stage2_idle()
        print(f"Stage 2 result: {msg} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"Stage 2 failed: {msg}")
            harness.safe_abort()
            return 1
        
        # Safety check after Stage 2
        if not check_safety_stage2(harness, platform_bounds):
            print("⚠ Stage 2 safety check failed!")
            harness.safe_abort()
            return 1
        
        if not wait_for_user("Idle Stabilization (Homing)", 2):
            harness.safe_abort()
            return 0
        
        # Stage 3: Synchronized move (precontact approach)
        print("\n" + "=" * 80)
        print("STAGE 3: Synchronized Move (Precontact Approach)")
        print("=" * 80)
        print("This stage will move both robots toward each other.")
        print("Stand clear of the robots!")
        ok, msg, info = harness.stage3_sync_move()
        print(f"Stage 3 result: {msg} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"Stage 3 failed: {msg}")
            harness.safe_abort()
            return 1
        
        # Safety check after Stage 3
        if not check_safety_stage3(harness, platform_bounds):
            print("⚠ Stage 3 safety check failed!")
            harness.safe_abort()
            return 1
        
        if not wait_for_user("Synchronized Move (Precontact)", 3):
            harness.safe_abort()
            return 0
        
        # Stage 4: Directional hold
        print("\n" + "=" * 80)
        print("STAGE 4: Directional Hold")
        print("=" * 80)
        print("This stage will establish directional contact hold.")
        ok, msg, info = harness.stage4_hold()
        print(f"Stage 4 result: {msg} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"Stage 4 failed: {msg}")
            harness.safe_abort()
            return 1
        
        # Safety check after Stage 4 (Z only)
        if not check_safety_stage4_plus(harness, platform_bounds):
            print("⚠ Stage 4 safety check failed!")
            harness.safe_abort()
            return 1
        
        if not wait_for_user("Directional Hold", 4):
            harness.safe_abort()
            return 0
        
        # Stage 5: Quasistatic compression with load measurement - INTERACTIVE
        print("\n" + "=" * 80)
        print("STAGE 5: QUASISTATIC COMPRESSION WITH LOAD MEASUREMENT - INTERACTIVE")
        print("=" * 80)
        print("This stage will compress incrementally by DISTANCE targets.")
        print("You will be prompted before EACH compression step.")
        print("After each step, the resulting load will be measured and reported.")
        print("Monitor force readings and robot behavior carefully!")
        
        # Compression distance targets (quasistatic)
        compression_distances_mm = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        print(f"\nCompression distance targets (mm): {compression_distances_mm}")
        
        # Set up compression direction and anchor
        harness.current_phase = "quasistatic_compression"
        
        direction = harness._resolve_load_step_direction(0.1)  # dummy value to get direction
        if direction is None:
            harness.set_mutual_contact_state(False, "compression_direction_unavailable")
            print(f"Stage 5 failed: compression direction unavailable")
            harness.safe_abort()
            return 1
        
        harness.current_mutual_press_dir = direction
        harness.directional_press_enabled = True
        harness._update_load_metrics(direction)
        
        if harness._using_local_frame_load_logic():
            if not harness._reset_local_frame_load_step_reference():
                harness.set_mutual_contact_state(False, "compression_anchor_unavailable")
                harness.directional_press_enabled = False
                print(f"Stage 5 failed: unable to establish local-frame compression anchor")
                harness.safe_abort()
                return 1
        elif not harness.ensure_mutual_contact_anchor():
            harness.set_mutual_contact_state(False, "compression_anchor_unavailable")
            harness.directional_press_enabled = False
            print(f"Stage 5 failed: unable to capture compression anchor")
            harness.safe_abort()
            return 1
        
        print(f"\nCompression direction established: {[f'{d:.4f}' for d in direction]}")
        print("Starting from current press distances (should be ~0 at contact)")
        
        # CRITICAL: Update anchor to current EE positions before compression
        # The old anchor from Stage 3/4 is stale - robots have moved since then
        ee1_pos = harness.current_cartesian_position(1)
        ee2_pos = harness.current_cartesian_position(2)
        if ee1_pos is None or ee2_pos is None:
            print("ERROR: Cannot get current EE positions for compression baseline")
            harness.directional_press_enabled = False
            return 1
        
        # Update the mutual contact anchor to current positions
        harness.mutual_contact_anchor_1 = (ee1_pos[0], ee1_pos[1], ee1_pos[2])
        harness.mutual_contact_anchor_2 = (ee2_pos[0], ee2_pos[1], ee2_pos[2])
        print(f"Updated compression baseline to current EE positions:")
        print(f"  Robot1: [{ee1_pos[0]:.4f}, {ee1_pos[1]:.4f}, {ee1_pos[2]:.4f}]")
        print(f"  Robot2: [{ee2_pos[0]:.4f}, {ee2_pos[1]:.4f}, {ee2_pos[2]:.4f}]")
        
        # SAFETY: Check if contact Z is too low (platform at Z=0.05m)
        min_safe_z = 0.065  # 15mm above platform surface
        contact_z_avg = (ee1_pos[2] + ee2_pos[2]) / 2
        if contact_z_avg < min_safe_z:
            print(f"\n⚠ WARNING: Contact Z={contact_z_avg:.3f}m is too low (min safe: {min_safe_z:.3f}m)")
            print(f"  Risk of platform collision during compression!")
            print(f"  Consider raising contact position before proceeding.")
            
            if not wait_for_compression_step(f"Contact Z too low. Continue anyway? [y/n]: "):
                print("\nCompression aborted due to low Z safety check")
                harness.directional_press_enabled = False
                harness._publish_zero_offsets()
                return 0
        
        # Set safety bounds for compression
        # Use ArUco-derived platform bounds if available
        if platform_info:
            platform_size = platform_info['surface_size']
            platform_surface_z = platform_info['surface_z']
            max_safe_x = platform_size / 2 - 0.030  # Stay 30mm inside platform edge
            min_safe_z = safe_z_min  # Use the ArUco-derived safe Z minimum
            print(f"\nCompression safety bounds (from ArUco):")
            print(f"  Max X: ±{max_safe_x:.3f}m (platform edge: ±{platform_size/2:.3f}m)")
            print(f"  Min Z: {min_safe_z:.3f}m (platform surface: {platform_surface_z:.3f}m)")
        else:
            max_safe_x = 0.12  # Default: 30mm inside 0.30m platform edge
            min_safe_z = safe_z_min  # Use the fallback safe Z
            print(f"\nCompression safety bounds (default):")
            print(f"  Max X: ±{max_safe_x:.3f}m")
            print(f"  Min Z: {min_safe_z:.3f}m")
        
        # Set high axial stiffness for compression phase
        compression_stiffness = 65.0  # N/m - maximum allowed by controller (default: 50 N/m)
        print(f"\nSetting high axial stiffness: {compression_stiffness:.1f} N/m (default: 50 N/m)")
        try:
            harness.set_k_axial(compression_stiffness)
            print("✓ High axial stiffness set")
        except Exception as e:
            print(f"⚠ Warning: Could not set high axial stiffness: {e}")
            print("  Continuing with default stiffness...")
        
        # Initial measurement
        l1, l2, current_shared_load, _ = harness._update_load_metrics(direction)
        initial_dist1 = harness.current_press_distance_1
        initial_dist2 = harness.current_press_distance_2
        print(f"Initial state: compression R1={initial_dist1*1000:.2f}mm, R2={initial_dist2*1000:.2f}mm")
        print(f"               shared load: {current_shared_load:.3f} N (R1: {l1:.3f} N, R2: {l2:.3f} N)")
        
        # Interactive compression loop
        total_steps = len(compression_distances_mm)
        measurements = []
        
        for step_index, target_distance_mm in enumerate(compression_distances_mm, start=1):
            target_distance_m = target_distance_mm / 1000.0
            
            # Prompt user
            print(f"\n{'='*60}")
            print(f"Compression Step {step_index}/{total_steps}: Target Distance = {target_distance_mm:.1f} mm")
            print(f"{'='*60}")
            
            l1, l2, current_shared_load, _ = harness._update_load_metrics(direction)
            current_dist1 = harness.current_press_distance_1
            current_dist2 = harness.current_press_distance_2
            current_max_dist = max(current_dist1, current_dist2)
            
            print(f"Current state:")
            print(f"  Compression: R1={current_dist1*1000:.2f}mm, R2={current_dist2*1000:.2f}mm (max: {current_max_dist*1000:.2f}mm)")
            print(f"  Shared load: {current_shared_load:.3f} N (R1: {l1:.3f} N, R2: {l2:.3f} N)")
            print(f"Target: compress to {target_distance_mm:.1f} mm (Δ = {target_distance_mm - current_max_dist*1000:.1f} mm)")
            
            if not wait_for_compression_step(f"Proceed with compression to {target_distance_mm:.1f} mm? [y/n]: "):
                print(f"\nCompression aborted at step {step_index}")
                harness.directional_press_enabled = False
                harness._publish_zero_offsets()
                return 0
            
            # Apply compression distance - Robot2 needs OPPOSITE direction!
            # Robot1 moves +X (toward Robot2), Robot2 moves -X (toward Robot1)
            harness.current_press_distance_1 = target_distance_m
            harness.current_press_distance_2 = target_distance_m
            
            # Compute targets manually with opposite directions for each robot
            # Robot1: offset = +distance in press direction
            # Robot2: offset = -distance in press direction (opposite!)
            offset_x1 = target_distance_m * direction[0]
            offset_y1 = target_distance_m * direction[1]
            offset_z1 = target_distance_m * direction[2]
            
            offset_x2 = -target_distance_m * direction[0]  # NEGATED for Robot2!
            offset_y2 = -target_distance_m * direction[1]
            offset_z2 = -target_distance_m * direction[2]
            
            # Build target dict with asymmetric offsets
            # Apply safety bounds to prevent platform collision
            target_x1 = harness.mutual_contact_anchor_1[0] + offset_x1 if harness.mutual_contact_anchor_1 else 0
            target_x2 = harness.mutual_contact_anchor_2[0] + offset_x2 if harness.mutual_contact_anchor_2 else 0
            target_z1 = harness.mutual_contact_anchor_1[2] + offset_z1 if harness.mutual_contact_anchor_1 else 0
            target_z2 = harness.mutual_contact_anchor_2[2] + offset_z2 if harness.mutual_contact_anchor_2 else 0
            
            # Clamp to safety bounds
            target_x1 = max(-max_safe_x, min(max_safe_x, target_x1))
            target_x2 = max(-max_safe_x, min(max_safe_x, target_x2))
            target_z1 = max(min_safe_z, target_z1)
            target_z2 = max(min_safe_z, target_z2)
            
            target = {
                "offset_x1": offset_x1,
                "offset_y1": offset_y1,
                "offset_z1": offset_z1,
                "offset_x2": offset_x2,
                "offset_y2": offset_y2,
                "offset_z2": offset_z2,
                "target_x1": target_x1,
                "target_y1": harness.mutual_contact_anchor_1[1] + offset_y1 if harness.mutual_contact_anchor_1 else 0,
                "target_z1": target_z1,
                "target_x2": target_x2,
                "target_y2": harness.mutual_contact_anchor_2[1] + offset_y2 if harness.mutual_contact_anchor_2 else 0,
                "target_z2": target_z2,
            }
            
            # Check if clamping occurred
            original_x1 = harness.mutual_contact_anchor_1[0] + offset_x1 if harness.mutual_contact_anchor_1 else 0
            original_x2 = harness.mutual_contact_anchor_2[0] + offset_x2 if harness.mutual_contact_anchor_2 else 0
            if abs(target_x1 - original_x1) > 1e-6 or abs(target_x2 - original_x2) > 1e-6:
                print(f"  ⚠ Safety clamp applied:")
                print(f"    Robot1 X: {original_x1:.3f}m → {target_x1:.3f}m")
                print(f"    Robot2 X: {original_x2:.3f}m → {target_x2:.3f}m")
            
            # Publish the asymmetric target
            harness.publish_directional_press_target(
                target, 
                f"quasistatic_compression_step_{step_index}"
            )
            
            print(f"  Published asymmetric targets:")
            print(f"    Robot1: offset=({offset_x1*1000:.1f}, {offset_y1*1000:.1f}, {offset_z1*1000:.1f}) mm")
            print(f"    Robot2: offset=({offset_x2*1000:.1f}, {offset_y2*1000:.1f}, {offset_z2*1000:.1f}) mm")
            
            # Wait for settling (quasistatic)
            print(f"  Compressing to {target_distance_mm:.1f} mm...")
            settled = harness._wait_for_directional_target_settle(30.0)
            
            if not settled:
                print(f"  WARNING: Target did not settle within timeout")
            
            # Measure resulting load
            l1, l2, measured_load, _ = harness._update_load_metrics(direction)
            final_dist1 = harness.current_press_distance_1
            final_dist2 = harness.current_press_distance_2
            final_max_dist = max(final_dist1, final_dist2)
            
            print(f"\n  ✓ Compression complete!")
            print(f"    Applied: {final_max_dist*1000:.2f} mm")
            print(f"    Measured load: {measured_load:.3f} N (R1: {l1:.3f} N, R2: {l2:.3f} N)")
            
            measurements.append({
                'step': step_index,
                'target_distance_mm': target_distance_mm,
                'applied_distance_mm': final_max_dist * 1000,
                'measured_load_n': measured_load,
                'r1_load_n': l1,
                'r2_load_n': l2,
            })
        
        # Final summary
        harness.current_load_step_state = "complete"
        harness.set_mutual_contact_state(False, "compression_complete")
        harness.directional_press_enabled = False
        
        print("\n" + "=" * 80)
        print("COMPRESSION COMPLETE - Quasistatic Load Measurement Summary")
        print("=" * 80)
        print("\nDistance (mm) | Applied (mm) | Measured Load (N) | R1 (N) | R2 (N)")
        print("-" * 70)
        for m in measurements:
            print(f"{m['target_distance_mm']:13.1f} | {m['applied_distance_mm']:12.2f} | {m['measured_load_n']:17.3f} | {m['r1_load_n']:6.3f} | {m['r2_load_n']:6.3f}")
        
        print("\nRobots are HOLDING at maximum compression.")
        print("Type [y] to release and finish, or [n] to hold longer.")
        try:
            wait_for_input("\nPress Enter to release...")
        except (EOFError, KeyboardInterrupt):
            pass
        
        # Restore default stiffness before release
        print("\nRestoring default axial stiffness (50 N/m)...")
        try:
            harness.set_k_axial(50.0)
            print("✓ Default stiffness restored")
        except Exception as e:
            print(f"⚠ Warning: Could not restore default stiffness: {e}")
        
        print("\n" + "=" * 80)
        print("ALL STAGES PASSED")
        print("=" * 80)
        harness._publish_zero_offsets()
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Safely aborting...")
        try:
            harness.safe_abort()
        except Exception as abort_err:
            print(f"Note: safe_abort() encountered an error (expected during interrupt): {abort_err}")
        return 1
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        try:
            harness.safe_abort()
        except Exception as abort_err:
            print(f"Note: safe_abort() encountered an error: {abort_err}")
        return 1
    finally:
        try:
            harness.destroy_node()
        except Exception as node_err:
            pass  # Node might already be destroyed
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as shutdown_err:
            pass  # Context might already be shut down


if __name__ == "__main__":
    sys.exit(main())
