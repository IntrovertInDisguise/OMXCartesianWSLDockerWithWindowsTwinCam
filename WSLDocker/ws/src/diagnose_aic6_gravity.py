#!/usr/bin/env python3
"""
diagnose_aic6_gravity.py

READ-ONLY / NO-HARDWARE diagnostic for AIC6/AIC5.

It does two things:
  1) Fits measured calibration hold current against KDL gravity torque,
     with conditioning checks so nearly-constant joints are not misread.
  2) Expands the standard OpenMANIPULATOR-X xacro IN MEMORY and inventories
     which inertial links are on the current root->tip serial KDL chain and
     which inertial branch links are omitted by ChainDynParam.

It never opens the Dynamixel port and never writes the xacro/URDF.

Usage:
    python3 diagnose_aic6_gravity.py
    python3 diagnose_aic6_gravity.py --module AIC5
    python3 diagnose_aic6_gravity.py --exclude-waypoint 11
"""
import argparse
import importlib
import math


def fit_line(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-14:
        return float("nan"), float("nan"), float("nan"), sxx
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    pred = [a * x + b for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else float("nan")
    return a, b, r2, sxx


def fit_scale_only(xs, ys):
    sxx = sum(x * x for x in xs)
    if sxx <= 1e-14:
        return float("nan"), float("nan")
    a = sum(x * y for x, y in zip(xs, ys)) / sxx
    pred = [a * x for x in xs]
    my = sum(ys) / len(ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else float("nan")
    return a, r2


def mean_std(vals):
    m = sum(vals) / len(vals)
    if len(vals) > 1:
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    else:
        s = float("nan")
    return m, s


def expand_robot(aic):
    import xacro
    from urdf_parser_py.urdf import URDF

    path = aic._resolve_omx_xacro()
    doc = xacro.parse(None, path)
    root = doc.documentElement

    invocation = doc.createElementNS(
        "http://www.ros.org/wiki/xacro", "xacro:open_manipulator_x")
    invocation.setAttribute("prefix", "")
    root.appendChild(invocation)
    xacro.process_doc(doc)

    root = doc.documentElement
    if not root.hasAttribute("name") or not root.getAttribute("name").strip():
        root.setAttribute("name", "open_manipulator_x_gravity_diagnostic")

    return URDF.from_xml_string(doc.toxml()), path


def ancestry(robot, link_name):
    """Return root->link links and joints for one serial ancestry path."""
    links = [link_name]
    joints = []
    cur = link_name
    seen = set()
    while cur in robot.parent_map:
        if cur in seen:
            raise RuntimeError("Cycle in URDF ancestry")
        seen.add(cur)
        joint_name, parent_name = robot.parent_map[cur]
        joints.append(joint_name)
        links.append(parent_name)
        cur = parent_name
    links.reverse()
    joints.reverse()
    return links, joints


def mass_of(link):
    if link.inertial is None:
        return 0.0
    return float(link.inertial.mass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="AIC6")
    ap.add_argument("--exclude-waypoint", type=int, nargs="*", default=[11])
    args = ap.parse_args()

    aic = importlib.import_module(args.module)
    excluded = set(args.exclude_waypoint)

    print("=" * 72)
    print("AIC GRAVITY MODEL DIAGNOSTIC — READ ONLY / NO HARDWARE")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------
    samples = aic.load_calibration()
    model = aic.KDLGravityModel()

    rows = []
    for i, sample in enumerate(samples, 1):
        if i in excluded:
            continue
        nm = model.gravity_nm(sample["q"])
        measured = sample["gravity_current"]
        for j in range(aic.NJ):
            rows.append((i, j, float(nm[j]), float(measured[j])))

    print(f"\nCalibration file: {aic.CALIB_FILE}")
    print(f"Samples loaded: {len(samples)}; used: {len(samples)-len(excluded)}")
    print(f"Excluded waypoints: {sorted(excluded) if excluded else 'none'}")
    print(f"Assumed TORQUE_SCALE: {aic.TORQUE_SCALE:.3f} raw/N.m")

    print("\nPer-joint fit: measured_raw = a*KDL_Nm + b")
    print("(x-span is shown because a slope is meaningless if gravity torque "
          "barely changes over the sampled path.)\n")

    identifiable = []
    for j, name in enumerate(aic.JOINT_NAMES):
        xs = [nm for (wp, jj, nm, meas) in rows if jj == j]
        ys = [meas for (wp, jj, nm, meas) in rows if jj == j]
        xspan_nm = max(xs) - min(xs)
        xspan_raw_at_assumed_scale = abs(xspan_nm * aic.TORQUE_SCALE)
        residuals_assumed = [
            y - aic.TORQUE_SCALE * x for x, y in zip(xs, ys)
        ]
        rmean, rstd = mean_std(residuals_assumed)

        # Require at least ~5 raw units of model-gravity excitation before
        # interpreting a per-joint slope.
        conditioned = xspan_raw_at_assumed_scale >= 5.0

        if conditioned:
            a, b, r2, _ = fit_line(xs, ys)
            identifiable.append(j)
            print(
                f"  {name}: a={a:8.2f} raw/N.m  b={b:+8.2f} raw  "
                f"R^2={r2:6.3f}  "
                f"KDL span={xspan_nm:+.5f} N.m "
                f"(~{xspan_raw_at_assumed_scale:.1f} raw)"
            )
        else:
            print(
                f"  {name}: SLOPE NOT IDENTIFIABLE on this calibration path; "
                f"KDL span={xspan_nm:+.5f} N.m "
                f"(~{xspan_raw_at_assumed_scale:.2f} raw)"
            )

        print(
            f"           residual at assumed scale: mean={rmean:+.2f} raw, "
            f"std={rstd:.2f}, range=[{min(residuals_assumed):+.2f},"
            f"{max(residuals_assumed):+.2f}]"
        )

    xs_pool, ys_pool = [], []
    for (wp, j, nm, meas) in rows:
        if j in identifiable:
            xs_pool.append(nm)
            ys_pool.append(meas)
    if xs_pool:
        a0, r20 = fit_scale_only(xs_pool, ys_pool)
        print(
            f"\nPooled forced-origin fit over IDENTIFIABLE joints only: "
            f"TORQUE_SCALE~{a0:.2f} raw/N.m, R^2={r20:.3f}"
        )
        print("Do not interpret this pooled scale as a fix if the per-joint "
              "intercepts are materially nonzero.")

    # ------------------------------------------------------------------
    # URDF chain / branch inventory
    # ------------------------------------------------------------------
    robot, xacro_path = expand_robot(aic)
    root = aic.KDL_ROOT_LINK
    tip = aic.KDL_TIP_LINK

    links_path, joints_path = ancestry(robot, tip)
    if root not in links_path:
        raise RuntimeError(
            f"Configured root {root!r} is not an ancestor of tip {tip!r}")
    root_i = links_path.index(root)
    chain_links = links_path[root_i:]
    chain_joints = joints_path[root_i:]
    chain_set = set(chain_links)

    print("\n" + "=" * 72)
    print("URDF SERIAL-CHAIN / BRANCH-MASS INVENTORY")
    print("=" * 72)
    print(f"xacro: {xacro_path}")
    print(f"Configured KDL chain: {root} -> {tip}")
    print("Serial ancestry:")
    for i, lname in enumerate(chain_links):
        link = robot.link_map[lname]
        print(f"  LINK {lname:24s} mass={mass_of(link):.6f} kg")
        if i < len(chain_joints):
            jn = chain_joints[i]
            joint = robot.joint_map[jn]
            print(f"       -> joint {jn:24s} type={joint.type}")

    print("\nChildren of link5:")
    for joint_name, child_name in robot.child_map.get("link5", []):
        joint = robot.joint_map[joint_name]
        child = robot.link_map[child_name]
        status = "ON current chain" if child_name in chain_set else "OFF current chain"
        print(
            f"  {joint_name:28s} ({joint.type:9s}) -> "
            f"{child_name:24s} mass={mass_of(child):.6f} kg  [{status}]"
        )

    print("\nInertial links omitted from the current serial chain:")
    omitted_mass = 0.0
    for link in robot.links:
        m = mass_of(link)
        if m <= 0.0 or link.name in chain_set:
            continue

        # Find the nearest ancestor that lies on the arm chain.
        cur = link.name
        attach = None
        via = []
        while cur in robot.parent_map:
            joint_name, parent_name = robot.parent_map[cur]
            via.append(joint_name)
            if parent_name in chain_set:
                attach = parent_name
                break
            cur = parent_name

        if attach is not None:
            omitted_mass += m
            types = [robot.joint_map[jn].type for jn in reversed(via)]
            print(
                f"  {link.name:24s} mass={m:.6f} kg  "
                f"branches from {attach:18s} via "
                f"{list(reversed(via))} types={types}"
            )

    print(f"Total inertial mass in omitted branches attached to current chain: "
          f"{omitted_mass:.6f} kg")

    print("\nCandidate-tip warning:")
    for lname in [l.name for l in robot.links if "gripper" in l.name.lower()]:
        try:
            lp, jp = ancestry(robot, lname)
            if root not in lp:
                continue
            ri = lp.index(root)
            movable = [
                jn for jn in jp[ri:]
                if robot.joint_map[jn].type not in ("fixed", "floating")
            ]
            print(
                f"  tip={lname:24s}: movable joints root->tip = "
                f"{len(movable)} {movable}"
            )
        except Exception as exc:
            print(f"  tip={lname}: could not inspect: {exc}")

    print("\nInterpretation:")
    print("  * A serial ChainDynParam only sees inertias on its selected path.")
    print("  * A sibling gripper branch off link5 is not made visible merely by "
          "ending the chain at end_effector_link.")
    print("  * If a gripper tip adds a prismatic gripper DOF, it is not a safe "
          "drop-in replacement for AIC's 4-arm-joint gravity mapping.")
    print("  * Prefer an internal/in-memory branch-inertia aggregation or full-"
          "tree arm-gravity calculation; do not edit the standard xacro.")


if __name__ == "__main__":
    main()
