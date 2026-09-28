#!/usr/bin/env python3
import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path, 'r') as f:
        r = csv.reader(f)
        header = next(r)
        data = []
        for row in r:
            if not row:
                continue
            data.append([float(x) for x in row])
    if not data:
        return header, np.zeros((0, len(header)))
    return header, np.array(data)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def plot_ee(robot, base_dir, out_dir):
    desired_f = os.path.join(base_dir, f'{robot}_{robot}_variable_stiffness_cartesian_pose_desired.csv')
    actual_f = os.path.join(base_dir, f'{robot}_{robot}_variable_stiffness_end_effector_position.csv')
    contact_f = os.path.join(base_dir, f'{robot}_{robot}_variable_stiffness_contact_wrench.csv')

    h_d, d = load_csv(desired_f)
    h_a, a = load_csv(actual_f)
    h_c, c = load_csv(contact_f)

    if d.size == 0 or a.size == 0:
        print('missing data for', robot)
        return

    t0 = min(d[0,0], a[0,0])
    td = d[:,0] - t0
    ta = a[:,0] - t0

    # pick x,y,z columns
    dx = d[:,1]
    dy = d[:,2]
    dz = d[:,3]

    ax_x = a[:,1]
    ax_y = a[:,2]
    ax_z = a[:,3]

    # plot desired vs actual x
    ensure_dir(out_dir)
    plt.figure(figsize=(8,3))
    plt.plot(td, dx, label='EE_des_x')
    plt.plot(ta, ax_x, label='EE_act_x')
    plt.xlabel('time (s)')
    plt.ylabel('x (m)')
    plt.title(f'{robot} EE x desired vs actual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{robot}_ee_x.png'))
    plt.close()

    # plot Euclidean pos error norm
    # resample actual to desired times by nearest
    # interpolate actual to desired timebase using numpy (no SciPy required)
    # ensure ta is strictly increasing for np.interp
    order = np.argsort(ta)
    ta_sorted = ta[order]
    ax_x_sorted = ax_x[order]
    ax_y_sorted = ax_y[order]
    ax_z_sorted = ax_z[order]
    ax_x_r = np.interp(td, ta_sorted, ax_x_sorted)
    ax_y_r = np.interp(td, ta_sorted, ax_y_sorted)
    ax_z_r = np.interp(td, ta_sorted, ax_z_sorted)
    err = np.sqrt((dx-ax_x_r)**2 + (dy-ax_y_r)**2 + (dz-ax_z_r)**2)

    plt.figure(figsize=(8,3))
    plt.plot(td, err)
    plt.xlabel('time (s)')
    plt.ylabel('pos error (m)')
    plt.title(f'{robot} EE position error norm')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{robot}_ee_error.png'))
    plt.close()

    # contact forces (if present)
    if c.size != 0:
        tc = c[:,0] - t0
        fx_c = c[:,1]
        fy_c = c[:,2]
        fz_c = c[:,3]
        plt.figure(figsize=(8,3))
        plt.plot(tc, fx_c, label='fx')
        plt.plot(tc, fy_c, label='fy')
        plt.plot(tc, fz_c, label='fz')
        plt.xlabel('time (s)')
        plt.ylabel('force (N)')
        plt.title(f'{robot} contact wrench')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{robot}_contact_wrench.png'))
        plt.close()


def main():
    base = 'logs/run_mutual_contact/csv_extract'
    out = 'logs/run_mutual_contact/plots'
    ensure_dir(out)
    # numpy interp is used; no SciPy dependency required

    for robot in ['robot1', 'robot2']:
        try:
            plot_ee(robot, base, out)
        except Exception as e:
            print('failed to plot for', robot, e)

    print('plots written to', out)


if __name__ == '__main__':
    main()
