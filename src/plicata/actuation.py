"""
actuation.py -- lookup tables for two programmable servos with round winch drums.

For each cable: payout from t_min, drum angle (turns and degrees) for a given
drum radius, cable speed per unit ring motion (transmission ratio), and the
relation between the two servos. Export as CSV or JSON for firmware.

    tab = table(p, con, exp, r_drum_con=10.0, r_drum_exp=10.0)
    export_csv("servo_table.csv", tab)
"""

from __future__ import annotations

import csv
import json

import numpy as np

from . import hoberman as H


def table(p: H.Params, con: H.Cable, exp: H.Cable, r_drum_con: float, r_drum_exp: float, n: int = 101):
    t = np.linspace(p.t_min, p.t_max, n)
    rho, r_in, r_out = np.array([H.state_radii(p, tt) for tt in t]).T
    Lc = H.length_curve(p, con, t)
    Le = H.length_curve(p, exp, t)
    payout_c = Lc - Lc[0]           # cable released by the contraction servo (>= 0)
    takeup_e = Le[0] - Le            # cable wound by the expansion servo (>= 0)
    ang_c = payout_c / r_drum_con    # rad
    ang_e = takeup_e / r_drum_exp
    # transmission: mm of cable per mm of rho (kink radius, monotone in t; r_out is not near full expansion)
    dro = np.gradient(rho, t)
    ratio_c = np.gradient(Lc, t) / dro
    ratio_e = -np.gradient(Le, t) / dro
    # servo speed ratio: expansion servo speed / contraction servo speed
    speed_ratio = np.gradient(ang_e, t) / np.where(np.abs(np.gradient(ang_c, t)) > 1e-12, np.gradient(ang_c, t), np.nan)
    return {
        "t": t, "rho": rho, "r_in": r_in, "r_out": r_out,
        "L_con": Lc, "L_exp": Le,
        "payout_con_mm": payout_c, "takeup_exp_mm": takeup_e,
        "servo_con_turns": ang_c / (2 * np.pi), "servo_exp_turns": ang_e / (2 * np.pi),
        "servo_con_deg": np.degrees(ang_c), "servo_exp_deg": np.degrees(ang_e),
        "ratio_con": ratio_c, "ratio_exp": ratio_e,
        "speed_ratio_exp_over_con": speed_ratio,
        "meta": {"r_drum_con": r_drum_con, "r_drum_exp": r_drum_exp, "n_units": p.n, "arm_len": p.arm_len},
    }


def summary(tab) -> str:
    m = tab["meta"]
    return "\n".join([
        f"drums: contraction r={m['r_drum_con']} mm, expansion r={m['r_drum_exp']} mm",
        f"stroke: rho {tab['rho'][0]:.1f} -> {tab['rho'][-1]:.1f} mm, r_out {tab['r_out'][0]:.1f} -> {tab['r_out'][-1]:.1f} mm",
        f"contraction servo: {tab['payout_con_mm'][-1]:.1f} mm, {tab['servo_con_turns'][-1]:.2f} turns",
        f"expansion servo:   {tab['takeup_exp_mm'][-1]:.1f} mm, {tab['servo_exp_turns'][-1]:.2f} turns",
        f"transmission (mm cable / mm rho): contraction {tab['ratio_con'].min():.2f}..{tab['ratio_con'].max():.2f}, "
        f"expansion {tab['ratio_exp'].min():.2f}..{tab['ratio_exp'].max():.2f}",
        f"servo speed ratio exp/con: {np.nanmin(tab['speed_ratio_exp_over_con']):.2f}..{np.nanmax(tab['speed_ratio_exp_over_con']):.2f}",
    ])


def export_csv(path: str, tab):
    keys = [k for k in tab if k != "meta"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(tab["t"])):
            w.writerow([f"{tab[k][i]:.4f}" for k in keys])
    return path


def export_json(path: str, tab, keys=("t", "rho", "r_out", "servo_con_deg", "servo_exp_deg")):
    out = {"meta": tab["meta"], **{k: [round(float(v), 3) for v in tab[k]] for k in keys}}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return path


def plot(tab, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(tab["rho"], tab["servo_con_turns"], c="#c0392b", label="contraction servo")
    ax[0].plot(tab["rho"], tab["servo_exp_turns"], c="#2471a3", label="expansion servo")
    ax[0].set_xlabel("rho [mm]"); ax[0].set_ylabel("drum turns from t_min"); ax[0].legend()
    ax[1].plot(tab["rho"], tab["ratio_con"], c="#c0392b"); ax[1].plot(tab["rho"], tab["ratio_exp"], c="#2471a3")
    ax[1].set_xlabel("rho [mm]"); ax[1].set_ylabel("mm cable per mm radius")
    ax[1].set_title("transmission ratio (torque vs speed)")
    ax[2].plot(tab["servo_con_deg"], tab["servo_exp_deg"], "k")
    ax[2].set_xlabel("contraction servo [deg]"); ax[2].set_ylabel("expansion servo [deg]")
    ax[2].set_title("follower table")
    for a in ax:
        a.grid(alpha=.3)
    return ax
