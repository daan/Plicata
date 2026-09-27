"""
drums.py -- one motor, two winches on a common shaft.

Opening the ring (t increasing) pays out the contraction cable and takes in the
expansion cable. With both drums on one shaft turning by theta, the local
winding radii must satisfy

    dLc/dtheta =  r_c(theta)        (contraction drum pays out)
    dLe/dtheta = -r_e(theta)        (expansion drum takes in)

Lc(t), Le(t) come from hoberman.length_curve. The free choice is the map
theta(t); each mode fixes it differently:

  'single'     contraction drum round (r_c = R), expansion drum shaped.
  'geometric'  r_c * r_e = R^2 : log-radius variation split evenly.
  'sum'        r_c + r_e = 2R  : like a differential, one grows as the other shrinks.

Results are polar spirals r(theta). For theta_max <= 2*pi this is directly a
cam-shaped pulley profile (with the usual cam approximation that the lever arm
equals the polar radius, good when r varies slowly). For more than one turn
it is the radius schedule of a helical groove; loft it in CAD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import hoberman as H


@dataclass
class Drum:
    name: str
    theta: np.ndarray      # shaft angle [rad], 0 at t_min
    r: np.ndarray          # winding radius [mm]
    t: np.ndarray          # matching deployment

    @property
    def turns(self) -> float:
        return float(self.theta[-1] / (2 * np.pi))

    def profile(self, n_per_turn: int = 180):
        """Polar spiral as xy points (plan view)."""
        th = np.linspace(self.theta[0], self.theta[-1], max(50, int(self.turns * n_per_turn) + 2))
        rr = np.interp(th, self.theta, self.r)
        return np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)

    def summary(self) -> str:
        return (f"{self.name}: r {self.r.min():.2f} .. {self.r.max():.2f} mm "
                f"(ratio {self.r.max()/self.r.min():.2f}), {self.turns:.2f} turns, "
                f"{np.degrees(self.theta[-1]):.0f} deg total")


def design(p: H.Params, con: H.Cable, exp: H.Cable, R: float = 15.0, mode: str = "single",
           n: int = 300):
    """Return (drum_contraction, drum_expansion, info dict)."""
    t = np.linspace(p.t_min, p.t_max, n)
    Lc = H.length_curve(p, con, t)
    Le = H.length_curve(p, exp, t)
    dLc = np.gradient(Lc, t)            # > 0
    dLe = -np.gradient(Le, t)           # > 0 (take-in rate)
    if np.any(dLc <= 0) or np.any(dLe <= 0):
        raise ValueError("payout rates change sign inside the t range; a single shaft cannot do this")
    q = dLc / dLe                       # ratio of the two rates

    if mode == "single":
        rc = np.full_like(t, R)
        re = R / q
    elif mode == "geometric":
        rc = R * np.sqrt(q)
        re = R / np.sqrt(q)
    elif mode == "sum":
        # rc/re = q and rc + re = 2R
        re = 2 * R / (1 + q)
        rc = q * re
    else:
        raise ValueError(mode)

    dtheta = dLc / rc                   # dtheta/dt
    theta = np.concatenate([[0.0], np.cumsum(0.5 * (dtheta[1:] + dtheta[:-1]) * np.diff(t))])
    info = {
        "payout_contraction": Lc[-1] - Lc[0],
        "takeup_expansion": Le[0] - Le[-1],
        "ratio_min": float(q.min()),
        "ratio_max": float(q.max()),
        "theta_max_deg": float(np.degrees(theta[-1])),
    }
    return Drum("contraction drum", theta, rc, t), Drum("expansion drum", theta, re, t), info


def plot(dc: Drum, de: Drum, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(np.degrees(dc.theta), dc.r, c="#c0392b", label=dc.name)
    ax[0].plot(np.degrees(de.theta), de.r, c="#2471a3", label=de.name)
    ax[0].set_xlabel("shaft angle [deg]")
    ax[0].set_ylabel("winding radius [mm]")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    for a, d, col in ((ax[1], dc, "#c0392b"), (ax[2], de, "#2471a3")):
        xy = d.profile()
        a.plot(xy[:, 0], xy[:, 1], c=col)
        a.plot(*xy[0], "o", c=col)
        a.plot(*xy[-1], "s", c=col)
        a.plot(0, 0, "k+")
        a.set_aspect("equal")
        a.set_title(f"{d.name}: {d.turns:.2f} turns")
        a.grid(alpha=.3)
    return ax


def export_dxf(path: str, dc: Drum, de: Drum, spacing: float = 80.0):
    import ezdxf
    from ezdxf import units
    doc = ezdxf.new("R2010", setup=True)
    doc.units = units.MM
    msp = doc.modelspace()
    for j, d in enumerate((dc, de)):
        lay = d.name.upper().replace(" ", "_")
        doc.layers.add(lay, color=1 if j == 0 else 5)
        xy = d.profile() + (j * spacing, 0.0)
        msp.add_lwpolyline(xy.tolist(), dxfattribs={"layer": lay})
        msp.add_circle((j * spacing, 0.0), 1.0, dxfattribs={"layer": lay})
        msp.add_text(d.summary(), height=2.5, dxfattribs={"layer": lay}).set_placement(
            (j * spacing - d.r.max(), -d.r.max() - 6))
    doc.saveas(path)
    return path
