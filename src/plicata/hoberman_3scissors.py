"""
Three-scissor detail view of the Hoberman ring, aligned on the center CCW link.

Reuses all parameters/geometry from hoberman_ring.py. Instead of the full ring,
only 3 consecutive scissor units (6 links) are drawn per deployment state. Each
state is rigidly transformed so that the CCW-rotating link of the middle unit
is always at the same pose: kink pivot at the origin, bisector along +x.

"CCW link" = the link family whose body rotation angle increases with the
deployment fraction t (determined numerically, not assumed).

Layers:
  STATE_<pct>    link outlines + holes (6 links)
  SKELETON_<pct> pivot centerlines, kink-pivot circle (rho) and ring center mark
  ANNOTATIONS    parameters
"""

import math
import ezdxf
from ezdxf import units

from . import hoberman_ring as hr

# ----------------------------- parameters ------------------------------
CENTER_UNIT = 0                         # index of the middle scissor unit
STATE_FRACS = hr.STATE_FRACS            # keep settings from hoberman_ring
OUT_FILE    = "drawings/hoberman_3scissors.dxf"
# ------------------------------------------------------------------------


def unit_links(t, i):
    """Both links (phi, P, mirror, (P, Ja, Jb)) of unit i at deployment t."""
    rho, r_in, r_out = hr.state_radii(t)
    g0 = i * 2 * hr.ALPHA
    g1 = (i + 1) * 2 * hr.ALPHA
    gb = g0 + hr.ALPHA
    P = (rho * math.cos(gb), rho * math.sin(gb))
    Jo0 = (r_out * math.cos(g0), r_out * math.sin(g0))
    Ji0 = (r_in * math.cos(g0), r_in * math.sin(g0))
    Jo1 = (r_out * math.cos(g1), r_out * math.sin(g1))
    Ji1 = (r_in * math.cos(g1), r_in * math.sin(g1))
    out = []
    for (Ja, Jb) in [(Jo0, Ji1), (Ji0, Jo1)]:
        phi, mir = hr.link_pose(P, Ja, Jb)
        out.append((phi, P, mir, (P, Ja, Jb)))
    return out


def ccw_family_index():
    """0 or 1: which of the two links in a unit rotates CCW as t increases."""
    a = unit_links(0.40, CENTER_UNIT)
    b = unit_links(0.41, CENTER_UNIT)
    d0 = (b[0][0] - a[0][0] + math.pi) % (2 * math.pi) - math.pi
    d1 = (b[1][0] - a[1][0] + math.pi) % (2 * math.pi) - math.pi
    assert d0 * d1 < 0, "expected the two link families to counter-rotate"
    return 0 if d0 > 0 else 1


CCW = ccw_family_index()


def world_to_ref(phi_c, Pc):
    """Return f(pt) mapping world -> frame where link (phi_c, Pc) is at origin/+x."""
    c, s = math.cos(-phi_c), math.sin(-phi_c)

    def f(pt):
        x, y = pt[0] - Pc[0], pt[1] - Pc[1]
        return (c * x - s * y, s * x + c * y)
    return f


def main():
    doc = ezdxf.new("R2010", setup=True)
    doc.units = units.MM
    msp = doc.modelspace()
    doc.layers.add("ANNOTATIONS", color=7)

    colors = [3, 5, 1, 2, 4, 6]
    report = []
    for k, t in enumerate(STATE_FRACS):
        pct = int(round(t * 100))
        lay_s, lay_o = f"SKELETON_{pct}", f"STATE_{pct}"
        doc.layers.add(lay_s, color=8)
        doc.layers.add(lay_o, color=colors[k % len(colors)])
        rho, r_in, r_out = hr.state_radii(t)

        # reference link: CCW link of the center unit
        phi_c, Pc, mir_c, _ = unit_links(t, CENTER_UNIT)[CCW]
        f = world_to_ref(phi_c, Pc)

        # ring center + rho circle (through all kink pivots) in the new frame
        O = f((0.0, 0.0))
        msp.add_circle(O, rho, dxfattribs={"layer": lay_s, "linetype": "DASHED"})
        msp.add_lwpolyline([(O[0] - 5, O[1]), (O[0] + 5, O[1])], dxfattribs={"layer": lay_s})
        msp.add_lwpolyline([(O[0], O[1] - 5), (O[0], O[1] + 5)], dxfattribs={"layer": lay_s})

        for i in (CENTER_UNIT - 1, CENTER_UNIT, CENTER_UNIT + 1):
            for phi, P, mir, (P_, Ja, Jb) in unit_links(t, i % hr.N_UNITS):
                phi_new = phi - phi_c
                tx, ty = f(P)
                msp.add_lwpolyline([f(Ja), f(P), f(Jb)], dxfattribs={"layer": lay_s})
                poly = hr.transform(hr.LINK_POLY, phi_new, tx, ty, mir)
                msp.add_lwpolyline(poly, close=True, dxfattribs={"layer": lay_o})
                for hc in hr.transform(hr.LINK_HOLES, phi_new, tx, ty, mir):
                    msp.add_circle(hc, hr.HOLE_DIA / 2.0, dxfattribs={"layer": lay_o})

        report.append(f"state t={t:.2f}: rho={rho:.2f}  r_in={r_in:.2f}  r_out={r_out:.2f} mm"
                      f"  ring center at ({O[0]:.1f}, {O[1]:.1f})")

    lines = [
        f"HOBERMAN 3-SCISSOR DETAIL  n={hr.N_UNITS}, units {CENTER_UNIT-1}..{CENTER_UNIT+1}",
        f"aligned on CCW link of unit {CENTER_UNIT} (family {CCW}): kink at origin, bisector +x",
        f"arm A = {hr.ARM_LEN:.2f} mm, beta = {math.degrees(hr.BETA):.3f} deg, "
        f"width = {hr.LINK_WIDTH:.2f} mm, hole = {hr.HOLE_DIA:.2f} mm",
    ] + report
    y = -2 * hr.ARM_LEN - 30.0
    for ln in lines:
        msp.add_text(ln, height=4.0, dxfattribs={"layer": "ANNOTATIONS"}).set_placement((-2 * hr.ARM_LEN, y))
        y -= 7.0

    doc.saveas(OUT_FILE)
    print("\n".join(lines))
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
