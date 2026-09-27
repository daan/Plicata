"""
Parametric Hoberman ring generator with DXF export.

Generates:
  - LINK layer: the manufacturable angulated link (exact lines/arcs + pivot holes),
    drawn flat near the origin, ready to extrude in Fusion.
  - STATE_<pct> layers: full ring assembly (link outlines + holes) at each
    deployment state, for clearance checking.
  - SKELETON_<pct> layers: pivot-to-pivot centerlines + joint circles.
  - ANNOTATIONS layer: all driving parameters and computed radii as text.

Math
----
Ring of n identical scissor units. Rays spaced 2*alpha, alpha = pi/n.
Angulated link: two arms of length A joined at kink angle beta = pi - 2*alpha.
Kink pivot at radius rho on the bisector ray. Then the end joints sit at:
    r_out = rho*cos(alpha) + sqrt(A^2 - rho^2 sin^2(alpha))
    r_in  = rho*cos(alpha) - sqrt(A^2 - rho^2 sin^2(alpha))
Deployment range: rho in [A, A/sin(alpha)]  (r_in = 0  ...  fully opened).
Deployment fraction t maps linearly onto that range.
"""

import math
import ezdxf
from ezdxf import units

# ----------------------------- parameters ------------------------------
N_UNITS      = 20          # number of scissor units
ARM_LEN      = 53.8        # mm, kink pivot -> end pivot (per arm)
LINK_WIDTH   = 20.0         # mm, physical bar width
HOLE_DIA     = 8         # mm, pivot holes
STATE_FRACS  = [0.25, 0.5, 0.95]  # deployment fractions to export
OUT_FILE     = "drawings/hoberman_n20_53_8--.dxf"
# ------------------------------------------------------------------------

ALPHA = math.pi / N_UNITS
BETA = math.pi - 2 * ALPHA          # internal (kink) angle of the link
THETA = BETA / 2.0                  # half kink angle
H = LINK_WIDTH / 2.0

RHO_MIN = ARM_LEN
RHO_MAX = ARM_LEN / math.sin(ALPHA)


def state_radii(t):
    """Deployment fraction t in [0,1] -> (rho, r_in, r_out)."""
    rho = RHO_MIN + t * (RHO_MAX - RHO_MIN)
    root = math.sqrt(max(ARM_LEN**2 - (rho * math.sin(ALPHA))**2, 0.0))
    r_out = rho * math.cos(ALPHA) + root
    r_in = rho * math.cos(ALPHA) - root
    return rho, r_in, r_out


# ------------------- link outline in local coordinates -------------------
# Kink pivot at origin, bisector along +x, arms at +/- THETA.
U1 = (math.cos(THETA), math.sin(THETA))
U2 = (math.cos(THETA), -math.sin(THETA))
E1 = (ARM_LEN * U1[0], ARM_LEN * U1[1])
E2 = (ARM_LEN * U2[0], ARM_LEN * U2[1])


def _arc_pts(cx, cy, r, a0, a1, seg=24):
    """CCW arc sample points from angle a0 to a1 (radians)."""
    while a1 < a0:
        a1 += 2 * math.pi
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / seg),
             cy + r * math.sin(a0 + (a1 - a0) * i / seg))
            for i in range(seg + 1)]


def link_outline_polygon():
    """Closed CCW point list of the link outline (Minkowski of the V with a disc,
    concave side left as a sharp intersection point)."""
    n1 = (math.sin(THETA), -math.cos(THETA))    # inner normal, arm 1
    m1 = (-math.sin(THETA), math.cos(THETA))    # outer normal, arm 1
    n2 = (math.sin(THETA), math.cos(THETA))     # inner normal, arm 2
    m2 = (-math.sin(THETA), -math.cos(THETA))   # outer normal, arm 2
    v = (H / math.sin(THETA), 0.0)              # concave-side intersection

    pts = [v]
    # inner edge arm1 -> end cap 1 (semi-circle around E1: n1 dir -> m1 dir, through u1)
    pts.append((E1[0] + H * n1[0], E1[1] + H * n1[1]))
    a0 = math.atan2(n1[1], n1[0])
    a1 = math.atan2(m1[1], m1[0])
    pts += _arc_pts(E1[0], E1[1], H, a0, a1)[1:]
    # outer edge arm1 -> kink convex arc (m1 dir -> m2 dir, through -x)
    pts.append((H * m1[0], H * m1[1]))
    a0 = math.atan2(m1[1], m1[0])
    a1 = math.atan2(m2[1], m2[0])
    pts += _arc_pts(0, 0, H, a0, a1)[1:]
    # outer edge arm2 -> end cap 2 (m2 dir -> n2 dir, through u2)
    pts.append((E2[0] + H * m2[0], E2[1] + H * m2[1]))
    a0 = math.atan2(m2[1], m2[0])
    a1 = math.atan2(n2[1], n2[0])
    pts += _arc_pts(E2[0], E2[1], H, a0, a1)[1:]
    return pts


LINK_POLY = link_outline_polygon()
LINK_HOLES = [(0.0, 0.0), E1, E2]


def draw_link_exact(msp, layer, offset=(0.0, 0.0)):
    """Manufacturing version with true LINE/ARC entities."""
    ox, oy = offset
    n1 = (math.sin(THETA), -math.cos(THETA)); m1 = (-math.sin(THETA), math.cos(THETA))
    n2 = (math.sin(THETA), math.cos(THETA)); m2 = (-math.sin(THETA), -math.cos(THETA))
    v = (ox + H / math.sin(THETA), oy)
    e1 = (ox + E1[0], oy + E1[1]); e2 = (ox + E2[0], oy + E2[1])
    k = (ox, oy)
    L = dict(dxfattribs={"layer": layer})

    def deg(vec): return math.degrees(math.atan2(vec[1], vec[0]))

    msp.add_line(v, (e1[0] + H * n1[0], e1[1] + H * n1[1]), **L)
    msp.add_arc(e1, H, deg(n1), deg(m1), **L)
    msp.add_line((e1[0] + H * m1[0], e1[1] + H * m1[1]),
                 (k[0] + H * m1[0], k[1] + H * m1[1]), **L)
    msp.add_arc(k, H, deg(m1), deg(m2), **L)
    msp.add_line((k[0] + H * m2[0], k[1] + H * m2[1]),
                 (e2[0] + H * m2[0], e2[1] + H * m2[1]), **L)
    msp.add_arc(e2, H, deg(m2), deg(n2), **L)
    msp.add_line((e2[0] + H * n2[0], e2[1] + H * n2[1]), v, **L)
    for c in [k, e1, e2]:
        msp.add_circle(c, HOLE_DIA / 2.0, **L)


# ----------------------- ring assembly placement -------------------------
def transform(pts, phi, tx, ty, mirror=False):
    out = []
    c, s = math.cos(phi), math.sin(phi)
    for x, y in pts:
        if mirror:
            y = -y
        out.append((c * x - s * y + tx, s * x + c * y + ty))
    return out


def link_pose(P, J1, J2):
    """Rotation phi (+ mirror flag) mapping local arms (+/-THETA) onto world arms P->J1, P->J2."""
    a1 = math.atan2(J1[1] - P[1], J1[0] - P[0])
    a2 = math.atan2(J2[1] - P[1], J2[0] - P[0])
    d = (a1 - a2) % (2 * math.pi)
    if abs(d - BETA) < 1e-6:
        return (a1 - THETA), False
    return (a1 + THETA), True


def ring_links(t):
    """Yield (phi, tx, ty, mirror) for all 2*N links at deployment t."""
    rho, r_in, r_out = state_radii(t)
    for i in range(N_UNITS):
        g0 = i * 2 * ALPHA
        g1 = (i + 1) * 2 * ALPHA
        gb = g0 + ALPHA
        P = (rho * math.cos(gb), rho * math.sin(gb))
        Jo0 = (r_out * math.cos(g0), r_out * math.sin(g0))
        Ji0 = (r_in * math.cos(g0), r_in * math.sin(g0))
        Jo1 = (r_out * math.cos(g1), r_out * math.sin(g1))
        Ji1 = (r_in * math.cos(g1), r_in * math.sin(g1))
        for (Ja, Jb) in [(Jo0, Ji1), (Ji0, Jo1)]:
            phi, mir = link_pose(P, Ja, Jb)
            yield phi, P[0], P[1], mir, (P, Ja, Jb)


# ------------------------------ DXF export -------------------------------
def main():
    doc = ezdxf.new("R2010", setup=True)
    doc.units = units.MM
    msp = doc.modelspace()

    doc.layers.add("LINK", color=1)
    doc.layers.add("ANNOTATIONS", color=7)

    # sanity check: kink angle in assembly equals BETA
    for _, _, _, _, (P, Ja, Jb) in ring_links(0.5):
        v1 = (Ja[0] - P[0], Ja[1] - P[1]); v2 = (Jb[0] - P[0], Jb[1] - P[1])
        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (ARM_LEN ** 2)
        assert abs(math.acos(max(-1, min(1, dot))) - BETA) < 1e-9
        break

    # 1) manufacturable link, placed to the right of the ring
    _, _, r_out_max = state_radii(max(STATE_FRACS))
    link_ox = r_out_max + 60.0
    draw_link_exact(msp, "LINK", offset=(link_ox, 0.0))

    # 2) states
    for t in STATE_FRACS:
        pct = int(round(t * 100))
        lay_s = f"SKELETON_{pct}"
        lay_o = f"STATE_{pct}"
        doc.layers.add(lay_s, color=8)
        doc.layers.add(lay_o, color=3 if t == min(STATE_FRACS) else 5)
        rho, r_in, r_out = state_radii(t)

        msp.add_circle((0, 0), r_in, dxfattribs={"layer": lay_s, "linetype": "DASHED"})
        msp.add_circle((0, 0), r_out, dxfattribs={"layer": lay_s, "linetype": "DASHED"})

        for phi, tx, ty, mir, (P, Ja, Jb) in ring_links(t):
            msp.add_lwpolyline([Ja, P, Jb], dxfattribs={"layer": lay_s})
            poly = transform(LINK_POLY, phi, tx, ty, mir)
            msp.add_lwpolyline(poly, close=True, dxfattribs={"layer": lay_o})
            for hc in transform(LINK_HOLES, phi, tx, ty, mir):
                msp.add_circle(hc, HOLE_DIA / 2.0, dxfattribs={"layer": lay_o})

    # 3) annotations
    lines = [
        f"HOBERMAN RING  n={N_UNITS} units ({2*N_UNITS} links)",
        f"arm length (pivot-pivot) A = {ARM_LEN:.2f} mm",
        f"internal kink angle beta = {math.degrees(BETA):.3f} deg",
        f"link width = {LINK_WIDTH:.2f} mm, hole dia = {HOLE_DIA:.2f} mm",
        f"theoretical rho range: {RHO_MIN:.2f} .. {RHO_MAX:.2f} mm",
    ]
    for t in STATE_FRACS:
        rho, r_in, r_out = state_radii(t)
        lines.append(f"state t={t:.2f}: rho={rho:.2f}  r_in={r_in:.2f}  r_out={r_out:.2f} mm")
    r1 = state_radii(STATE_FRACS[0])[2]
    r2 = state_radii(STATE_FRACS[1])[2]
    lines.append(f"expansion ratio (outer, between states) = {r2/r1:.3f}")

    y = -r_out_max - 30.0
    for ln in lines:
        msp.add_text(ln, height=6.0, dxfattribs={"layer": "ANNOTATIONS"}).set_placement((-r_out_max, y))
        y -= 10.0

    doc.saveas(OUT_FILE)
    print("\n".join(lines))
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
