"""
hoberman.py  --  geometry core for a cable-actuated Hoberman ring.

No notebook or plotting dependency. numpy + shapely (+ ezdxf for export).

Conventions
-----------
* Ring centre O at the origin. n scissor units, unit k spans the angular
  sector [g_k, g_{k+1}] with g_k = NUM_DIR * 2*alpha*k and alpha = pi/n.
  NUM_DIR = -1, so unit numbers increase CLOCKWISE on screen (x right, y up).
  Mirroring the ring this way is free: in a link's own frame every cable
  route, length and channel comes out identical to the other handedness.
* Per unit k there are three pivot types:
      Jo[k]  outer joint on the ray g_k            (radius r_out)
      Ji[k]  inner joint on the ray g_k            (radius r_in)
      P[k]   kink / centre joint on the bisector   (radius rho)
* Two angulated links per unit, equal arms A, apex angle pi - 2*alpha:
      family 0 :  Jo[k] -> P[k] -> Ji[k+1]
      family 1 :  Ji[k] -> P[k] -> Jo[k+1]

Link types (what actually gets manufactured)
--------------------------------------------
Three bodies, distinguished by the channel milled into them:
      type 1 "plain"     family 0, no cable            -- yellow
      type 2 "expand"    family 1, expansion cable     -- green
      type 3 "contract"  family 1, contraction cable   -- blue
The link a cable is anchored on is drawn red ("anchor"); family 1 links that
no cable reaches yet are grey ("idle"). `link_roles()` resolves all of this
from the cables themselves, so the colouring can never drift from the routing.
* Deployment t in [0, 1]: t=0 fully contracted (r_in -> 0),
  t=1 fully expanded (chord Jo[k]-Ji[k+1] tangential, r_in == r_out).
  Practical builds live somewhere inside; use t_min/t_max.
* Link local frame: kink at origin, apex bisector along +x, first arm end
  (Ja) at angle -kappa, second (Jb) at +kappa, kappa = pi/2 - alpha.
  `mirror` is True when the world arrangement is the reflection of that.

Cables
------
A cable is a list of Waypoints. Each waypoint is a pivot the cable passes
over on a bearing of radius `radius`, on a given `side` (+1 = bearing
centre on the LEFT of the travel direction, -1 = RIGHT, None = automatic:
the inside of the turn, which is what a taut cable does). The first and
last waypoint are anchors (fixation / motor) and are treated as points.
Between waypoints the cable is a straight tangent span; on a waypoint it
follows an arc. Span + arc lengths give the cable length for a given t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

Vec = Tuple[float, float]

# Unit numbers increase clockwise on screen. See the module docstring.
NUM_DIR = -1

# Link body fill per role, and the stroke colour of the cable that causes it.
LINK_COL = {
    "plain":    "#f4d03f",   # type 1, family 0, no cable
    "expand":   "#58d68d",   # type 2, family 1, expansion cable
    "contract": "#5dade2",   # type 3, family 1, contraction cable
    "anchor":   "#e74c3c",   # the link a cable is fixed to
    "both":     "#af7ac5",   # two cables claim this link -- routing clash
    "idle":     "#d5d8dc",   # family 1, no cable reaches it
}
CABLE_COL = {"expand": "#196f3d", "contract": "#1a5276"}


# ============================================================ parameters ====

@dataclass
class Params:
    n: int = 12                 # scissor units around the ring
    arm_len: float = 60.0       # A, pivot-to-pivot arm length [mm]
    link_width: float = 12.0    # link body width [mm]
    hole_dia: float = 4.0       # pivot hole in the link [mm]
    pin_dia: float = 8.0        # bearing / bushing diameter the cable runs over [mm]
    cable_dia: float = 1.0      # cable diameter [mm]
    clearance: float = 0.5      # extra radial clearance for the channel [mm]
    t_min: float = 0.10         # practical deployment limits
    t_max: float = 0.90

    @property
    def alpha(self) -> float:
        return math.pi / self.n

    @property
    def kappa(self) -> float:
        return math.pi / 2 - self.alpha

    @property
    def apex(self) -> float:
        """Apex (kink) angle of an angulated link, pi - 2*alpha."""
        return math.pi - 2 * self.alpha

    @property
    def chord(self) -> float:
        """Constant distance between the two ends of a link."""
        return 2 * self.arm_len * math.cos(self.alpha)

    @property
    def psi_max(self) -> float:
        return math.pi / 2 - self.alpha


# ============================================================ kinematics ====

def state_radii(p: Params, t: float) -> Tuple[float, float, float]:
    """(rho, r_in, r_out) for deployment t in [0, 1]."""
    a, L = p.alpha, p.chord
    psi = (1.0 - t) * p.psi_max            # angle of the link chord from tangential
    r_out = 0.5 * (L * math.sin(psi) / math.cos(a) + L * math.cos(psi) / math.sin(a))
    r_in = 0.5 * (L * math.cos(psi) / math.sin(a) - L * math.sin(psi) / math.cos(a))
    rho = (r_out + r_in) / (2 * math.cos(a))
    return rho, r_in, r_out


def state_steps(p: Params, step: float = 0.1) -> Tuple[float, ...]:
    """Deployment values from t_min to t_max every `step`, both endpoints included.

    t_max is appended when the step does not divide the range evenly, so the fully
    deployed state is never missed off the end.
    """
    m = max(1, int(round((p.t_max - p.t_min) / step)))
    ts = [p.t_min + i * step for i in range(m + 1) if p.t_min + i * step <= p.t_max + 1e-9]
    if abs(ts[-1] - p.t_max) > 1e-9:
        ts.append(p.t_max)
    return tuple(round(t, 6) for t in ts)


def pivots(p: Params, t: float):
    """Arrays Jo, Ji, P of shape (n, 2)."""
    rho, r_in, r_out = state_radii(p, t)
    k = np.arange(p.n)
    g = NUM_DIR * 2 * p.alpha * k
    gb = g + NUM_DIR * p.alpha
    Jo = np.stack([r_out * np.cos(g), r_out * np.sin(g)], axis=1)
    Ji = np.stack([r_in * np.cos(g), r_in * np.sin(g)], axis=1)
    P = np.stack([rho * np.cos(gb), rho * np.sin(gb)], axis=1)
    return Jo, Ji, P


def link_ends(p: Params, t: float, family: int, k: int):
    """(Ja, P, Jb) world coordinates of link (family, k)."""
    Jo, Ji, P = pivots(p, t)
    k2 = (k + 1) % p.n
    if family == 0:
        return Jo[k], P[k], Ji[k2]
    return Ji[k], P[k], Jo[k2]


def link_pose(Ja, P, Jb) -> Tuple[float, bool]:
    """(phi, mirror): rotation of the local +x (apex bisector) and handedness."""
    ua = (np.asarray(Ja) - P)
    ub = (np.asarray(Jb) - P)
    ua = ua / np.linalg.norm(ua)
    ub = ub / np.linalg.norm(ub)
    b = ua + ub
    phi = math.atan2(b[1], b[0])
    cross = ua[0] * ub[1] - ua[1] * ub[0]
    mirror = cross < 0          # local convention: Ja at -kappa, Jb at +kappa -> cross > 0
    return phi, mirror


def world_to_local(pts, phi: float, P, mirror: bool):
    """Map world points into the link frame (kink at origin, bisector +x)."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float)) - np.asarray(P)
    c, s = math.cos(-phi), math.sin(-phi)
    out = np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], axis=1)
    if mirror:
        out[:, 1] *= -1
    return out


def local_to_world(pts, phi: float, P, mirror: bool):
    pts = np.atleast_2d(np.asarray(pts, dtype=float)).copy()
    if mirror:
        pts[:, 1] *= -1
    c, s = math.cos(phi), math.sin(phi)
    out = np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], axis=1)
    return out + np.asarray(P)


def link_centerline_local(p: Params):
    """Ja, P, Jb in the local frame."""
    k = p.kappa
    A = p.arm_len
    return np.array([[A * math.cos(-k), A * math.sin(-k)], [0.0, 0.0], [A * math.cos(k), A * math.sin(k)]])


def link_polygon_local(p: Params) -> Polygon:
    """Link body in the local frame (rounded ends, width link_width)."""
    cl = link_centerline_local(p)
    return LineString(cl).buffer(p.link_width / 2.0, cap_style="round", join_style="round", resolution=24)


def link_holes_local(p: Params):
    return link_centerline_local(p)


# ------------------------------------------------------------ part frame ----
# The link frame has the apex bisector along +x, which is convenient for maths
# but awkward to draw a part from. The *part* frame turns it so the arm that
# runs to the ring centre points straight down, kink still at the origin.

def part_rotation(p: Params, family: int) -> float:
    """Rotation taking the link frame to the part frame.

    In the link frame Ja sits at -kappa and Jb at +kappa, for every t. For family 1
    the inner joint (the end facing the ring centre) is Ja, so Ja swings from -kappa
    to -90 deg, a turn of -alpha. For family 0 the inner joint is Jb instead, which
    needs alpha - pi.
    """
    return -p.alpha if family == 1 else p.alpha - math.pi


def _rotate(pts, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    return np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], axis=1)


def world_to_part(p: Params, pts, phi: float, P, mirror: bool, family: int) -> np.ndarray:
    """World -> part frame: kink at the origin, inner arm straight down."""
    return _rotate(world_to_local(pts, phi, P, mirror), part_rotation(p, family))


def link_part_polygon(p: Params, family: int) -> np.ndarray:
    return _rotate(np.asarray(link_polygon_local(p).exterior.coords), part_rotation(p, family))


def link_part_holes(p: Params, family: int) -> np.ndarray:
    return _rotate(link_centerline_local(p), part_rotation(p, family))


# ================================================================ cables ====

@dataclass
class Waypoint:
    kind: str                       # 'Jo' | 'Ji' | 'P'
    k: int                          # unit index
    side: Optional[int] = None      # +1 left, -1 right, None auto (inside of turn)
    radius: Optional[float] = None  # override bearing radius; None -> pin_dia/2

    def label(self) -> str:
        return f"{self.kind}{self.k}"


@dataclass
class Cable:
    name: str
    waypoints: List[Waypoint]
    color: str = "#d33"
    role: str = ""              # 'expand' | 'contract' -- drives the link type
    fam: int = 1                # link family this cable is embedded in
    units: Tuple[int, ...] = () # link units it runs through, anchor first

    def centers(self, p: Params, t: float) -> np.ndarray:
        Jo, Ji, P = pivots(p, t)
        table = {"Jo": Jo, "Ji": Ji, "P": P}
        return np.array([table[w.kind][w.k % p.n] for w in self.waypoints], dtype=float)


@dataclass
class Span:
    a: np.ndarray
    b: np.ndarray

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))


@dataclass
class Arc:
    center: np.ndarray
    radius: float
    th0: float      # start angle
    th1: float      # end angle, reached by travelling `side` direction (+1 CCW, -1 CW)
    side: int
    wrap: float     # wrap angle [rad], >= 0

    @property
    def length(self) -> float:
        return self.radius * self.wrap

    def sample(self, step_deg: float = 4.0) -> np.ndarray:
        m = max(2, int(math.degrees(self.wrap) / step_deg) + 1)
        th = self.th0 + self.side * np.linspace(0.0, self.wrap, m)
        return self.center + self.radius * np.stack([np.cos(th), np.sin(th)], axis=1)


@dataclass
class Route:
    cable: Cable
    t: float
    elements: list = field(default_factory=list)   # alternating Span / Arc
    warnings: List[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return sum(e.length for e in self.elements)

    def polyline(self) -> np.ndarray:
        pts = []
        for e in self.elements:
            if isinstance(e, Span):
                pts.append(e.a[None, :])
                pts.append(e.b[None, :])
            else:
                pts.append(e.sample())
        return np.vstack(pts)


def _tangent(c1, r1, s1, c2, r2, s2):
    """Tangent points (p1, p2) of the belt line from circle 1 to circle 2.

    s = +1: circle centre on the left of travel direction, -1 on the right.
    Returns None if the circles are too close for the requested tangent.
    """
    d = c2 - c1
    D = float(np.linalg.norm(d))
    h = s2 * r2 - s1 * r1
    if D < 1e-9 or abs(h) > D:
        return None
    thd = math.atan2(d[1], d[0])
    th = thd - math.asin(h / D)
    u = np.array([math.cos(th), math.sin(th)])
    nrm = np.array([-u[1], u[0]])
    p1 = c1 - s1 * r1 * nrm
    p2 = c2 - s2 * r2 * nrm
    return p1, p2


def route_cable(p: Params, cable: Cable, t: float) -> Route:
    """Resolve spans and arcs of a cable at deployment t."""
    C = cable.centers(p, t)
    m = len(C)
    R = np.array([p.pin_dia / 2.0 if w.radius is None else w.radius for w in cable.waypoints])
    R[0] = 0.0
    R[-1] = 0.0                      # anchors are points
    S = np.zeros(m, dtype=int)
    for i in range(1, m - 1):
        w = cable.waypoints[i]
        if w.side is None:
            d1 = C[i] - C[i - 1]
            d2 = C[i + 1] - C[i]
            cr = d1[0] * d2[1] - d1[1] * d2[0]
            S[i] = 1 if cr >= 0 else -1
        else:
            S[i] = w.side
    route = Route(cable, t)
    tangents = []
    for i in range(m - 1):
        tp = _tangent(C[i], R[i], S[i], C[i + 1], R[i + 1], S[i + 1])
        if tp is None:
            route.warnings.append(f"no tangent between {cable.waypoints[i].label()} and {cable.waypoints[i+1].label()}")
            tp = (C[i], C[i + 1])
        tangents.append(tp)
    for i in range(m - 1):
        if i > 0:
            p_in = tangents[i - 1][1]
            p_out = tangents[i][0]
            th0 = math.atan2(*(p_in - C[i])[::-1])
            th1 = math.atan2(*(p_out - C[i])[::-1])
            wrap = (S[i] * (th1 - th0)) % (2 * math.pi)
            if wrap > math.pi:
                route.warnings.append(
                    f"{cable.waypoints[i].label()}: wrap {math.degrees(wrap):.0f} deg on side {S[i]:+d}, cable would lift off")
            route.elements.append(Arc(C[i].copy(), R[i], th0, th1, int(S[i]), wrap))
        route.elements.append(Span(tangents[i][0].copy(), tangents[i][1].copy()))
    return route


def length_curve(p: Params, cable: Cable, ts: Sequence[float]) -> np.ndarray:
    return np.array([route_cable(p, cable, t).length for t in ts])


def route_window(p: Params, cable: Cable, t: float, k: int, span: int = 1):
    """The stretch of cable from `span` links before unit k to `span` links after it.

    Returns a polyline, or None if the cable does not reach that far. The wrap arcs
    on the neighbouring pivots are included, so the drawing shows where the cable
    actually leaves the previous link and where it meets the next one.
    """
    route = route_cable(p, cable, t)
    wu = waypoint_units(p, cable)
    want = {(k + d) % p.n for d in range(-span, span + 1)}
    idx = [i for i, u in enumerate(wu) if u in want]
    if len(idx) < 2:
        return None
    i0, i1 = min(idx), max(idx)
    els = route.elements            # [Span0, Arc1, Span1, Arc2, Span2, ...]
    lo = max(0, 2 * i0 - 1)         # Arc(i0) when it exists, else the first span
    hi = min(len(els) - 1, 2 * i1 - 1)
    pts = [np.vstack([e.a, e.b]) if isinstance(e, Span) else e.sample() for e in els[lo:hi + 1]]
    return np.vstack(pts) if pts else None


# ------------------------------------------------------- standard routings --

def contraction_cable(k0: int, n_units: int, step: int = -1, side=None) -> Cable:
    """Fixed at P[k0], over n_units further centre joints, motor at P[k0 + step*n_units].

    Contraction runs *backwards* around the ring: step=-1 walks towards lower
    unit numbers, so a cable anchored high (say k0=17) ends at 17-n_units.
    Default side None = automatic (inside of the turn, i.e. towards the ring centre).
    """
    ks = [k0 + step * i for i in range(n_units + 1)]
    wps = [Waypoint("P", k, side=side) for k in ks]
    return Cable("contraction", wps, color=CABLE_COL["contract"],
                 role="contract", fam=1, units=tuple(ks))


def expansion_cable(k0: int, n_units: int, step: int = +1, sides=None) -> Cable:
    """Jo[k] -> Ji[k] -> P[k] -> Jo[k+1] -> Ji[k+1] -> P[k+1] ... over n_units units.

    Expansion runs *forwards* around the ring: step=+1 walks towards higher unit
    numbers, so a cable anchored at k0=0 ends at unit n_units-1.
    `sides` may be a dict {'Jo': s, 'Ji': s, 'P': s} to force wrap sides per pivot type.
    """
    sides = sides or {}
    ks = [k0 + step * i for i in range(n_units)]
    wps = []
    for k in ks:
        wps.append(Waypoint("Jo", k, side=sides.get("Jo")))
        wps.append(Waypoint("Ji", k, side=sides.get("Ji")))
        wps.append(Waypoint("P", k, side=sides.get("P")))
    return Cable("expansion", wps, color=CABLE_COL["expand"],
                 role="expand", fam=1, units=tuple(ks))


# ---------------------------------------------------------- link typing ----

def waypoint_units(p: Params, cable: Cable) -> List[int]:
    """The link unit each waypoint belongs to, one entry per waypoint.

    family 1 link k spans Ji[k] -> P[k] -> Jo[k+1]; family 0 link k spans
    Jo[k] -> P[k] -> Ji[k+1]. So the far end of the link is one unit back.
    """
    start = "Ji" if cable.fam == 1 else "Jo"
    return [(w.k if w.kind in ("P", start) else w.k - 1) % p.n for w in cable.waypoints]


def cable_link_units(p: Params, cable: Cable) -> List[int]:
    """Unit indices (mod n) of the family-`cable.fam` links the cable runs through."""
    if cable.units:
        return [k % p.n for k in cable.units]
    out: List[int] = []                     # fallback for hand-built cables
    for k in waypoint_units(p, cable):
        if k not in out:
            out.append(k)
    return out


def link_roles(p: Params, cables: Sequence[Cable]) -> dict:
    """{(family, k): role} for every link in the ring. Roles index LINK_COL."""
    roles = {(0, k): "plain" for k in range(p.n)}
    roles.update({(1, k): "idle" for k in range(p.n)})
    for cab in cables:
        if not cab.role:
            continue
        for k in cable_link_units(p, cab):
            key = (cab.fam, k)
            cur = roles.get(key, "idle")
            roles[key] = cab.role if cur in ("plain", "idle", cab.role) else "both"
    for cab in cables:                      # anchors last, they win over the run
        if cab.role:
            roles[(cab.fam, cable_link_units(p, cab)[0])] = "anchor"
    return roles


def cables_through(p: Params, cables: Sequence[Cable], family: int, k: int) -> List[Cable]:
    """The cables that are routed through link (family, k)."""
    return [c for c in cables if c.fam == family and (k % p.n) in cable_link_units(p, c)]


def representative_link(p: Params, cable: Cable) -> Tuple[int, int]:
    """(family, unit) of the most typical link for this cable -- the body to build.

    The middle of the run. Units at the ends are atypical: the cable terminates
    there as a point (anchor / motor), which moves the tangents and so reshapes the
    channel. How far that disturbance reaches depends on the weave -- for a cable
    with one waypoint per unit (contraction) it reaches the *second* unit too, since
    that unit's span starts at the point anchor. Only the middle of a run is
    therefore reliably representative; call link_variants() to see exactly which
    units share a channel before committing to a single body design.
    """
    ks = cable_link_units(p, cable)
    return cable.fam, ks[len(ks) // 2]


# ============================================================== envelopes ====

def cable_in_link_frame(p: Params, cable: Cable, family: int, k: int, ts: Sequence[float]):
    """List of (t, polyline in link local frame) for the whole cable."""
    out = []
    for t in ts:
        Ja, P, Jb = link_ends(p, t, family, k)
        phi, mir = link_pose(Ja, P, Jb)
        pl = route_cable(p, cable, t).polyline()
        out.append((t, world_to_local(pl, phi, P, mir)))
    return out


def channel_envelope(p: Params, cable: Cable, family: int, k: int, ts: Sequence[float],
                     extra: float = 0.0, only_inside_link: bool = True):
    """Swept region of the cable (plus clearance) in the link frame, as a shapely geometry.

    only_inside_link: intersect with a generous neighbourhood of the link body so that
    far-away parts of the cable do not clutter the result.
    """
    r = p.cable_dia / 2.0 + p.clearance + extra
    polys = [LineString(pl).buffer(r, resolution=8) for _, pl in cable_in_link_frame(p, cable, family, k, ts)]
    env = unary_union(polys)
    if only_inside_link:
        env = env.intersection(link_polygon_local(p).buffer(p.link_width))
    return env


def link_variants(p: Params, cable: Cable, ts: Sequence[float], tol: float = 1e-3):
    """Group a cable's links by channel shape: [[units sharing one body], ...].

    Two links share a body when the symmetric difference of their channel envelopes
    is a negligible fraction (< tol) of their area. Ends of a run form their own
    groups, so a run generally needs more than one body design -- unless you widen
    every link's channel to the union of the groups and cut one part for all.
    """
    ks = cable_link_units(p, cable)
    envs = {k: channel_envelope(p, cable, cable.fam, k, ts) for k in ks}
    groups: List[List[int]] = []
    for k in ks:
        for g in groups:
            ref = envs[g[0]]
            denom = max(ref.area, envs[k].area, 1e-12)
            if ref.symmetric_difference(envs[k]).area / denom < tol:
                g.append(k)
                break
        else:
            groups.append([k])
    return groups


# ================================================================ export ====

def export_ring_dxf(path: str, p: Params, cables: Sequence[Cable], ts_states: Sequence[float]):
    """DXF of the whole ring in several deployment states, with the cable paths.

    One drawing per t in ts_states, all sharing the origin so the states overlay.
    For the part drawing of a single link see export_link_dxf.

    Layers:  RING_<pct>, CABLE_<name>_<pct>, ANNOTATIONS
    """
    import ezdxf
    from ezdxf import units

    doc = ezdxf.new("R2010", setup=True)
    doc.units = units.MM
    msp = doc.modelspace()

    aci = {"contract": 5, "expand": 3}      # ezdxf ACI: 3 green, 5 blue

    def layer(name, **kw):
        """Add once: two exported links can share a family, and a cable can repeat."""
        if name not in doc.layers:
            doc.layers.add(name, **kw)
        return name

    link_poly = link_polygon_local(p)
    for ti, t in enumerate(ts_states):
        pct = int(round(t * 100))
        layer(f"RING_{pct}", color=(ti % 6) + 1)
        for fam in (0, 1):
            for k in range(p.n):
                Ja, P, Jb = link_ends(p, t, fam, k)
                phi, mir = link_pose(Ja, P, Jb)
                xy = local_to_world(np.asarray(link_poly.exterior.coords), phi, P, mir)
                msp.add_lwpolyline(xy.tolist(), close=True, dxfattribs={"layer": f"RING_{pct}"})
        for c in cables:
            lay = layer(f"CABLE_{c.name.upper()}_{pct}", color=aci.get(c.role, 1))
            msp.add_lwpolyline(route_cable(p, c, t).polyline().tolist(), dxfattribs={"layer": lay})

    # annotations sit below the ring
    y0 = -2.2 * state_radii(p, max(ts_states))[2]

    # parameter block, so the file states the geometry it was built from
    layer("ANNOTATIONS", color=7)
    lines = [
        f"HOBERMAN RING  n={p.n} units ({2 * p.n} links)",
        f"arm length (pivot-pivot) A = {p.arm_len:.3f} mm",
        f"internal kink angle beta = {math.degrees(p.apex):.3f} deg",
        f"half-sector alpha = {math.degrees(p.alpha):.3f} deg, "
        f"chord Ja-Jb = {p.chord:.3f} mm",
        f"link width = {p.link_width:.2f} mm, hole dia = {p.hole_dia:.2f} mm, "
        f"pin dia = {p.pin_dia:.2f} mm",
        f"cable dia = {p.cable_dia:.2f} mm, clearance = {p.clearance:.2f} mm",
    ]
    for t in ts_states:
        rho, r_in, r_out = state_radii(p, t)
        lines.append(f"state t={t:.2f}: rho={rho:.2f}  r_in={r_in:.2f}  r_out={r_out:.2f} mm")
    ay = y0 - 1.2 * p.arm_len
    for i, s in enumerate(lines):
        msp.add_text(s, height=4.0, dxfattribs={"layer": "ANNOTATIONS"}).set_placement(
            (-p.arm_len, ay - i * 6.0))

    doc.saveas(path)
    return path


def export_link_dxf(path: str, p: Params, cables: Sequence[Cable], ts=None,
                    step: float = 0.1, span: int = 1, links=None):
    """DXF of ONE link body at the origin with the cable paths that sweep over it.

    The link sits in the part frame: kink at the origin, the arm running to the ring
    centre pointing straight down. Both cabled types share this body, so one outline
    serves the green and the blue part -- only the cable paths differ.

    For each cable and each state in `ts` -- by default every `step` of deployment
    from t_min to t_max -- the cable is drawn from the previous link to the next one,
    as raw polylines with no boolean, each cable on its own layer so a channel can be
    traced off the whole sheaf of states.

    Layers:  LINK_OUTLINE, LINK_HOLES, ORIGIN, CABLE_<ROLE>, ANNOTATIONS
    """
    import ezdxf
    from ezdxf import units

    doc = ezdxf.new("R2010", setup=True)
    doc.units = units.MM
    msp = doc.modelspace()
    aci = {"contract": 5, "expand": 3}

    if ts is None:
        ts = state_steps(p, step)
    roled = [c for c in cables if c.role]
    if links is None:
        links = {c.name: representative_link(p, c) for c in roled}
    fam = roled[0].fam if roled else 1

    for name, col in (("LINK_OUTLINE", 7), ("LINK_HOLES", 8), ("ORIGIN", 2), ("ANNOTATIONS", 7)):
        doc.layers.add(name, color=col)

    msp.add_lwpolyline(link_part_polygon(p, fam).tolist(), close=True,
                       dxfattribs={"layer": "LINK_OUTLINE"})
    for h in link_part_holes(p, fam):
        msp.add_circle((h[0], h[1]), p.hole_dia / 2, dxfattribs={"layer": "LINK_HOLES"})
        msp.add_circle((h[0], h[1]), p.pin_dia / 2,
                       dxfattribs={"layer": "LINK_HOLES", "linetype": "DASHED"})
    cross = 0.6 * p.arm_len
    msp.add_line((-cross, 0), (cross, 0), dxfattribs={"layer": "ORIGIN"})
    msp.add_line((0, -cross), (0, cross), dxfattribs={"layer": "ORIGIN"})

    for c in roled:
        lay = f"CABLE_{c.role.upper()}"
        if lay not in doc.layers:
            doc.layers.add(lay, color=aci.get(c.role, 1))
        k = links[c.name][1]
        for t in ts:
            pl = route_window(p, c, t, k, span=span)
            if pl is None:
                continue
            Ja, P, Jb = link_ends(p, t, c.fam, k)
            phi, mir = link_pose(Ja, P, Jb)
            msp.add_lwpolyline(world_to_part(p, pl, phi, P, mir, c.fam).tolist(),
                               dxfattribs={"layer": lay})

    lines = [
        f"LINK PART  family {fam}   arm A = {p.arm_len:.2f} mm   width = {p.link_width:.2f} mm",
        "kink at the origin; the arm to the ring centre points straight down",
        f"hole dia {p.hole_dia:.2f}, pin dia {p.pin_dia:.2f} (dashed), cable dia {p.cable_dia:.2f} mm",
        f"cable states: {len(ts)}, t = {ts[0]:.2f} .. {ts[-1]:.2f}",
        f"each cable drawn from {span} link before to {span} link after this one",
    ]
    for c in roled:
        lines.append(f"CABLE_{c.role.upper()}: {c.name}, traced on link unit {links[c.name][1]}")
    ay = -1.5 * p.arm_len
    for i, s in enumerate(lines):
        msp.add_text(s, height=3.0, dxfattribs={"layer": "ANNOTATIONS"}).set_placement(
            (-p.arm_len, ay - i * 5.0))

    doc.saveas(path)
    return path


# ========================================================= quick mpl view ====

def plot_state(p: Params, cables: Sequence[Cable], t: float, ax=None, show_pins=True):
    """Static matplotlib rendering (fallback / figure export)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly, Circle

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    link_poly = link_polygon_local(p)
    roles = link_roles(p, cables)
    for fam in (0, 1):
        for k in range(p.n):
            Ja, P, Jb = link_ends(p, t, fam, k)
            phi, mir = link_pose(Ja, P, Jb)
            xy = local_to_world(np.asarray(link_poly.exterior.coords), phi, P, mir)
            ax.add_patch(MplPoly(xy, closed=True, fc=LINK_COL[roles[(fam, k)]],
                                 ec="#444", lw=0.6, alpha=0.55, zorder=fam))
    Jo, Ji, P = pivots(p, t)
    if show_pins:
        for arr in (Jo, Ji, P):
            for c in arr:
                ax.add_patch(Circle(c, p.pin_dia / 2, fc="none", ec="#333", lw=0.6, zorder=5))
    for c in cables:
        r = route_cable(p, c, t)
        pl = r.polyline()
        ax.plot(pl[:, 0], pl[:, 1], color=c.color, lw=2.0, zorder=6, label=f"{c.name}: {r.length:.1f} mm")
        ax.plot(*pl[0], "o", color=c.color, ms=6, zorder=7)
        ax.plot(*pl[-1], "s", color=c.color, ms=6, zorder=7)
    ax.set_aspect("equal")
    R = state_radii(p, 1.0)[2] * 1.1
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"t = {t:.2f}")
    return ax
