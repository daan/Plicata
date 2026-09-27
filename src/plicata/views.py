"""
views.py -- step-by-step notebook views sharing one Model.

    from plicata.hoberman import Params, contraction_cable, expansion_cable
    from plicata.views import Model, RingView, LinkView, EnvelopeView

    m = Model(Params(), [contraction_cable(0, 6), expansion_cable(6, 6)])
    RingView(m)            # cell 1: structure + parameter sliders
    LinkView(m)            # cell 2: one link held fixed, neighbours and cables, t slider
    EnvelopeView(m)        # cell 3: swept channel (shapely)

All views observe the same Model. A widget displayed in several cells is the
same object, so the t slider in cell 2 and cell 1 stay in sync automatically.
Exceptions raised inside callbacks are shown in the view's log box instead of
being swallowed by the widget machinery.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import replace

import numpy as np
import ipywidgets as W
from ipycanvas import Canvas, hold_canvas
from IPython.display import display

from . import hoberman as H

LINK_COL = H.LINK_COL        # role -> fill; see hoberman.link_roles
SEL_COL = "#111111"          # the selected link is outlined, not recoloured,
SEL_WIDTH = 3.5              # so its own type colour stays readable

ROLE_LABEL = {
    "plain":    "type 1  plain (family 0)",
    "expand":   "type 2  expand cable",
    "contract": "type 3  contract cable",
    "anchor":   "cable anchor",
    "both":     "two cables (clash)",
    "idle":     "family 1, no cable",
}


# ================================================================== model ==

class Model:
    """Parameters, cables, current t and the selected link. Observers get notified."""

    def __init__(self, params: H.Params, cables, fam=1, unit=0):
        self.p = params
        self.cables = list(cables)
        self.t = 0.5 * (params.t_min + params.t_max)
        self.fam, self.unit = fam, unit
        self._obs = []

        self.s_t = W.FloatSlider(self.t, min=params.t_min, max=params.t_max, step=0.005,
                                 description="t", readout_format=".3f", continuous_update=True,
                                 layout=W.Layout(width="600px"))
        self.s_t.observe(self._on_t, "value")

        self.param_sliders = {
            "n": W.IntSlider(params.n, min=6, max=36, description="n units"),
            "arm_len": W.FloatSlider(params.arm_len, min=20, max=200, step=0.1, description="arm A",
                                          readout_format=".1f"),
            "link_width": W.FloatSlider(params.link_width, min=4, max=40, step=0.5, description="width"),
            "hole_dia": W.FloatSlider(params.hole_dia, min=1, max=20, step=0.5, description="hole"),
            "pin_dia": W.FloatSlider(params.pin_dia, min=2, max=30, step=0.5, description="pin dia"),
            "cable_dia": W.FloatSlider(params.cable_dia, min=0.2, max=4, step=0.1, description="cable"),
            "clearance": W.FloatSlider(params.clearance, min=0, max=3, step=0.1, description="clear."),
            "t_min": W.FloatSlider(params.t_min, min=0.0, max=0.5, step=0.01, description="t min"),
            "t_max": W.FloatSlider(params.t_max, min=0.5, max=1.0, step=0.01, description="t max"),
        }
        for s in self.param_sliders.values():
            s.observe(self._on_param, "value")

        self.d_fam = W.Dropdown(options=[("family 0 - plain", 0), ("family 1 - cabled", 1)],
                                value=fam, description="link")
        self.s_unit = W.IntSlider(unit, min=0, max=params.n - 1, description="unit")
        self.d_fam.observe(self._on_sel, "value")
        self.s_unit.observe(self._on_sel, "value")

    # -- subscription
    def observe(self, fn):
        self._obs.append(fn)

    def notify(self, what):
        for fn in self._obs:
            fn(what)

    # -- callbacks
    def _on_t(self, ch):
        self.t = ch["new"]
        self.notify("t")

    def _on_param(self, ch):
        kw = {k: s.value for k, s in self.param_sliders.items()}
        self.p = replace(self.p, **kw)
        self.s_unit.max = self.p.n - 1
        self.s_t.max = max(self.p.t_max, self.s_t.min)
        self.s_t.min = self.p.t_min
        self.s_t.max = self.p.t_max
        self.t = min(max(self.t, self.p.t_min), self.p.t_max)
        if self.s_t.value != self.t:
            self.s_t.value = self.t       # triggers _on_t -> notify("t")
        self.notify("params")

    def _on_sel(self, ch):
        self.fam, self.unit = self.d_fam.value, self.s_unit.value
        self.notify("sel")

    # -- convenience
    def roles(self):
        """{(family, k): role} for the current parameters and cables."""
        return H.link_roles(self.p, self.cables)

    def role_of(self, fam=None, unit=None):
        fam = self.fam if fam is None else fam
        unit = self.unit if unit is None else unit
        return self.roles().get((fam, unit % self.p.n), "idle")

    def ref_pose(self, t=None):
        """(phi, P, mirror) of the selected link."""
        t = self.t if t is None else t
        Ja, P, Jb = H.link_ends(self.p, t, self.fam, self.unit)
        phi, mir = H.link_pose(Ja, P, Jb)
        return phi, P, mir


# =============================================================== helpers ==

class _View:
    """Base: a canvas, a log Output, and safe redraw."""

    def __init__(self, model: Model, w: int, h: int):
        self.m = model
        self.c = Canvas(width=int(w), height=int(h))
        self.c.layout.border = "1px solid #ccc"
        self.log = W.Output(layout=W.Layout(max_height="90px", overflow="auto"))
        model.observe(self._on_model)

    def _on_model(self, what):
        self.redraw()

    def redraw(self):
        try:
            with hold_canvas(self.c):
                self.c.clear()
                self.c.fill_style = "#fafafa"
                self.c.fill_rect(0, 0, self.c.width, self.c.height)
                self.draw()
        except Exception:
            self.log.clear_output()
            with self.log:
                traceback.print_exc()

    def draw(self):
        raise NotImplementedError

    def _ipython_display_(self):
        display(self.widget())


def _draw_link(c: Canvas, xy, fill, sel=False):
    """Fill is the link's type colour; selection is a heavy outline on top, so that
    the selected link still shows which of the three types it is."""
    c.fill_style = fill
    c.global_alpha = 0.75 if sel else 0.55
    c.fill_polygon(xy.tolist())
    c.global_alpha = 1.0
    c.stroke_style = SEL_COL if sel else "#444"
    c.line_width = SEL_WIDTH if sel else 1
    c.stroke_polygon(xy.tolist())


def _draw_pin(c: Canvas, x, y, r_pin, r_hole=None):
    c.fill_style = "#fff"
    c.stroke_style = "#222"
    c.line_width = 1
    c.fill_circle(x, y, r_pin)
    c.stroke_circle(x, y, r_pin)
    if r_hole:
        c.stroke_circle(x, y, r_hole)


# ============================================================== ring view ==

class RingView(_View):
    """Cell 1: whole structure plus all parameter sliders. Drag on the canvas to change t."""

    def __init__(self, model: Model, size=600):
        super().__init__(model, size, size)
        self.size = size
        self.chk_pins = W.Checkbox(True, description="pins")
        self.chk_cables = W.Checkbox(True, description="cables")
        for chk in (self.chk_pins, self.chk_cables):
            chk.observe(lambda ch: self.redraw(), "value")
        self._drag = False
        self.c.on_mouse_down(self._down)
        self.c.on_mouse_up(self._up)
        self.c.on_mouse_move(self._move)
        self.redraw()

    def _scale(self):
        return self.size / (2 * H.state_radii(self.m.p, 1.0)[2] * 1.08)

    def w2c(self, xy):
        s = self._scale()
        xy = np.atleast_2d(xy)
        return np.stack([self.size / 2 + s * xy[:, 0], self.size / 2 - s * xy[:, 1]], axis=1)

    def _down(self, x, y):
        self._drag = True
        self._move(x, y)

    def _up(self, x, y):
        self._drag = False

    def _move(self, x, y):
        if not self._drag:
            return
        r = math.hypot(x - self.size / 2, y - self.size / 2) / self._scale()
        ts = np.linspace(self.m.p.t_min, self.m.p.t_max, 300)
        ro = np.array([H.state_radii(self.m.p, t)[2] for t in ts])
        self.m.s_t.value = float(ts[np.argmin(np.abs(ro - r))])

    def draw(self):
        m, p, t, c = self.m, self.m.p, self.m.t, self.c
        s = self._scale()
        cx = cy = self.size / 2
        c.stroke_style = "#ddd"
        c.line_width = 1
        for rmm in range(50, int(H.state_radii(p, 1.0)[2]) + 1, 50):
            c.stroke_circle(cx, cy, rmm * s)
        c.stroke_line(cx - 8, cy, cx + 8, cy)
        c.stroke_line(cx, cy - 8, cx, cy + 8)

        poly = np.asarray(H.link_polygon_local(p).exterior.coords)
        roles = m.roles()
        # selected link last, so its outline is not overdrawn by a neighbour
        order = [(f, k) for f in (0, 1) for k in range(p.n) if not (f == m.fam and k == m.unit)]
        order.append((m.fam, m.unit))
        for fam, k in order:
            Ja, P, Jb = H.link_ends(p, t, fam, k)
            phi, mir = H.link_pose(Ja, P, Jb)
            xy = self.w2c(H.local_to_world(poly, phi, P, mir))
            _draw_link(c, xy, LINK_COL[roles[(fam, k)]], sel=(fam == m.fam and k == m.unit))
        if self.chk_pins.value:
            Jo, Ji, P = H.pivots(p, t)
            for arr in (Jo, Ji, P):
                for x, y in self.w2c(arr):
                    _draw_pin(c, x, y, p.pin_dia / 2 * s)
            # unit number at the kink, the pivot both links of unit k share.
            # Numbers run clockwise on screen (hoberman.NUM_DIR).
            c.fill_style = "#333"
            c.font = "11px sans-serif"
            for k, (x, y) in enumerate(self.w2c(P)):
                c.fill_text(str(k), x + 6, y - 6)
        ytxt = 18
        c.font = "13px sans-serif"
        if self.chk_cables.value:
            for cab in m.cables:
                r = H.route_cable(p, cab, t)
                pl = self.w2c(r.polyline())
                c.stroke_style = cab.color
                c.line_width = max(2.0, p.cable_dia * s)
                c.stroke_lines(pl.tolist())
                c.fill_style = cab.color
                c.fill_circle(pl[0, 0], pl[0, 1], 5)
                c.fill_rect(pl[-1, 0] - 5, pl[-1, 1] - 5, 10, 10)
                c.fill_text(f"{cab.name}: {r.length:.1f} mm", 10, ytxt)
                ytxt += 16
                if r.warnings:
                    self.log.clear_output()
                    with self.log:
                        print(cab.name, r.warnings[0])
        rho, ri, ro = H.state_radii(p, t)
        c.fill_style = "#333"
        c.fill_text(f"t={t:.3f}   r_in={ri:.1f}   rho={rho:.1f}   r_out={ro:.1f} mm", 10, self.size - 10)
        self._legend(roles)

    def _legend(self, roles):
        """Colour key, listing only the link types actually present."""
        c = self.c
        present = [r for r in ("plain", "expand", "contract", "anchor", "both", "idle")
                   if r in roles.values()]
        c.font = "11px sans-serif"
        x, y = self.size - 175, 18
        for r in present:
            c.fill_style = LINK_COL[r]
            c.global_alpha = 0.55
            c.fill_rect(x, y - 9, 12, 12)
            c.global_alpha = 1.0
            c.stroke_style = "#444"
            c.line_width = 1
            c.stroke_rect(x, y - 9, 12, 12)
            c.fill_style = "#333"
            c.fill_text(ROLE_LABEL[r], x + 18, y)
            y += 16

    def widget(self):
        ps = list(self.m.param_sliders.values())
        ctrls = W.VBox(ps + [W.HTML("<b>selected link</b>"), self.m.d_fam, self.m.s_unit,
                             W.HBox([self.chk_pins, self.chk_cables])])
        return W.VBox([W.HBox([self.c, ctrls]), self.m.s_t, self.log])


# ============================================================== link view ==

class LinkView(_View):
    """Cell 2: the selected link held fixed (kink at origin, bisector +x); its two
    neighbouring scissor units and the cables move around it. Option to draw the
    cable paths of all t as thin trails, which is the channel before any offsetting."""

    def __init__(self, model: Model, w=800, h=600, n_units_shown=1):
        super().__init__(model, w, h)
        self.n_units_shown = n_units_shown
        self.chk_trails = W.Checkbox(False, description="cable trails over t range")
        self.s_trails = W.IntSlider(25, min=5, max=80, description="trail states")
        self.chk_trails.observe(lambda ch: self.redraw(), "value")
        self.s_trails.observe(lambda ch: self.redraw(), "value")
        self.redraw()

    def _scale(self):
        return (self.c.height - 40) / (2 * self.m.p.arm_len + self.m.p.link_width)

    def l2c(self, xy):
        s = self._scale()
        xy = np.atleast_2d(xy)
        return np.stack([self.c.width * 0.45 + s * xy[:, 0], self.c.height / 2 - s * xy[:, 1]], axis=1)

    def draw(self):
        m, p, t, c = self.m, self.m.p, self.m.t, self.c
        s = self._scale()
        phi_r, P_r, mir_r = m.ref_pose()
        to_ref = lambda xy: H.world_to_local(xy, phi_r, P_r, mir_r)

        # ring centre and rho circle in this frame
        O = self.l2c(to_ref([[0.0, 0.0]]))[0]
        rho = H.state_radii(p, t)[0]
        c.stroke_style = "#ccc"
        c.set_line_dash([6, 4])
        c.stroke_circle(O[0], O[1], rho * s)
        c.set_line_dash([])
        c.stroke_line(O[0] - 10, O[1], O[0] + 10, O[1])
        c.stroke_line(O[0], O[1] - 10, O[0], O[1] + 10)

        poly = np.asarray(H.link_polygon_local(p).exterior.coords)
        roles = m.roles()
        ks = [(m.unit + d) % p.n for d in range(-self.n_units_shown, self.n_units_shown + 1)]
        # draw neighbours first, selected link last
        order = [(f, k) for k in ks for f in (0, 1) if not (f == m.fam and k == m.unit)]
        order.append((m.fam, m.unit))
        for fam, k in order:
            Ja, P, Jb = H.link_ends(p, t, fam, k)
            phi, mir = H.link_pose(Ja, P, Jb)
            xy = self.l2c(to_ref(H.local_to_world(poly, phi, P, mir)))
            _draw_link(c, xy, LINK_COL[roles[(fam, k)]], sel=(fam == m.fam and k == m.unit))
        # pins of the shown units
        Jo, Ji, Pk = H.pivots(p, t)
        ks_pins = ks + [(ks[-1] + 1) % p.n]
        for arr, lab in ((Jo, "Jo"), (Ji, "Ji")):
            for k in ks_pins:
                x, y = self.l2c(to_ref(arr[k]))[0]
                _draw_pin(c, x, y, p.pin_dia / 2 * s, p.hole_dia / 2 * s)
                c.fill_style = "#333"
                c.font = "10px sans-serif"
                c.fill_text(f"{lab}{k}", x + 6, y - 6)
        for k in ks:
            x, y = self.l2c(to_ref(Pk[k]))[0]
            _draw_pin(c, x, y, p.pin_dia / 2 * s, p.hole_dia / 2 * s)
            c.fill_style = "#333"
            c.fill_text(f"P{k}", x + 6, y - 6)

        # cable trails (all states) then the current path on top
        if self.chk_trails.value:
            ts = np.linspace(p.t_min, p.t_max, self.s_trails.value)
            c.line_width = 1
            c.global_alpha = 0.35
            for cab in m.cables:
                c.stroke_style = cab.color
                for tt in ts:
                    phi_t, P_t, mir_t = m.ref_pose(tt)
                    pl = H.world_to_local(H.route_cable(p, cab, tt).polyline(), phi_t, P_t, mir_t)
                    c.stroke_lines(self.l2c(pl).tolist())
            c.global_alpha = 1.0
        c.line_width = max(2.0, p.cable_dia * s)
        c.font = "13px sans-serif"
        ytxt = 18
        for cab in m.cables:
            r = H.route_cable(p, cab, t)
            pl = self.l2c(to_ref(r.polyline()))
            c.stroke_style = cab.color
            c.stroke_lines(pl.tolist())
            c.fill_style = cab.color
            c.fill_text(f"{cab.name}: {r.length:.1f} mm", 10, ytxt)
            ytxt += 16
        c.fill_style = "#333"
        c.fill_text(f"link family {m.fam}, unit {m.unit} fixed: kink at origin, bisector +x.  t={t:.3f}",
                    10, self.c.height - 10)

    def widget(self):
        return W.VBox([self.c, self.m.s_t, W.HBox([self.m.d_fam, self.m.s_unit, self.chk_trails, self.s_trails]),
                       self.log])


# ========================================================== envelope view ==

class EnvelopeView(_View):
    """Cell 3: swept channel region (shapely union of buffered cable paths) in the
    frame of the selected link, with the current path on top."""

    def __init__(self, model: Model, w=800, h=600, n_states=31):
        super().__init__(model, w, h)
        self.s_states = W.IntSlider(n_states, min=5, max=100, description="states")
        self.s_states.observe(lambda ch: self.redraw(), "value")
        self._cache = {}
        self.redraw()

    def _on_model(self, what):
        if what in ("params", "sel"):
            self._cache = {}
        self.redraw()

    def _scale(self):
        return (self.c.height - 40) / (2 * self.m.p.arm_len + self.m.p.link_width)

    def l2c(self, xy):
        s = self._scale()
        xy = np.atleast_2d(xy)
        return np.stack([self.c.width * 0.45 + s * xy[:, 0], self.c.height / 2 - s * xy[:, 1]], axis=1)

    def cables_here(self):
        """Only the cables actually routed through the selected link, i.e. the green
        and blue types. A plain (yellow) or idle link has no channel at all."""
        return H.cables_through(self.m.p, self.m.cables, self.m.fam, self.m.unit)

    def envelopes(self):
        key = (self.s_states.value,)
        if key not in self._cache:
            ts = np.linspace(self.m.p.t_min, self.m.p.t_max, self.s_states.value)
            self._cache[key] = {cab.name: H.channel_envelope(self.m.p, cab, self.m.fam, self.m.unit, ts)
                                for cab in self.cables_here()}
        return self._cache[key]

    def draw(self):
        m, p, c = self.m, self.m.p, self.c
        s = self._scale()
        poly = H.link_polygon_local(p)
        xy = self.l2c(np.asarray(poly.exterior.coords))
        c.fill_style = "#eeeeee"
        c.fill_polygon(xy.tolist())
        c.stroke_style = "#222"
        c.line_width = 1.2
        c.stroke_polygon(xy.tolist())
        for h in H.link_holes_local(p):
            x, y = self.l2c(h)[0]
            c.stroke_circle(x, y, p.hole_dia / 2 * s)
            c.set_line_dash([4, 3])
            c.stroke_circle(x, y, p.pin_dia / 2 * s)
            c.set_line_dash([])
        here = self.cables_here()
        envs = self.envelopes()
        for cab in here:
            env = envs[cab.name]
            if env.is_empty:
                continue
            c.fill_style = cab.color
            c.stroke_style = cab.color
            c.global_alpha = 0.35
            for g in getattr(env, "geoms", [env]):
                if not hasattr(g, "exterior"):
                    continue
                c.fill_polygon(self.l2c(np.asarray(g.exterior.coords)).tolist())
            c.global_alpha = 1.0
            c.line_width = 1
            for g in getattr(env, "geoms", [env]):
                if hasattr(g, "exterior"):
                    c.stroke_polygon(self.l2c(np.asarray(g.exterior.coords)).tolist())
            phi, P, mir = m.ref_pose()
            pl = H.world_to_local(H.route_cable(p, cab, m.t).polyline(), phi, P, mir)
            c.line_width = max(2.0, p.cable_dia * s)
            c.stroke_lines(self.l2c(pl).tolist())
        c.fill_style = "#333"
        c.font = "13px sans-serif"
        c.fill_text(f"link family {m.fam} unit {m.unit}  --  {ROLE_LABEL[m.role_of()]}", 10, 18)
        if here:
            c.fill_text(f"channel = cable r + clearance = {p.cable_dia/2 + p.clearance:.2f} mm "
                        f"({', '.join(cab.name for cab in here)})", 10, 36)
        else:
            c.fill_text("no cable runs through this link -- nothing to mill", 10, 36)

    def widget(self):
        return W.VBox([self.c, self.m.s_t, W.HBox([self.m.d_fam, self.m.s_unit, self.s_states]), self.log])
