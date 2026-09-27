"""
viewer.py -- ipycanvas + ipywidgets front end for hoberman.py

    from plicata.hoberman import Params, contraction_cable, expansion_cable
    from plicata.viewer import RingViewer
    v = RingViewer(Params(), [contraction_cable(0, 6), expansion_cable(6, 6)])
    v.show()

Three canvases: ring view (drag on it to change t), cable-length curves,
and the channel envelope of one selected link in its local frame.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from ipycanvas import MultiCanvas, Canvas, hold_canvas
import ipywidgets as W

from . import hoberman as H

FAM_COL = ["#f4d03f", "#58d68d"]


class RingViewer:
    def __init__(self, params: H.Params, cables, size=560, n_curve=60, n_env=25):
        self.p = params
        self.cables = list(cables)
        self.size = size
        self.t = 0.5 * (params.t_min + params.t_max)
        self.n_curve = n_curve
        self.n_env = n_env
        self.fam, self.unit = 0, 0

        # ---- canvases
        self.ring = MultiCanvas(3, width=size, height=size)      # 0 bg, 1 links, 2 cables
        self.curve = Canvas(width=size * 0.8, height=size * 0.55)
        self.env = Canvas(width=size * 0.8, height=size * 0.9)
        for c in (self.ring, self.curve, self.env):
            c.layout.border = "1px solid #ccc"

        # ---- widgets
        self.s_t = W.FloatSlider(self.t, min=params.t_min, max=params.t_max, step=0.005,
                                 description="t", readout_format=".3f", continuous_update=True,
                                 layout=W.Layout(width="95%"))
        self.s_n = W.IntSlider(params.n, min=6, max=36, description="n units")
        self.s_A = W.FloatSlider(params.arm_len, min=20, max=200, step=0.1, description="arm A",
                                     readout_format=".1f")
        self.s_w = W.FloatSlider(params.link_width, min=4, max=40, step=0.5, description="width")
        self.s_pin = W.FloatSlider(params.pin_dia, min=2, max=30, step=0.5, description="pin dia")
        self.s_cab = W.FloatSlider(params.cable_dia, min=0.2, max=4, step=0.1, description="cable")
        self.s_clr = W.FloatSlider(params.clearance, min=0, max=3, step=0.1, description="clear.")
        self.s_tmin = W.FloatSlider(params.t_min, min=0.0, max=0.5, step=0.01, description="t min")
        self.s_tmax = W.FloatSlider(params.t_max, min=0.5, max=1.0, step=0.01, description="t max")
        self.d_fam = W.Dropdown(options=[("family 0 (Jo-P-Ji)", 0), ("family 1 (Ji-P-Jo)", 1)],
                                value=0, description="link")
        self.s_unit = W.IntSlider(0, min=0, max=params.n - 1, description="unit")
        self.chk_pins = W.Checkbox(True, description="show pins")
        self.out = W.Output(layout=W.Layout(max_height="120px", overflow="auto"))

        self.s_t.observe(self._on_t, "value")
        for s in (self.s_n, self.s_A, self.s_w, self.s_pin, self.s_cab, self.s_clr, self.s_tmin, self.s_tmax):
            s.observe(self._on_param, "value")
        self.d_fam.observe(self._on_link, "value")
        self.s_unit.observe(self._on_link, "value")
        self.chk_pins.observe(lambda ch: self.draw_ring(), "value")

        self._drag = False
        self.ring.on_mouse_down(self._mouse_down)
        self.ring.on_mouse_up(self._mouse_up)
        self.ring.on_mouse_move(self._mouse_move)

        self._cache_curves()
        self.draw_all()

    # ------------------------------------------------------------ transforms
    def _scale(self):
        R = H.state_radii(self.p, 1.0)[2] * 1.08
        return self.size / (2 * R)

    def _w2c(self, xy):
        """world mm -> ring canvas px (y up)."""
        s = self._scale()
        xy = np.atleast_2d(xy)
        return np.stack([self.size / 2 + s * xy[:, 0], self.size / 2 - s * xy[:, 1]], axis=1)

    # -------------------------------------------------------------- events
    def _on_t(self, ch):
        self.t = ch["new"]
        self.draw_ring()
        self.draw_curves()
        self.draw_env()

    def _on_param(self, ch):
        self.p = replace(self.p, n=self.s_n.value, arm_len=self.s_A.value, link_width=self.s_w.value,
                         pin_dia=self.s_pin.value, cable_dia=self.s_cab.value, clearance=self.s_clr.value,
                         t_min=self.s_tmin.value, t_max=self.s_tmax.value)
        self.s_unit.max = self.p.n - 1
        self.s_t.min, self.s_t.max = self.p.t_min, self.p.t_max
        self.t = min(max(self.t, self.p.t_min), self.p.t_max)
        self.s_t.value = self.t
        self._cache_curves()
        self.draw_all()

    def _on_link(self, ch):
        self.fam, self.unit = self.d_fam.value, self.s_unit.value
        self.draw_ring()
        self.draw_env()

    def _mouse_down(self, x, y):
        self._drag = True
        self._mouse_move(x, y)

    def _mouse_up(self, x, y):
        self._drag = False

    def _mouse_move(self, x, y):
        if not self._drag:
            return
        # radius of the mouse position sets r_out -> t (monotone), crude but intuitive
        s = self._scale()
        r = math.hypot(x - self.size / 2, y - self.size / 2) / s
        ts = np.linspace(self.p.t_min, self.p.t_max, 200)
        ro = np.array([H.state_radii(self.p, t)[2] for t in ts])
        self.s_t.value = float(ts[np.argmin(np.abs(ro - r))])

    # --------------------------------------------------------------- data
    def _cache_curves(self):
        self.ts = np.linspace(self.p.t_min, self.p.t_max, self.n_curve)
        self.L = {c.name: H.length_curve(self.p, c, self.ts) for c in self.cables}

    # ------------------------------------------------------------ drawing
    def draw_all(self):
        self.draw_bg()
        self.draw_ring()
        self.draw_curves()
        self.draw_env()

    def draw_bg(self):
        c = self.ring[0]
        with hold_canvas(c):
            c.clear()
            c.fill_style = "#fafafa"
            c.fill_rect(0, 0, self.size, self.size)
            c.stroke_style = "#ddd"
            c.line_width = 1
            cx = cy = self.size / 2
            c.stroke_line(cx - 8, cy, cx + 8, cy)
            c.stroke_line(cx, cy - 8, cx, cy + 8)
            s = self._scale()
            for rmm in range(50, int(H.state_radii(self.p, 1.0)[2]) + 1, 50):
                c.stroke_circle(cx, cy, rmm * s)

    def draw_ring(self):
        p, t = self.p, self.t
        link_poly = np.asarray(H.link_polygon_local(p).exterior.coords)
        s = self._scale()
        c = self.ring[1]
        with hold_canvas(c):
            c.clear()
            c.line_width = 1
            for fam in (0, 1):
                for k in range(p.n):
                    Ja, P, Jb = H.link_ends(p, t, fam, k)
                    phi, mir = H.link_pose(Ja, P, Jb)
                    xy = self._w2c(H.local_to_world(link_poly, phi, P, mir))
                    sel = (fam == self.fam and k == self.unit)
                    c.fill_style = "#e74c3c" if sel else FAM_COL[fam]
                    c.global_alpha = 0.75 if sel else 0.55
                    c.fill_polygon(xy.tolist())
                    c.global_alpha = 1.0
                    c.stroke_style = "#444"
                    c.stroke_polygon(xy.tolist())
            if self.chk_pins.value:
                Jo, Ji, P = H.pivots(p, t)
                c.stroke_style = "#222"
                c.fill_style = "#fff"
                for arr in (Jo, Ji, P):
                    for cc in self._w2c(arr):
                        c.fill_circle(cc[0], cc[1], p.pin_dia / 2 * s)
                        c.stroke_circle(cc[0], cc[1], p.pin_dia / 2 * s)
        c2 = self.ring[2]
        with hold_canvas(c2):
            c2.clear()
            c2.line_width = max(1.5, p.cable_dia * s)
            c2.font = "13px sans-serif"
            y_txt = 18
            warns = []
            for cab in self.cables:
                r = H.route_cable(p, cab, t)
                pl = self._w2c(r.polyline())
                c2.stroke_style = cab.color
                c2.stroke_lines(pl.tolist())
                c2.fill_style = cab.color
                c2.fill_circle(pl[0, 0], pl[0, 1], 5)
                c2.fill_rect(pl[-1, 0] - 5, pl[-1, 1] - 5, 10, 10)
                c2.fill_text(f"{cab.name}: {r.length:.1f} mm", 10, y_txt)
                y_txt += 16
                warns += [f"{cab.name}: {w}" for w in r.warnings]
            rho, ri, ro = H.state_radii(p, t)
            c2.fill_style = "#333"
            c2.fill_text(f"t={t:.3f}  r_in={ri:.1f}  rho={rho:.1f}  r_out={ro:.1f} mm", 10, self.size - 10)
        self.out.clear_output()
        if warns:
            with self.out:
                for w in warns[:8]:
                    print(w)

    def draw_curves(self):
        c = self.curve
        Wd, Hd = c.width, c.height
        m = 40
        allL = np.concatenate(list(self.L.values()))
        lo, hi = allL.min(), allL.max()
        pad = 0.05 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad

        def xy(t, L):
            x = m + (t - self.p.t_min) / (self.p.t_max - self.p.t_min) * (Wd - 2 * m)
            y = Hd - m - (L - lo) / (hi - lo) * (Hd - 2 * m)
            return x, y

        with hold_canvas(c):
            c.clear()
            c.fill_style = "#fafafa"
            c.fill_rect(0, 0, Wd, Hd)
            c.stroke_style = "#999"
            c.line_width = 1
            c.stroke_rect(m, m, Wd - 2 * m, Hd - 2 * m)
            c.fill_style = "#333"
            c.font = "11px sans-serif"
            c.fill_text(f"{lo:.0f}", 4, Hd - m)
            c.fill_text(f"{hi:.0f}", 4, m + 4)
            c.fill_text("cable length [mm] vs t", m, m - 8)
            c.fill_text(f"{self.p.t_min:.2f}", m, Hd - m + 14)
            c.fill_text(f"{self.p.t_max:.2f}", Wd - m - 20, Hd - m + 14)
            for cab in self.cables:
                L = self.L[cab.name]
                pts = [xy(t, l) for t, l in zip(self.ts, L)]
                c.stroke_style = cab.color
                c.line_width = 2
                c.stroke_lines(pts)
                Lt = H.route_cable(self.p, cab, self.t).length
                x, y = xy(self.t, Lt)
                c.fill_style = cab.color
                c.fill_circle(x, y, 5)
                # transmission ratio dL/dt at current t
                dL = np.gradient(L, self.ts)
                i = int(np.argmin(np.abs(self.ts - self.t)))
                c.fill_text(f"{cab.name}  dL/dt = {dL[i]:.0f} mm", x + 8, y - 6)
            x0, _ = xy(self.t, lo)
            c.stroke_style = "#aaa"
            c.line_width = 1
            c.stroke_line(x0, m, x0, Hd - m)

    def draw_env(self):
        c = self.env
        p = self.p
        Wd, Hd = c.width, c.height
        poly = H.link_polygon_local(p)
        A = p.arm_len
        s = (Hd - 40) / (2 * A + p.link_width)
        cx, cy = Wd * 0.5, Hd * 0.5

        def l2c(xy):
            xy = np.atleast_2d(xy)
            # local frame rotated so the bisector (+x) points right and arms up/down
            return np.stack([cx + s * xy[:, 0], cy - s * xy[:, 1]], axis=1)

        ts_env = np.linspace(p.t_min, p.t_max, self.n_env)
        with hold_canvas(c):
            c.clear()
            c.fill_style = "#fafafa"
            c.fill_rect(0, 0, Wd, Hd)
            # link body
            c.fill_style = "#eeeeee"
            c.stroke_style = "#222"
            c.line_width = 1.2
            xy = l2c(np.asarray(poly.exterior.coords))
            c.fill_polygon(xy.tolist())
            c.stroke_polygon(xy.tolist())
            for h in H.link_holes_local(p):
                hc = l2c(h)[0]
                c.stroke_style = "#222"
                c.stroke_circle(hc[0], hc[1], p.hole_dia / 2 * s)
                c.set_line_dash([4, 3])
                c.stroke_circle(hc[0], hc[1], p.pin_dia / 2 * s)
                c.set_line_dash([])
            # channel envelopes
            for cab in self.cables:
                env = H.channel_envelope(p, cab, self.fam, self.unit, ts_env)
                if env.is_empty:
                    continue
                c.fill_style = cab.color
                c.stroke_style = cab.color
                c.global_alpha = 0.35
                for g in getattr(env, "geoms", [env]):
                    if not hasattr(g, "exterior"):
                        continue
                    c.fill_polygon(l2c(np.asarray(g.exterior.coords)).tolist())
                c.global_alpha = 1.0
                # current-t path on top
                Ja, P, Jb = H.link_ends(p, self.t, self.fam, self.unit)
                phi, mir = H.link_pose(Ja, P, Jb)
                pl = H.world_to_local(H.route_cable(p, cab, self.t).polyline(), phi, P, mir)
                c.line_width = max(1.5, p.cable_dia * s)
                c.stroke_lines(l2c(pl).tolist())
            c.fill_style = "#333"
            c.font = "12px sans-serif"
            c.fill_text(f"link family {self.fam}, unit {self.unit}  (kink at origin, bisector +x)", 10, 16)
            c.fill_text(f"envelope over t in [{p.t_min:.2f}, {p.t_max:.2f}], {self.n_env} states", 10, 32)

    # ------------------------------------------------------------ layout
    def show(self):
        left = W.VBox([self.ring, self.s_t, self.chk_pins, self.out])
        ctrls = W.VBox([self.s_n, self.s_A, self.s_w, self.s_pin, self.s_cab, self.s_clr,
                        self.s_tmin, self.s_tmax, W.HTML("<b>link detail</b>"), self.d_fam, self.s_unit])
        right = W.VBox([ctrls, self.curve])
        return W.VBox([W.HBox([left, right]), self.env])
