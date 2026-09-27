"""
plicata.cli -- drive the cable-actuated Hoberman ring with two Dynamixels.

    plicata status              positions, loads, limits, ring state estimated from each servo
    plicata off                 torque off (turn the drums by hand)
    plicata jog con 90          wind the contraction cable 90 deg in (negative: pay out),
                                at low PWM, so it stalls once the cable is taut
    plicata set-home            the ring is at the hard stop now: zero both servos
    plicata expand              open to T_OPEN
    plicata contract            close to T_CLOSED (just off the hard stop)
    plicata goto 0.5            go to deployment t
    plicata goto --dia 450      go to an outer diameter [mm]

Homing, once per power-up (the servos forget their turn count at power-off):
  1. plicata off
  2. close the ring until the links under the cables touch (the rest of the
     ring may lag, outer diameter ~273-283 mm)
  3. take up the slack on both drums by hand, or with `plicata jog`
  4. plicata set-home

Angles in this file are "cable wound onto the drum" in degrees, 0 at home.
The contraction drum unwinds while the ring opens, so its angle goes negative.
WIND maps that onto the servo's own rotation direction; check it with `jog`
before the first `expand`.

Both servos follow the table in lock step (XL430, extended position mode).
The servo that pulls (expansion servo when opening, contraction servo when
closing) runs at full PWM; the other one gets FOLLOWER_PWM so a mismatch
between model and ring turns into a little slack or yield instead of the two
servos fighting. The XL430 has no current sensor: PWM caps the effort and
Present Load (% of max torque) is the only force readout.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import dynamixels as D
from . import hoberman as H

CON_ID = 3
EXP_ID = 2
WIND = {"con": -1, "exp": +1}   # +1: positive servo rotation winds cable in. Flip if `jog` pays out.
DRUM_R = 7.5                    # winding radius, both drums [mm]

T_HOME = 0.0661                 # links under the cables touching (model dia 272.7 mm; whole ring ~283, play)
T_CLOSED = 0.10
T_OPEN = 0.90

SPEED_DEG_S = 120.0             # speed of whichever servo turns fastest
RATE_HZ = 50
LEADER_PWM = D.PWM_MAX
FOLLOWER_PWM = 300              # ~34 %
LOAD_LIMIT = 80.0               # % ; stop when a servo stays above this
JOG_PWM = 250                   # low, so `jog` stalls on a taut cable
JOG_PROFILE_VELOCITY = 40       # x 0.229 rpm


def ring_params() -> H.Params:
    # same as hoberman_ring.ipynb
    return H.Params(n=20, arm_len=53.8, link_width=10.0, hole_dia=3.0, pin_dia=3.0,
                    cable_dia=0.5, clearance=0.5, t_min=T_CLOSED, t_max=T_OPEN)


class Ring:
    """Lookup between deployment t, outer diameter and wound drum angles."""

    def __init__(self, n: int = 400):
        p = ring_params()
        con = H.contraction_cable(k0=17, n_units=6)
        exp = H.expansion_cable(k0=0, n_units=6)
        self.t = np.linspace(T_HOME, T_OPEN, n)
        Lc = H.length_curve(p, con, self.t)
        Le = H.length_curve(p, exp, self.t)
        self.con = -np.degrees((Lc - Lc[0]) / DRUM_R)     # decreasing in t
        self.exp = np.degrees((Le[0] - Le) / DRUM_R)      # increasing in t
        self.dia = 2 * np.array([H.state_radii(p, tt)[2] for tt in self.t])
        self.dcon = np.gradient(self.con, self.t)
        self.dexp = np.gradient(self.exp, self.t)

    def angles(self, t):
        return float(np.interp(t, self.t, self.con)), float(np.interp(t, self.t, self.exp))

    def outer_dia(self, t):
        return float(np.interp(t, self.t, self.dia))

    def t_from_con(self, deg):
        return float(np.interp(-deg, -self.con, self.t))

    def t_from_exp(self, deg):
        return float(np.interp(deg, self.exp, self.t))

    def t_from_dia(self, dia):
        return float(np.interp(dia, self.dia, self.t))

    def t_from_both(self, con_deg, exp_deg):
        """Trust the servo whose angle is most sensitive to t at that point."""
        tc, te = self.t_from_con(con_deg), self.t_from_exp(exp_deg)
        wc = np.interp(tc, self.t, self.dcon) ** 2
        we = np.interp(te, self.t, self.dexp) ** 2
        return (wc * tc + we * te) / (wc + we), tc, te

    def max_rate(self, t):
        """deg per unit t of the faster servo."""
        return max(abs(np.interp(t, self.t, self.dcon)), abs(np.interp(t, self.t, self.dexp)))


class Rig:
    def __init__(self, port=None):
        self.bus = D.DynamixelBus(port)
        self.servo = {"con": D.DynamixelServo(self.bus, CON_ID, "contraction"),
                      "exp": D.DynamixelServo(self.bus, EXP_ID, "expansion")}
        for s in self.servo.values():
            if self.bus.ping(s.id) is None:
                raise SystemExit(f"{s.name} servo (id {s.id}) does not answer")
            err = s.hardware_error()
            if err:
                raise SystemExit(f"{s.name} servo has hardware error 0x{err:02x} "
                                 "(overload/overheat/voltage); power-cycle or reboot it")

    def close(self):
        self.bus.close_port()

    # cable angle (wound, deg) <-> servo ticks
    def wound(self, key):
        s = self.servo[key]
        return WIND[key] * s.to_angle(s.present_position())

    def ticks(self, key, wound_deg):
        return self.servo[key].to_position(WIND[key] * wound_deg)

    def ensure_mode(self):
        for s in self.servo.values():
            if s.operating_mode() != D.MODE_EXTENDED_POSITION:
                s.disable_torque()
                s.set_operating_mode(D.MODE_EXTENDED_POSITION)
                print(f"{s.name}: switched to extended position mode -- run set-home again")

    def torque_on_holding(self):
        """Enable torque without a jump: goal = where it is now."""
        for s in self.servo.values():
            if not s.torque_enabled():
                s.set_goal_position(s.present_position())
                s.enable_torque()

    def hold(self):
        for s in self.servo.values():
            s.set_goal_position(s.present_position())


def cmd_status(rig: Rig, ring: Ring, args):
    for key, s in rig.servo.items():
        mode = {3: "position", 4: "extended"}.get(s.operating_mode(), str(s.operating_mode()))
        print(f"{s.name:12s} id {s.id}  mode {mode:8s}  torque {'on ' if s.torque_enabled() else 'off'}  "
              f"wound {rig.wound(key):8.1f} deg  load {s.present_load():6.1f} %  "
              f"{s.present_voltage():4.1f} V  {s.present_temperature()} C  PWM limit {s.pwm_limit()}/{D.PWM_MAX}")
    t, tc, te = ring.t_from_both(rig.wound("con"), rig.wound("exp"))
    print(f"ring: t = {t:.3f} (contraction servo says {tc:.3f}, expansion servo {te:.3f})  "
          f"outer dia ~ {ring.outer_dia(t):.0f} mm")
    if abs(tc - te) > 0.01:
        print("  the two servos disagree: not homed since power-up, or a cable slipped/stretched")


def cmd_off(rig: Rig, ring: Ring, args):
    for s in rig.servo.values():
        s.disable_torque()
    print("torque off")


def cmd_jog(rig: Rig, ring: Ring, args):
    rig.ensure_mode()
    rig.torque_on_holding()
    s = rig.servo[args.cable]
    s.set_goal_pwm(JOG_PWM)
    s.set_acceleration_profile(0)
    s.set_velocity_profile(JOG_PROFILE_VELOCITY)
    goal = rig.wound(args.cable) + args.deg
    s.set_goal_position(rig.ticks(args.cable, goal))
    print(f"{s.name}: {'winding in' if args.deg > 0 else 'paying out'} {abs(args.deg):.0f} deg")
    try:
        time.sleep(0.1)
        while s.is_moving():
            time.sleep(0.05)
    except KeyboardInterrupt:
        rig.hold()
    rig.hold()   # a stalled jog would otherwise keep pulling at JOG_PWM
    print(f"{s.name}: wound {rig.wound(args.cable):.1f} deg, load {s.present_load():.1f} %")


def cmd_set_home(rig: Rig, ring: Ring, args):
    for s in rig.servo.values():
        s.disable_torque()
        if s.operating_mode() != D.MODE_EXTENDED_POSITION:
            s.set_operating_mode(D.MODE_EXTENDED_POSITION)
        raw = s.present_position() - s.homing_offset()
        s.set_homing_offset(-raw)
    for key, s in rig.servo.items():
        print(f"{s.name}: homed, now at {rig.wound(key):.1f} deg (torque off)")
    print(f"home = links touching, outer dia {ring.outer_dia(T_HOME):.1f} mm")


def move_to(rig: Rig, ring: Ring, t_target: float, speed: float):
    rig.ensure_mode()
    t_target = min(max(t_target, T_HOME), T_OPEN)
    t, tc, te = ring.t_from_both(rig.wound("con"), rig.wound("exp"))
    if abs(tc - te) > 0.02:
        raise SystemExit(f"servos disagree about the ring state (t {tc:.3f} vs {te:.3f}); run set-home first")
    opening = t_target > t
    leader, follower = ("exp", "con") if opening else ("con", "exp")
    rig.servo[leader].set_goal_pwm(LEADER_PWM)
    rig.servo[follower].set_goal_pwm(FOLLOWER_PWM)
    for s in rig.servo.values():
        s.set_acceleration_profile(0)
        s.set_velocity_profile(0)      # follow the streamed goals directly
    rig.torque_on_holding()

    print(f"{'opening' if opening else 'closing'}: t {t:.3f} -> {t_target:.3f}, "
          f"outer dia {ring.outer_dia(t):.0f} -> {ring.outer_dia(t_target):.0f} mm "
          f"({rig.servo[leader].name} leads)")
    step = speed / RATE_HZ
    over = lost = 0
    i = 0
    try:
        while True:
            # read before the sync write: a read sent right after it gets no reply
            if i % 5 == 0:
                try:
                    lc = rig.servo["con"].present_load()
                    le = rig.servo["exp"].present_load()
                    lost = 0
                except IOError as err:
                    lost += 1
                    if lost >= 3:
                        rig.hold()
                        print(f"\nstopped: no reply from a servo ({err}). Holding position.")
                        return
                else:
                    print(f"\r  t {t:.3f}  dia {ring.outer_dia(t):5.0f} mm  "
                          f"load con {lc:6.1f} %  exp {le:6.1f} %   ", end="", flush=True)
                    over = over + 1 if max(abs(lc), abs(le)) > LOAD_LIMIT else 0
                    if over >= 3:
                        rig.hold()
                        print(f"\nstopped: load above {LOAD_LIMIT:.0f} %. Holding position.")
                        return
            dt = step / ring.max_rate(t)
            t = min(t + dt, t_target) if opening else max(t - dt, t_target)
            c, e = ring.angles(t)
            rig.bus.sync_goal_positions({CON_ID: rig.ticks("con", c), EXP_ID: rig.ticks("exp", e)})
            if t == t_target:
                break
            i += 1
            time.sleep(1.0 / RATE_HZ)
        time.sleep(0.01)
        t0 = time.time()
        while any(s.is_moving() for s in rig.servo.values()) and time.time() - t0 < 2.0:
            time.sleep(0.05)
    except KeyboardInterrupt:
        time.sleep(0.01)
        rig.hold()
        print("\ninterrupted, holding position")
        return
    print()
    cmd_status(rig, ring, None)


def cmd_expand(rig: Rig, ring: Ring, args):
    move_to(rig, ring, T_OPEN, args.speed)


def cmd_contract(rig: Rig, ring: Ring, args):
    move_to(rig, ring, T_CLOSED, args.speed)


def cmd_goto(rig: Rig, ring: Ring, args):
    if args.dia is not None:
        t = ring.t_from_dia(args.dia)
    elif args.t is not None:
        t = args.t
    else:
        raise SystemExit("give t or --dia")
    move_to(rig, ring, t, args.speed)


def main():
    ap = argparse.ArgumentParser(prog="plicata", description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--port", help="serial port (default: first U2D2 found)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("off").set_defaults(fn=cmd_off)
    sub.add_parser("set-home").set_defaults(fn=cmd_set_home)
    j = sub.add_parser("jog", help="wind a cable in (positive deg) or out (negative)")
    j.add_argument("cable", choices=["con", "exp"])
    j.add_argument("deg", type=float)
    j.set_defaults(fn=cmd_jog)
    for name, fn in (("expand", cmd_expand), ("contract", cmd_contract)):
        s = sub.add_parser(name)
        s.add_argument("--speed", type=float, default=SPEED_DEG_S, help="deg/s of the faster servo")
        s.set_defaults(fn=fn)
    g = sub.add_parser("goto")
    g.add_argument("t", type=float, nargs="?")
    g.add_argument("--dia", type=float, help="outer diameter [mm]")
    g.add_argument("--speed", type=float, default=SPEED_DEG_S)
    g.set_defaults(fn=cmd_goto)
    args = ap.parse_args()

    rig = Rig(args.port)
    try:
        args.fn(rig, Ring(), args)
    finally:
        rig.close()


if __name__ == "__main__":
    main()
