"""Fly test card FT 001 in ArduPilot SITL: QuadPlane VTOL takeoff, transition to forward flight,
a waypoint circuit, fixed wing AUTOTUNE (roll and pitch stick doublets), a second circuit on the
tuned gains, back transition and VTOL landing.

Start SITL first, for example (from an ArduPilot checkout built with `./waf plane`):
  build/sitl/bin/arduplane --model quadplane --speedup 5 \
      --defaults Tools/autotest/default_params/plane.parm,Tools/autotest/default_params/quadplane.parm
then run:
  python sitl/fly_ft001.py --connect tcp:127.0.0.1:5760
The DataFlash log appears in SITL's logs/ folder.
"""
from __future__ import annotations

import argparse
import math
import time

from pymavlink import mavutil

M = mavutil.mavlink


def offset(lat, lon, north_m, east_m):
    return (lat + north_m / 111320.0, lon + east_m / (111320.0 * math.cos(math.radians(lat))))


def wait_msg(mav, kind, timeout=60, cond=None):
    end = time.time() + timeout
    while time.time() < end:
        m = mav.recv_match(type=kind, blocking=True, timeout=1)
        if m and (cond is None or cond(m)):
            return m
    raise TimeoutError(f"no {kind}")


def set_param(mav, name, value):
    mav.mav.param_set_send(mav.target_system, mav.target_component, name.encode(), float(value),
                           M.MAV_PARAM_TYPE_REAL32)
    wait_msg(mav, "PARAM_VALUE", 10, lambda m: m.param_id == name)


def upload_mission(mav, items):
    mav.mav.mission_clear_all_send(mav.target_system, mav.target_component)
    wait_msg(mav, "MISSION_ACK", 10)
    mav.mav.mission_count_send(mav.target_system, mav.target_component, len(items))
    sent = set()
    while len(sent) < len(items):
        req = wait_msg(mav, ["MISSION_REQUEST_INT", "MISSION_REQUEST"], 20)
        seq = req.seq
        cmd, lat, lon, alt, p1, p2 = (items[seq] + (0,))[:6]
        mav.mav.mission_item_int_send(mav.target_system, mav.target_component, seq,
                                      M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, cmd, 0, 1, p1, p2, 0, 0,
                                      int(lat * 1e7), int(lon * 1e7), alt)
        sent.add(seq)
    ack = wait_msg(mav, "MISSION_ACK", 20)
    if ack.type != M.MAV_MISSION_ACCEPTED:
        raise RuntimeError(f"mission rejected: {ack.type}")


def set_mode(mav, number):
    mav.mav.command_long_send(mav.target_system, mav.target_component, M.MAV_CMD_DO_SET_MODE, 0,
                              M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, number, 0, 0, 0, 0, 0)


def rc(mav, roll=1500, pitch=1500, throttle=1000, yaw=1500):
    # RC override on channels 1 to 4; other channels released
    mav.mav.rc_channels_override_send(mav.target_system, mav.target_component,
                                      roll, pitch, throttle, yaw, 0, 0, 0, 0)


def rc_neutral(mav):
    rc(mav)


def print_texts(mav):
    while True:
        m = mav.recv_match(type="STATUSTEXT", blocking=False)
        if m is None:
            return
        print("  ", m.text)


def autotune(mav, target_alt=100.0, max_axis_s=300):
    """Fixed wing AUTOTUNE: sharp roll doublets until roll is finished, then pitch doublets
    (in a gentle turn so the aircraft stays near the field) until pitch is finished.
    Throttle holds the circuit altitude, as a pilot would during the test point."""
    print("AUTOTUNE test point")
    set_mode(mav, 8)  # AUTOTUNE
    state = {"alt": target_alt}

    def pump():
        while True:
            m = mav.recv_match(type=["STATUSTEXT", "GLOBAL_POSITION_INT"], blocking=False)
            if m is None:
                return None
            if m.get_type() == "GLOBAL_POSITION_INT":
                state["alt"] = m.relative_alt / 1000.0
            else:
                print("  ", m.text)
                if m.text.endswith("Finished"):
                    return m.text

    for axis in ("Roll", "Pitch"):
        end = time.time() + max_axis_s
        while time.time() < end:
            throttle = int(min(1800, max(1200, 1450 + 15 * (target_alt - state["alt"]))))
            phase = int((time.time() % 3.0) // 0.75)            # one way, centre, other way, centre
            if axis == "Roll":
                rc(mav, roll=[1900, 1500, 1100, 1500][phase], throttle=throttle)
            else:
                rc(mav, roll=1600, pitch=[1700, 1500, 1300, 1500][phase], throttle=throttle)
            done = pump()
            if done and done.startswith(axis):
                break
            time.sleep(0.1)
    set_mode(mav, 10)  # back to AUTO: leaving AUTOTUNE keeps the new gains
    print("AUTOTUNE done, back to AUTO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--leg", type=float, default=600.0, help="Circuit leg length in metres")
    ap.add_argument("--alt", type=float, default=100.0, help="Circuit altitude in metres")
    ap.add_argument("--autotune", action="store_true", help="Run AUTOTUNE after the first circuit, then fly it again")
    a = ap.parse_args()

    mav = mavutil.mavlink_connection(a.connect, source_system=255)
    mav.wait_heartbeat(timeout=60)
    while mav.target_system == 0:
        mav.wait_heartbeat(timeout=60)
    print("connected to system", mav.target_system)
    mav.mav.request_data_stream_send(mav.target_system, mav.target_component, M.MAV_DATA_STREAM_ALL, 10, 1)
    for name, value in [("LOG_DISARMED", 0), ("Q_ASSIST_SPEED", 12), ("LOG_FILE_RATEMAX", 10)]:
        try:
            set_param(mav, name, value)
        except TimeoutError:
            print("param not set:", name)

    gps = wait_msg(mav, "GLOBAL_POSITION_INT", 60, lambda m: m.lat != 0)
    home_lat, home_lon = gps.lat / 1e7, gps.lon / 1e7
    L = a.leg
    wps = [offset(home_lat, home_lon, n, e) for n, e in [(L, 0), (L, L), (0, L), (-150, L / 2)]]
    land = offset(home_lat, home_lon, 0, 0)
    items = [(M.MAV_CMD_NAV_WAYPOINT, home_lat, home_lon, 0, 0),
             (M.MAV_CMD_NAV_VTOL_TAKEOFF, home_lat, home_lon, 30, 0)]
    items += [(M.MAV_CMD_NAV_WAYPOINT, la, lo, a.alt, 0) for la, lo in wps]
    if a.autotune:
        items += [(M.MAV_CMD_DO_JUMP, 0, 0, 0, 2, 1)]          # fly the circuit a second time
    items += [(M.MAV_CMD_NAV_VTOL_LAND, land[0], land[1], 0, 0)]
    upload_mission(mav, items)
    print("mission uploaded:", len(items), "items")

    # Wait until the vehicle reports it is ready to arm, then arm in AUTO
    end = time.time() + 180
    while time.time() < end:
        rc_neutral(mav)
        set_mode(mav, 10)  # AUTO
        mav.mav.command_long_send(mav.target_system, mav.target_component,
                                  M.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        hb = wait_msg(mav, "HEARTBEAT", 5, lambda m: m.get_srcSystem() == mav.target_system)
        if hb.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED:
            print("armed in AUTO")
            break
        time.sleep(2)
    else:
        raise RuntimeError("could not arm")

    # Fly until the vehicle disarms after landing
    tuned = False
    end = time.time() + 1200
    while time.time() < end:
        rc_neutral(mav)
        m = mav.recv_match(type=["STATUSTEXT", "HEARTBEAT"], blocking=True, timeout=1)
        if m is None:
            continue
        if m.get_type() == "STATUSTEXT":
            print("  ", m.text)
            if a.autotune and not tuned and m.text.startswith("Reached waypoint #5"):
                autotune(mav)
                tuned = True
        elif m.get_srcSystem() == mav.target_system and not (m.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED):
            print("disarmed: flight complete")
            return
    raise RuntimeError("flight did not finish")


if __name__ == "__main__":
    main()
