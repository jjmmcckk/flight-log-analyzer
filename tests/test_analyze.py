"""Unit tests on synthetic log data with known answers."""
import numpy as np
import pandas as pd
import pytest

from flightlog.analyze import (analyse, armed_segments, mode_segments, run_checks, step_response,
                               transitions, DEFAULT_LIMITS)
from flightlog.load import FlightLog


def make_log(duration=60.0, rate=25.0, tau=0.3, armed_from=5.0, armed_to=55.0, alt=50.0, extra=None, messages=()):
    """A synthetic fixed wing log: bank angle steps every 10 s followed by a first order lag."""
    t = np.arange(0, duration, 1 / rate)
    des_roll = np.where((t // 10) % 2 == 1, 40.0, 0.0)
    roll = np.zeros_like(t)
    for i in range(1, len(t)):
        roll[i] = roll[i - 1] + (des_roll[i] - roll[i - 1]) * (1 - np.exp(-(1 / rate) / tau))
    flying = ((t >= armed_from + 1) & (t <= armed_to - 1)).astype(int)
    frames = {
        "ATT": pd.DataFrame({"t": t, "DesRoll": des_roll, "Roll": roll, "DesPitch": 0.0, "Pitch": 0.0}),
        "STAT": pd.DataFrame({"t": t, "Armed": ((t >= armed_from) & (t <= armed_to)).astype(int), "isFlying": flying}),
        "POS": pd.DataFrame({"t": t, "Lat": 43.9 + t * 1e-5, "Lng": -79.0 + 0 * t, "Alt": 100 + alt * flying,
                             "RelHomeAlt": alt * flying}),
        "MODE": pd.DataFrame({"t": [0.0, 30.0], "Mode": [5, 10], "ModeNum": [5, 10]}),
        "VIBE": pd.DataFrame({"t": t, "IMU": 0, "VibeX": 5.0, "VibeY": 5.0, "VibeZ": 8.0, "Clip": 0}),
        "XKF4": pd.DataFrame({"t": t, "C": 0, "SV": 0.2, "SP": 0.2, "SH": 0.1, "SM": 0.1}),
        "GPS": pd.DataFrame({"t": t, "I": 0, "Status": 3, "NSats": 12, "HDop": 0.8, "Spd": 20.0}),
    }
    frames.update(extra or {})
    msgs = [(0.5, "ArduPlane V4.6.0 (abcdef12)")] + list(messages)
    return FlightLog(path="synthetic.BIN", frames=frames, params={}, messages=msgs)


def test_step_response_matches_first_order_lag():
    tau = 0.3
    log = make_log(tau=tau)
    att = log.get("ATT")
    fly = np.ones(len(att), dtype=bool)
    r = step_response(att, "DesRoll", "Roll", fly)
    assert r["steps"] >= 4
    assert r["rise90_median_s"] == pytest.approx(np.log(10) * tau, abs=0.06)   # 90% rise = 2.3 tau
    assert r["overshoot_max_pct"] == pytest.approx(0, abs=1)
    assert r["settled_rms"] < 0.5


def test_slow_response_fails_rise_time_check():
    s = analyse(make_log(tau=1.0))
    rise = next(c for c in s["checks"] if c["check"].startswith("Roll step response, 90% rise"))
    assert rise["pass"] is False


def test_armed_segments_from_stat():
    segs = armed_segments(make_log(armed_from=5, armed_to=55))
    assert len(segs) == 1
    assert segs[0][0] == pytest.approx(5, abs=0.05)
    assert segs[0][1] == pytest.approx(55, abs=0.05)


def test_armed_from_events_when_disarm_comes_first():
    log = FlightLog(path="x", frames={"EV": pd.DataFrame({"t": [40.0], "Id": [11]}),
                                      "POS": pd.DataFrame({"t": [0.0, 45.0]})})
    assert armed_segments(log) == [(0.0, 40.0)]


def test_plane_mode_names():
    segs = mode_segments(make_log(), "ArduPlane")
    assert list(segs["mode"]) == ["FBWA", "AUTO"]


def test_high_vibration_and_ekf_fail():
    log = make_log()
    log.frames["VIBE"]["VibeZ"] = 45.0
    log.frames["XKF4"]["SV"] = 1.3
    checks = {c["check"]: c for c in analyse(log)["checks"]}
    assert checks["Vibration (max axis)"]["pass"] is False
    assert checks["EKF innovation test ratios (max)"]["pass"] is False


def test_never_armed_reports_prearm_reason():
    log = make_log(armed_from=1e9, armed_to=1e9, messages=[(2.0, "PreArm: GPS: Need 3D Fix")])
    checks = {c["check"]: c for c in analyse(log)["checks"]}
    assert checks["Aircraft armed"]["pass"] is False
    assert "Need 3D Fix" in checks["Pre-arm failures"]["value"]


def test_transition_timing_and_altitude_loss():
    t = np.arange(0, 30, 0.1)
    rel = np.where(t < 10, 30.0, np.where(t < 14, 30.0 - (t - 10), 26.0 + (t - 14) * 0.5))
    log = FlightLog(path="q", frames={"POS": pd.DataFrame({"t": t, "RelHomeAlt": rel}),
                                      "CTUN": pd.DataFrame({"t": t, "As": np.clip((t - 10) * 3, 0, 22)})},
                    messages=[(10.0, "Transition started"), (16.5, "Transition airspeed reached 19.0"),
                              (18.0, "Transition done")])
    tr = transitions(log)
    assert len(tr) == 1
    assert tr[0]["duration_s"] == pytest.approx(8.0)
    assert tr[0]["to_airspeed_s"] == pytest.approx(6.5)
    assert tr[0]["alt_loss_m"] == pytest.approx(4.0, abs=0.15)


def test_custom_limits_change_verdict():
    default = analyse(make_log(tau=0.3))
    strict = analyse(make_log(tau=0.3), {"rise90_s": 0.1})
    pick = lambda s: next(c for c in s["checks"] if c["check"].startswith("Roll step response, 90% rise"))
    assert pick(default)["pass"] is True
    assert pick(strict)["pass"] is False
    assert DEFAULT_LIMITS["rise90_s"] == 1.5
