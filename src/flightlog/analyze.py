"""Flight test metrics from an ArduPilot DataFlash log."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pymavlink import mavutil

from .load import FlightLog

EVENTS = {10: "Armed", 11: "Disarmed", 15: "Auto armed", 16: "Takeoff", 18: "Land complete maybe",
          19: "Land complete", 25: "Set home", 28: "Not landed"}

ERR_SUBSYSTEMS = {
    1: "Main", 2: "Radio", 3: "Compass", 4: "Optical flow", 5: "Radio failsafe", 6: "Battery failsafe",
    7: "GPS failsafe", 8: "GCS failsafe", 9: "Fence failsafe", 10: "Flight mode", 11: "GPS",
    12: "Crash check", 13: "Flip", 14: "Autotune", 15: "Parachute", 16: "EKF check",
    17: "EKF inertial nav failsafe", 18: "Barometer", 19: "CPU", 20: "ADSB failsafe", 21: "Terrain",
    22: "Navigation", 23: "Terrain failsafe", 24: "EKF primary", 25: "Thrust loss", 26: "Sensor failsafe",
    27: "Leak failsafe", 28: "Pilot input", 29: "Vibration failsafe", 30: "Internal error",
    31: "Dead reckoning failsafe",
}

NOTABLE = re.compile(r"PreArm|Arm:|failsafe|Failsafe|EKF.*(fail|reset|variance)|Crash|Error|error|"
                     r"[Tt]ransition|Mission|Reached|Land|Takeoff|AUTOTUNE|[Aa]utotune|GPS Glitch|Vibration")


# ---------------------------------------------------------------- identification
def vehicle_info(log: FlightLog) -> dict:
    firmware, vehicle = None, None
    for _, text in log.messages:
        m = re.match(r"(Ardu\w+|ArduPilot\w*) (V[\w.\-]+)", text)
        if m:
            firmware, vehicle = text.split(" (")[0], m.group(1)
            break
    quadplane = vehicle == "ArduPlane" and log.params.get("Q_ENABLE", 0) >= 1
    return {"firmware": firmware, "vehicle": vehicle, "quadplane": bool(quadplane),
            "frame": "QuadPlane (VTOL)" if quadplane else (vehicle or "unknown").replace("Ardu", "")}


def mode_names(vehicle: str | None) -> dict[int, str]:
    if vehicle == "ArduPlane":
        return mavutil.mode_mapping_apm
    if vehicle == "ArduCopter":
        return mavutil.mode_mapping_acm
    if vehicle == "ArduRover":
        return mavutil.mode_mapping_rover
    return {}


def duration(log: FlightLog) -> float:
    """Length of the logged data. Text messages are ignored, since the autopilot can keep writing
    them for a while after the last flight data."""
    ends = [df["t"].max() for name, df in log.frames.items() if name != "MSG" and "t" in df and len(df)]
    return float(max(ends)) if ends else 0.0


# ---------------------------------------------------------------- timeline
def mode_segments(log: FlightLog, vehicle: str | None) -> pd.DataFrame:
    df = log.get("MODE")
    end = duration(log)
    if df.empty:
        return pd.DataFrame(columns=["mode", "start", "end", "duration"])
    names = mode_names(vehicle)
    col = "ModeNum" if "ModeNum" in df else "Mode"
    starts = df["t"].to_numpy()
    ends = np.append(starts[1:], end)
    modes = [names.get(int(n), f"Mode {int(n)}") for n in df[col]]
    seg = pd.DataFrame({"mode": modes, "start": starts, "end": ends})
    seg["duration"] = seg["end"] - seg["start"]
    return seg


def _edges(t: np.ndarray, state: np.ndarray, end: float) -> list[tuple[float, float]]:
    on = np.r_[0, (state > 0).astype(int), 0]
    edges = np.flatnonzero(np.diff(on))
    return [(float(t[a]), float(t[b]) if b < len(t) else end) for a, b in zip(edges[::2], edges[1::2])]


def armed_segments(log: FlightLog) -> list[tuple[float, float]]:
    """Armed periods: from STAT.Armed (sampled continuously), else ARM messages, else EV events."""
    end = duration(log)
    if log.has("STAT", "Armed"):
        st = log.get("STAT")
        return _edges(st["t"].to_numpy(), st["Armed"].to_numpy(), end)
    if log.has("ARM", "ArmState"):
        arm = log.get("ARM")
        return _edges(arm["t"].to_numpy(), arm["ArmState"].to_numpy(), end)
    ev = log.get("EV")
    segs, start = [], None
    for t, i in zip(ev.get("t", []), ev.get("Id", [])):
        if i in (10, 15):
            if start is None:
                start = t
        elif i == 11:
            if start is not None:
                segs.append((start, t))
                start = None
            elif not segs:            # disarmed with no arm event logged: armed since the log began
                segs.append((0.0, t))
    if start is not None:
        segs.append((start, end))
    return segs


def flight_phases(log: FlightLog) -> list[tuple[float, float]]:
    """Airborne periods from STAT.isFlying."""
    if not log.has("STAT", "isFlying"):
        return []
    st = log.get("STAT")
    return _edges(st["t"].to_numpy(), st["isFlying"].to_numpy(), duration(log))


def flying_mask(df: pd.DataFrame, log: FlightLog) -> np.ndarray:
    """True where the aircraft is flying (STAT.isFlying when logged, otherwise armed)."""
    if df.empty:
        return np.zeros(0, dtype=bool)
    if log.has("STAT", "isFlying"):
        st = log.get("STAT")
        idx = np.searchsorted(st["t"].to_numpy(), df["t"].to_numpy(), side="right") - 1
        return st["isFlying"].to_numpy()[np.clip(idx, 0, len(st) - 1)] > 0
    mask = np.zeros(len(df), dtype=bool)
    for a, b in armed_segments(log):
        mask |= (df["t"] >= a).to_numpy() & (df["t"] <= b).to_numpy()
    return mask


def airborne_mask(df: pd.DataFrame, log: FlightLog, min_alt_m: float = 5.0) -> np.ndarray:
    """Flying and more than min_alt_m above home, so takeoff runs and touchdowns (where the
    demanded attitude cannot be followed on the ground) do not count against tracking."""
    fly = flying_mask(df, log)
    pos = log.get("POS")
    if df.empty or pos.empty or "RelHomeAlt" not in pos:
        return fly
    alt = np.interp(df["t"].to_numpy(), pos["t"].to_numpy(), pos["RelHomeAlt"].to_numpy(float))
    return fly & (alt > min_alt_m)


def vtol_mask(df: pd.DataFrame, log: FlightLog, gap_s: float = 0.5) -> np.ndarray:
    """True where a QuadPlane is flying on its lift motors (QTUN is logged then)."""
    q = log.get("QTUN")
    if df.empty or q.empty:
        return np.zeros(len(df), dtype=bool)
    qt = q["t"].to_numpy()
    t = df["t"].to_numpy()
    i = np.clip(np.searchsorted(qt, t), 1, len(qt) - 1)
    nearest = np.minimum(np.abs(t - qt[i - 1]), np.abs(qt[i] - t))
    return nearest <= gap_s


def in_modes(df: pd.DataFrame, segs: pd.DataFrame, modes) -> np.ndarray:
    m = np.zeros(len(df), dtype=bool)
    if df.empty or segs.empty:
        return m
    t = df["t"].to_numpy()
    for _, s in segs[segs["mode"].isin(modes)].iterrows():
        m |= (t >= s["start"]) & (t < s["end"])
    return m


TUNING_MODES = ("AUTOTUNE", "QAUTOTUNE")


def fw_mask(df: pd.DataFrame, log: FlightLog, segs: pd.DataFrame) -> np.ndarray:
    """Airborne, not on lift motors, and not in a tuning mode: where fixed wing tracking is judged."""
    return airborne_mask(df, log) & ~vtol_mask(df, log) & ~in_modes(df, segs, TUNING_MODES)


def autotune_comparison(log: FlightLog, segs: pd.DataFrame) -> dict:
    """Step response before and after an AUTOTUNE test point in the same flight, plus the gains."""
    at = segs[segs["mode"].isin(TUNING_MODES)] if not segs.empty else segs
    att = log.get("ATT")
    if at.empty or att.empty:
        return {}
    a0, a1 = float(at["start"].min()), float(at["end"].max())
    base = fw_mask(att, log, segs)
    t = att["t"].to_numpy()
    out = {"start_s": round(a0, 1), "end_s": round(a1, 1), "axes": {}}
    for axis in ("Roll", "Pitch"):
        out["axes"][axis.lower()] = {
            "before": step_response(att, f"Des{axis}", axis, base & (t < a0)),
            "after": step_response(att, f"Des{axis}", axis, base & (t > a1)),
        }
    names = {"RLL_RATE_P": "Roll P", "RLL_RATE_I": "Roll I", "RLL_RATE_D": "Roll D", "RLL_RATE_FF": "Roll FF",
             "PTCH_RATE_P": "Pitch P", "PTCH_RATE_I": "Pitch I", "PTCH_RATE_D": "Pitch D", "PTCH_RATE_FF": "Pitch FF"}
    gains = []
    for key, label in names.items():
        b, f = log.params_initial.get(key), log.params.get(key)
        if b is not None and f is not None:
            gains.append({"gain": label, "before": round(b, 4), "after": round(f, 4)})
    out["gains"] = gains
    out["messages"] = [m for _, m in log.messages if re.match(r"(Roll|Pitch|Yaw)\w*: ", m)]
    return out


def events(log: FlightLog) -> list[dict]:
    out = []
    ev = log.get("EV")
    for t, i in zip(ev.get("t", []), ev.get("Id", [])):
        out.append({"t": round(float(t), 1), "kind": "event", "text": EVENTS.get(int(i), f"Event {int(i)}")})
    err = log.get("ERR")
    for t, s, c in zip(err.get("t", []), err.get("Subsys", []), err.get("ECode", [])):
        name = ERR_SUBSYSTEMS.get(int(s), f"Subsystem {int(s)}")
        out.append({"t": round(float(t), 1), "kind": "error" if c else "cleared",
                    "text": f"{name}: {'code ' + str(int(c)) if c else 'cleared'}"})
    for t, text in log.messages:
        if NOTABLE.search(text) and not text.startswith("SRC="):
            out.append({"t": round(float(t), 1), "kind": "message", "text": text})
    return sorted(out, key=lambda e: e["t"])


# ---------------------------------------------------------------- performance
def tracking(df: pd.DataFrame, desired: str, actual: str, mask: np.ndarray | None = None) -> dict:
    if df.empty or desired not in df or actual not in df:
        return {}
    d, a = df[desired].to_numpy(float), df[actual].to_numpy(float)
    if mask is not None and len(mask) == len(d):
        d, a = d[mask], a[mask]
    if d.size == 0:
        return {}
    err = d - a
    return {"rms": float(np.sqrt(np.mean(err ** 2))), "max_abs": float(np.max(np.abs(err))),
            "mean": float(np.mean(err)), "samples": int(err.size)}


def step_response(df: pd.DataFrame, desired: str, actual: str, fly: np.ndarray,
                  min_step: float = 15.0, window_s: float = 5.0) -> dict:
    """Response to step changes in the demand (for example a new waypoint bank angle):
    time for the actual angle to cover 90% of the step, and overshoot past the new demand.
    Also returns tracking RMS with 3 s after each step removed (settled tracking)."""
    if df.empty or desired not in df or actual not in df:
        return {}
    t, d, a = df["t"].to_numpy(), df[desired].to_numpy(float), df[actual].to_numpy(float)
    jumps = np.flatnonzero((np.abs(np.diff(d)) >= min_step) & fly[1:]) + 1
    rises, overs, settle_mask = [], [], fly.copy()
    for i in jumps:
        start, target = a[i - 1], d[i]
        span = target - start
        if abs(span) < min_step:
            continue
        w = (t >= t[i]) & (t <= t[i] + window_s)
        frac = (a[w] - start) / span
        hit = np.flatnonzero(frac >= 0.9)
        if hit.size:
            rises.append(float(t[w][hit[0]] - t[i]))
        overs.append(float(max(0.0, (frac.max() - 1.0) * 100)))
        settle_mask &= ~((t >= t[i]) & (t < t[i] + 3.0))
    err = (d - a)[settle_mask]
    out = {"steps": len(rises), "settled_rms": float(np.sqrt(np.mean(err ** 2))) if err.size else None}
    if rises:
        out["rise90_median_s"] = float(np.median(rises))
        out["rise90_max_s"] = float(np.max(rises))
        out["overshoot_max_pct"] = float(np.max(overs))
    return out


def attitude_response(log: FlightLog, segs: pd.DataFrame | None = None) -> dict:
    att = log.get("ATT")
    if att.empty:
        return {}
    fly = fw_mask(att, log, segs if segs is not None else pd.DataFrame(columns=["mode", "start", "end"]))
    return {"roll": step_response(att, "DesRoll", "Roll", fly), "pitch": step_response(att, "DesPitch", "Pitch", fly)}


def attitude_by_mode(log: FlightLog, segs: pd.DataFrame) -> list[dict]:
    att = log.get("ATT")
    if att.empty or segs.empty:
        return []
    fly = airborne_mask(att, log)
    rows = []
    for _, s in segs.iterrows():
        m = fly & (att["t"] >= s["start"]).to_numpy() & (att["t"] < s["end"]).to_numpy()
        if m.sum() < 5:
            continue
        r = tracking(att, "DesRoll", "Roll", m)
        p = tracking(att, "DesPitch", "Pitch", m)
        rows.append({"mode": s["mode"], "start": round(float(s["start"]), 1), "seconds": round(float(m.sum() and s["duration"]), 1),
                     "roll_rms": round(r["rms"], 2), "roll_max": round(r["max_abs"], 2),
                     "pitch_rms": round(p["rms"], 2), "pitch_max": round(p["max_abs"], 2)})
    return rows


def navigation(log: FlightLog, segs: pd.DataFrame | None = None) -> dict:
    """Altitude, airspeed and cross track tracking in fixed wing flight (TECS and NTUN)."""
    segs = segs if segs is not None else pd.DataFrame(columns=["mode", "start", "end"])
    out = {}
    tecs = log.get("TECS")
    if not tecs.empty:
        fly = fw_mask(tecs, log, segs)
        alt = tracking(tecs, "hdem", "h", fly)
        spd = tracking(tecs, "spdem", "sp", fly)
        if alt:
            out["altitude_m"] = alt
        if spd:
            out["airspeed_ms"] = spd
    ntun = log.get("NTUN")
    if not ntun.empty and "XT" in ntun:
        fly = fw_mask(ntun, log, segs)
        xt = ntun["XT"].to_numpy(float)[fly] if fly.any() else np.array([])
        if xt.size:
            out["cross_track_m"] = {"rms": float(np.sqrt(np.mean(xt ** 2))), "max_abs": float(np.max(np.abs(xt)))}
    return out


def mission(log: FlightLog) -> list[dict]:
    cmd = log.get("CMD")
    if cmd.empty:
        return []
    names = {v.value if hasattr(v, "value") else k: v.name for k, v in mavutil.mavlink.enums.get("MAV_CMD", {}).items()}
    rows, seen = [], set()
    for _, r in cmd.iterrows():
        key = int(r["CNum"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"seq": key, "command": names.get(int(r["CId"]), str(int(r["CId"]))).replace("MAV_CMD_", ""),
                     "alt_m": round(float(r["Alt"]), 1)})
    return rows


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def flight_summary(log: FlightLog) -> dict:
    out: dict = {}
    pos = log.get("POS")
    if not pos.empty:
        lat, lon = pos["Lat"].to_numpy(float), pos["Lng"].to_numpy(float)
        ok = (np.abs(lat) > 1e-6) | (np.abs(lon) > 1e-6)
        lat, lon = lat[ok], lon[ok]
        if lat.size > 1:
            out["distance_flown_m"] = float(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum())
            out["max_distance_from_start_m"] = float(haversine_m(lat[0], lon[0], lat, lon).max())
        if "RelHomeAlt" in pos:
            out["max_alt_above_home_m"] = float(pos["RelHomeAlt"].max())
    gps = log.get("GPS")
    if not gps.empty and "Spd" in gps:
        out["max_ground_speed_ms"] = float(gps["Spd"].max())
    ctun = log.get("CTUN")
    if not ctun.empty and "As" in ctun:
        out["max_airspeed_ms"] = float(ctun["As"].max())
    segs = armed_segments(log)
    out["armed_time_s"] = float(sum(b - a for a, b in segs))
    phases = flight_phases(log)
    if phases:
        out["flying_time_s"] = float(sum(b - a for a, b in phases))
        out["takeoff_s"] = round(phases[0][0], 1)
        out["landing_s"] = round(phases[-1][1], 1)
    return out


def transitions(log: FlightLog) -> list[dict]:
    """QuadPlane forward transitions: time from the start of the transition to 'done', with the
    altitude change and the airspeed reached. Uses QTUN.Trn when logged, else MSG text."""
    out = []
    starts = [t for t, m in log.messages if re.search(r"Transition (started|FW)|start(ing)? transition", m, re.I)]
    airspeed_reached = [t for t, m in log.messages if re.search(r"Transition airspeed reached", m, re.I)]
    done = [t for t, m in log.messages if re.search(r"Transition done|transition complete", m, re.I)]
    qtun = log.get("QTUN")
    if not starts and not qtun.empty and "Trn" in qtun:
        trn = qtun["Trn"].to_numpy()
        t = qtun["t"].to_numpy()
        rising = np.flatnonzero((trn[1:] > 0) & (trn[:-1] == 0)) + 1
        starts = list(t[rising])
    pos, ctun = log.get("POS"), log.get("CTUN")
    for s in starts:
        d = next((x for x in done if x >= s), None)
        a = next((x for x in airspeed_reached if x >= s), None)
        if d is None:
            continue
        row = {"start_s": round(s, 1), "duration_s": round(d - s, 1)}
        if a is not None:
            row["to_airspeed_s"] = round(a - s, 1)
        if not pos.empty:
            w = pos[(pos["t"] >= s) & (pos["t"] <= d)]
            if len(w):
                row["alt_change_m"] = round(float(w["RelHomeAlt"].iloc[-1] - w["RelHomeAlt"].iloc[0]), 1)
                row["min_alt_m"] = round(float(w["RelHomeAlt"].min()), 1)
                row["alt_loss_m"] = round(float(w["RelHomeAlt"].iloc[0] - w["RelHomeAlt"].min()), 1)
        if not ctun.empty and "As" in ctun:
            w = ctun[(ctun["t"] >= s) & (ctun["t"] <= d)]
            if len(w):
                row["airspeed_end_ms"] = round(float(w["As"].iloc[-1]), 1)
        out.append(row)
    return out


# ---------------------------------------------------------------- health
def gps_health(log: FlightLog) -> dict:
    gps = log.get("GPS")
    if gps.empty:
        return {}
    if "I" in gps:
        gps = gps[gps["I"] == gps["I"].min()]
    fly = flying_mask(gps, log)
    if fly.any():
        g = gps[fly]
    else:                                   # ground only log: ignore the acquisition period before a 3D fix
        g = gps[gps["Status"] >= 3] if "Status" in gps and (gps["Status"] >= 3).any() else gps
    return {"min_sats": int(g["NSats"].min()), "max_hdop": float(g["HDop"].max()), "flying_only": bool(fly.any())}


def ekf_health(log: FlightLog) -> dict:
    df = log.get("XKF4") if log.has("XKF4") else log.get("NKF4")
    if df.empty:
        return {}
    if "C" in df:
        df = df[df["C"] == df["C"].min()]
    fly = flying_mask(df, log)
    d = df[fly] if fly.any() else df
    out = {k: float(d[k].max()) for k in ["SV", "SP", "SH", "SM"] if k in d}
    out["flying_only"] = bool(fly.any())
    return out


def vibration(log: FlightLog) -> dict:
    v = log.get("VIBE")
    if v.empty:
        return {}
    if "IMU" in v:
        v = v[v["IMU"] == v["IMU"].min()]
    fly = flying_mask(v, log)
    d = v[fly] if fly.any() else v
    return {"max_x": float(d["VibeX"].max()), "max_y": float(d["VibeY"].max()), "max_z": float(d["VibeZ"].max()),
            "clips": int(v["Clip"].max() - v["Clip"].min()) if "Clip" in v else 0, "flying_only": bool(fly.any())}


def battery(log: FlightLog) -> dict:
    b = log.get("BAT")
    if b.empty:
        return {}
    if "Inst" in b:
        b = b[b["Inst"] == b["Inst"].min()]
    out = {"start_v": float(b["Volt"].iloc[0]), "min_v": float(b["Volt"].min()),
           "max_current_a": float(b["Curr"].max()) if "Curr" in b else None}
    if "CurrTot" in b:
        out["used_mah"] = float(b["CurrTot"].max() - b["CurrTot"].min())
    return out


def prearm_failures(log: FlightLog) -> list[str]:
    seen = []
    for _, text in log.messages:
        if text.startswith("PreArm") and text not in seen:
            seen.append(text)
    return seen


# ---------------------------------------------------------------- checks
def run_checks(s: dict, limits: dict) -> list[dict]:
    """Pass or fail against flight test limits. Returns rows of check, value, target, pass (None = info)."""
    rows = []
    armed = s["summary"].get("armed_time_s", 0) > 0
    rows.append({"check": "Aircraft armed", "value": "yes" if armed else "no",
                 "target": "yes", "pass": armed})
    if not armed and s["prearm"]:
        rows.append({"check": "Pre-arm failures", "value": "; ".join(s["prearm"][:3]), "target": "none", "pass": False})
    resp = s.get("response", {})
    for axis in ("roll", "pitch"):
        r = resp.get(axis, {})
        if r.get("settled_rms") is not None:
            rows.append({"check": f"{axis.title()} tracking, settled (RMS)", "value": f"{r['settled_rms']:.1f}°",
                         "target": f"≤ {limits['att_rms_deg']:g}°", "pass": r["settled_rms"] <= limits["att_rms_deg"]})
        if "rise90_median_s" in r:
            rows.append({"check": f"{axis.title()} step response, 90% rise (median of {r['steps']})",
                         "value": f"{r['rise90_median_s']:.2f} s", "target": f"≤ {limits['rise90_s']:g} s",
                         "pass": r["rise90_median_s"] <= limits["rise90_s"]})
            rows.append({"check": f"{axis.title()} step response, overshoot (max)", "value": f"{r['overshoot_max_pct']:.1f}%",
                         "target": f"≤ {limits['overshoot_pct']:g}%", "pass": r["overshoot_max_pct"] <= limits["overshoot_pct"]})
    for tr in s["transitions"]:
        if "alt_loss_m" in tr:
            rows.append({"check": f"Transition at {tr['start_s']} s: altitude loss", "value": f"{tr['alt_loss_m']:.1f} m",
                         "target": f"≤ {limits['transition_alt_loss_m']:g} m", "pass": tr["alt_loss_m"] <= limits["transition_alt_loss_m"]})
        rows.append({"check": f"Transition at {tr['start_s']} s: time to complete", "value": f"{tr['duration_s']:.1f} s",
                     "target": f"≤ {limits['transition_time_s']:g} s", "pass": tr["duration_s"] <= limits["transition_time_s"]})
    nav = s.get("navigation", {})
    for key, label, unit, lim in [("altitude_m", "Altitude tracking, fixed wing (RMS)", "m", "alt_rms_m"),
                                  ("airspeed_ms", "Airspeed tracking, fixed wing (RMS)", "m/s", "spd_rms_ms")]:
        if key in nav:
            v, L = nav[key]["rms"], limits.get(lim)
            rows.append({"check": label, "value": f"{v:.1f} {unit}", "target": f"≤ {L:g} {unit}" if L else "reported",
                         "pass": (v <= L) if L else None})
    if "cross_track_m" in nav:
        v, L = nav["cross_track_m"]["max_abs"], limits.get("xtrack_m")
        rows.append({"check": "Cross track error (max)", "value": f"{v:.1f} m", "target": f"≤ {L:g} m" if L else "reported",
                     "pass": (v <= L) if L else None})
    g = s["gps"]
    if g:
        rows.append({"check": "GPS satellites (minimum)", "value": str(g["min_sats"]),
                     "target": f"≥ {limits['min_sats']}", "pass": g["min_sats"] >= limits["min_sats"]})
        rows.append({"check": "GPS HDOP (maximum)", "value": f"{g['max_hdop']:.2f}",
                     "target": f"≤ {limits['max_hdop']:g}", "pass": g["max_hdop"] <= limits["max_hdop"]})
    e = s["ekf"]
    if e:
        worst = max(e.get(k, 0) for k in ["SV", "SP", "SH", "SM"])
        rows.append({"check": "EKF innovation test ratios (max)", "value": f"{worst:.2f}",
                     "target": "< 1.0", "pass": worst < 1.0})
    v = s["vibration"]
    if v:
        worst = max(v["max_x"], v["max_y"], v["max_z"])
        rows.append({"check": "Vibration (max axis)", "value": f"{worst:.1f} m/s²",
                     "target": f"< {limits['vibe_ms2']:g} m/s²", "pass": worst < limits["vibe_ms2"]})
        rows.append({"check": "Accelerometer clipping", "value": str(v["clips"]), "target": "0", "pass": v["clips"] == 0})
    errs = [e for e in s["events"] if e["kind"] == "error"]
    rows.append({"check": "Error events (ERR)", "value": str(len(errs)), "target": "0", "pass": len(errs) == 0})
    b = s["battery"]
    if b:
        rows.append({"check": "Battery minimum voltage", "value": f"{b['min_v']:.2f} V", "target": "reported", "pass": None})
    return rows


# Health limits follow ArduPilot guidance (vibration under 30 m/s², EKF test ratios under 1).
# Tracking limits are starting points; pass a test card JSON to set your own. A limit of None
# reports the value without a pass or fail.
DEFAULT_LIMITS = {"att_rms_deg": 3.0, "rise90_s": 1.5, "overshoot_pct": 20.0,
                  "alt_rms_m": 5.0, "spd_rms_ms": None, "xtrack_m": None,
                  "transition_alt_loss_m": 5.0, "transition_time_s": 15.0,
                  "min_sats": 6, "max_hdop": 2.0, "vibe_ms2": 30.0}


def analyse(log: FlightLog, limits: dict | None = None) -> dict:
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    info = vehicle_info(log)
    segs = mode_segments(log, info["vehicle"])
    s = {
        "info": info,
        "duration_s": duration(log),
        "summary": flight_summary(log),
        "modes": segs,
        "armed": armed_segments(log),
        "attitude": attitude_by_mode(log, segs),
        "response": attitude_response(log, segs),
        "autotune": autotune_comparison(log, segs),
        "transitions": transitions(log) if info["quadplane"] else [],
        "navigation": navigation(log, segs),
        "mission": mission(log),
        "gps": gps_health(log),
        "ekf": ekf_health(log),
        "vibration": vibration(log),
        "battery": battery(log),
        "events": events(log),
        "prearm": prearm_failures(log),
        "limits": limits,
    }
    s["checks"] = run_checks(s, limits)
    return s
