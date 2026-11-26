"""Write the flight test report: HTML with charts, JSON summary, CSV tables and a GeoJSON track."""
from __future__ import annotations

import base64
import html
import io
import json
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .load import FlightLog  # noqa: E402

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _mode_colors(segs: pd.DataFrame) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(dict.fromkeys(segs["mode"]))}


def _shade_modes(ax, segs, colors):
    for _, s in segs.iterrows():
        ax.axvspan(s["start"], s["end"], color=colors[s["mode"]], alpha=0.10, lw=0)


def overview_figure(log: FlightLog, s: dict) -> str | None:
    pos, ctun, segs = log.get("POS"), log.get("CTUN"), s["modes"]
    if pos.empty and ctun.empty:
        return None
    colors = _mode_colors(segs) if not segs.empty else {}
    fig, ax = plt.subplots(figsize=(11, 3.6))
    if not segs.empty:
        _shade_modes(ax, segs, colors)
    if not pos.empty:
        ax.plot(pos["t"], pos["RelHomeAlt"], color="#1f3b73", lw=1.4, label="Altitude above home (m)")
    ax.set_ylabel("Altitude (m)")
    ax2 = ax.twinx()
    if not ctun.empty and "As" in ctun:
        ax2.plot(ctun["t"], ctun["As"], color="#e15759", lw=1.0, label="Airspeed (m/s)")
        ax2.set_ylabel("Airspeed (m/s)")
    for e in s["events"]:
        if e["kind"] == "event" and e["text"] in ("Armed", "Disarmed", "Auto armed"):
            ax.axvline(e["t"], color="#333", ls=":", lw=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.35) for c in colors.values()]
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines + handles, labels + list(colors), loc="upper right", fontsize=8, ncol=2)
    ax.set_xlabel("Time (s)")
    ax.set_title("Flight overview (shading = flight mode)")
    return _png(fig)


def attitude_figure(log: FlightLog, s: dict) -> str | None:
    att = log.get("ATT")
    if att.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    tune = s["modes"][s["modes"]["mode"].isin(["AUTOTUNE", "QAUTOTUNE"])] if not s["modes"].empty else s["modes"]
    for ax in axes:
        for _, sg in tune.iterrows():
            ax.axvspan(sg["start"], sg["end"], color="#999", alpha=0.18, lw=0)
            ax.text(sg["start"], 0.97, f" {sg['mode']}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color="#555")
    for ax, axis in zip(axes, ["Roll", "Pitch"]):
        ax.plot(att["t"], att[f"Des{axis}"], lw=1.0, color="#4e79a7", label=f"Desired {axis.lower()}")
        ax.plot(att["t"], att[axis], lw=1.0, color="#f28e2b", label=f"Actual {axis.lower()}")
        ax.set_ylabel("degrees"); ax.legend(loc="upper right", fontsize=8)
        r = s["response"].get(axis.lower(), {})
        if r.get("settled_rms") is not None:
            ax.set_title(f"{axis}: settled RMS error {r['settled_rms']:.1f}°" +
                         (f", median 90% rise {r['rise90_median_s']:.2f} s over {r['steps']} steps" if "rise90_median_s" in r else ""),
                         fontsize=10)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return _png(fig)


def _with_gaps(df: pd.DataFrame, cols: list[str], gap_s: float = 1.0) -> pd.DataFrame:
    """Insert blank rows where logging pauses, so a line is not drawn across the gap."""
    t = df["t"].to_numpy()
    breaks = np.flatnonzero(np.diff(t) > gap_s)
    if not breaks.size:
        return df
    gaps = pd.DataFrame({"t": (t[breaks] + t[breaks + 1]) / 2, **{c: np.nan for c in cols}})
    return pd.concat([df[["t"] + cols], gaps]).sort_values("t")


def navigation_figure(log: FlightLog, s: dict) -> str | None:
    tecs, ntun, qtun = log.get("TECS"), log.get("NTUN"), log.get("QTUN")
    panels = []
    if not tecs.empty:
        panels.append("tecs")
    if not ntun.empty and "XT" in ntun:
        panels.append("xt")
    if not qtun.empty and "DAlt" in qtun:
        panels.append("qtun")
    if not panels:
        return None
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.6 * len(panels)), sharex=True, squeeze=False)
    for ax, p in zip(axes[:, 0], panels):
        if p == "tecs":
            ax.plot(tecs["t"], tecs["hdem"], lw=1, label="Altitude demand"); ax.plot(tecs["t"], tecs["h"], lw=1, label="Altitude")
            ax.set_ylabel("m"); ax.set_title("Fixed wing altitude tracking (TECS)", fontsize=10)
        elif p == "xt":
            ax.plot(ntun["t"], ntun["XT"], lw=1, color="#59a14f", label="Cross track error")
            ax.axhline(0, color="#999", lw=0.8); ax.set_ylabel("m"); ax.set_title("Cross track error (NTUN)", fontsize=10)
        else:
            q = _with_gaps(qtun, ["DAlt", "Alt"])
            ax.plot(q["t"], q["DAlt"], lw=1, label="VTOL altitude demand"); ax.plot(q["t"], q["Alt"], lw=1, label="VTOL altitude")
            ax.set_ylabel("m"); ax.set_title("VTOL altitude tracking (QTUN)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.tight_layout()
    return _png(fig)


def health_figure(log: FlightLog, s: dict) -> str | None:
    vibe = log.get("VIBE")
    ekf = log.get("XKF4") if log.has("XKF4") else log.get("NKF4")
    gps, bat = log.get("GPS"), log.get("BAT")
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4))
    a = axes[0, 0]
    if not vibe.empty:
        v = vibe[vibe["IMU"] == vibe["IMU"].min()] if "IMU" in vibe else vibe
        for c in ["VibeX", "VibeY", "VibeZ"]:
            a.plot(v["t"], v[c], lw=0.8, label=c[-1])
        a.axhline(30, color="#e15759", ls="--", lw=1, label="30 m/s² limit"); a.legend(fontsize=8)
    a.set_title("Vibration (m/s²)", fontsize=10)
    a = axes[0, 1]
    if not ekf.empty:
        e = ekf[ekf["C"] == ekf["C"].min()] if "C" in ekf else ekf
        for c, lab in [("SV", "velocity"), ("SP", "position"), ("SH", "height"), ("SM", "magnetometer")]:
            if c in e:
                a.plot(e["t"], e[c], lw=0.8, label=lab)
        a.axhline(1.0, color="#e15759", ls="--", lw=1, label="1.0 limit"); a.legend(fontsize=8)
    a.set_title("EKF innovation test ratios", fontsize=10)
    a = axes[1, 0]
    if not gps.empty:
        g = gps[gps["I"] == gps["I"].min()] if "I" in gps else gps
        a.plot(g["t"], g["NSats"], lw=1, label="satellites"); a.set_ylabel("satellites")
        a2 = a.twinx(); a2.plot(g["t"], g["HDop"], lw=1, color="#f28e2b", label="HDOP"); a2.set_ylabel("HDOP")
    a.set_title("GPS", fontsize=10)
    a = axes[1, 1]
    if not bat.empty:
        b = bat[bat["Inst"] == bat["Inst"].min()] if "Inst" in bat else bat
        a.plot(b["t"], b["Volt"], lw=1, label="voltage"); a.set_ylabel("V")
        if "Curr" in b:
            a2 = a.twinx(); a2.plot(b["t"], b["Curr"], lw=0.8, color="#e15759"); a2.set_ylabel("A")
    else:
        a.text(0.5, 0.5, "No battery data logged", ha="center", va="center", transform=a.transAxes, color="#777")
        a.set_xticks([]); a.set_yticks([])
    a.set_title("Battery voltage (blue) and current (red)", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    return _png(fig)


def track_figure(log: FlightLog, s: dict) -> str | None:
    pos = log.get("POS")
    if pos.empty:
        return None
    p = pos[(pos["Lat"].abs() > 1e-6) | (pos["Lng"].abs() > 1e-6)]
    if len(p) < 2:
        return None
    lat0 = float(p["Lat"].mean())
    x = (p["Lng"] - p["Lng"].iloc[0]) * 111320 * np.cos(np.radians(lat0))
    y = (p["Lat"] - p["Lat"].iloc[0]) * 110540
    segs = s["modes"]
    colors = _mode_colors(segs) if not segs.empty else {}
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if segs.empty:
        ax.plot(x, y, lw=1.4)
    else:
        for _, sg in segs.iterrows():
            m = (p["t"] >= sg["start"]) & (p["t"] <= sg["end"])
            ax.plot(x[m], y[m], lw=1.6, color=colors[sg["mode"]], label=sg["mode"])
    ax.plot(x.iloc[0], y.iloc[0], "o", color="#2e7d32", label="start")
    ax.plot(x.iloc[-1], y.iloc[-1], "s", color="#c62828", label="end")
    hl, lb = ax.get_legend_handles_labels()
    uniq = dict(zip(lb, hl))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")
    ax.set_aspect("equal"); ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("Ground track (coloured by mode)", fontsize=10)
    return _png(fig)


def write_track_geojson(log: FlightLog, path: str):
    pos = log.get("POS")
    p = pos[(pos["Lat"].abs() > 1e-6) | (pos["Lng"].abs() > 1e-6)] if not pos.empty else pos
    coords = [[round(float(a), 7), round(float(b), 7), round(float(c), 1)]
              for a, b, c in zip(p.get("Lng", []), p.get("Lat", []), p.get("Alt", []))][::5]
    gj = {"type": "FeatureCollection", "features": [] if len(coords) < 2 else [
        {"type": "Feature", "properties": {"name": os.path.basename(log.path)},
         "geometry": {"type": "LineString", "coordinates": coords}}]}
    with open(path, "w") as f:
        json.dump(gj, f)


def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p class='meta'>None.</p>"
    head = "".join(f"<th>{html.escape(h)}</th>" for _, h in cols)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(k, '')))}</td>" for k, _ in cols) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _badge(p):
    if p is None:
        return '<span class="info">INFO</span>'
    return '<span class="pass">PASS</span>' if p else '<span class="fail">FAIL</span>'


def _fmt(v, unit="", nd=1):
    return "" if v is None else f"{v:,.{nd}f}{unit}"


def autotune_html(s: dict) -> str:
    at = s.get("autotune") or {}
    if not at:
        return ""

    def v(d, k, unit, nd=2):
        return "" if d.get(k) is None else f"{d[k]:.{nd}f}{unit}"

    rows = []
    for axis in ("roll", "pitch"):
        b, a = at["axes"][axis]["before"], at["axes"][axis]["after"]
        rows.append({"metric": f"{axis.title()} settled tracking (RMS)", "before": v(b, "settled_rms", "°", 1), "after": v(a, "settled_rms", "°", 1)})
        if "rise90_median_s" in b or "rise90_median_s" in a:
            rows.append({"metric": f"{axis.title()} 90% rise time (median)", "before": v(b, "rise90_median_s", " s"),
                         "after": v(a, "rise90_median_s", " s")})
            rows.append({"metric": f"{axis.title()} overshoot (max)", "before": v(b, "overshoot_max_pct", "%", 1),
                         "after": v(a, "overshoot_max_pct", "%", 1)})
            rows.append({"metric": f"{axis.title()} steps measured", "before": b.get("steps", 0), "after": a.get("steps", 0)})
    return (f"<h2>AUTOTUNE test point: before and after</h2><p class='meta'>Tuning ran from {at['start_s']} s to {at['end_s']} s. "
            "Before and after are measured in fixed wing flight outside the tuning mode, on the same mission legs.</p>"
            "<div class='cols'><div>" + _table(rows, [("metric", "Metric"), ("before", "Before"), ("after", "After")]) +
            "</div><div>" + _table(at["gains"], [("gain", "Rate gain"), ("before", "Before"), ("after", "After")]) + "</div></div>")


def write_report(log: FlightLog, s: dict, out_dir: str, title: str | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(log.path)
    title = title or f"Flight log report: {name}"
    s["modes"].round(2).to_csv(os.path.join(out_dir, "modes.csv"), index=False)
    pd.DataFrame(s["events"]).to_csv(os.path.join(out_dir, "events.csv"), index=False)
    write_track_geojson(log, os.path.join(out_dir, "track.geojson"))

    summary = {k: v for k, v in s.items() if k != "modes"}
    summary["modes"] = s["modes"].round(2).to_dict(orient="records")
    summary["file"] = name
    summary["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    figs = [("Overview", overview_figure(log, s)), ("Attitude tracking", attitude_figure(log, s)),
            ("Navigation", navigation_figure(log, s)), ("Health", health_figure(log, s))]
    track = track_figure(log, s)

    info, sm = s["info"], s["summary"]
    facts = [("Firmware", info["firmware"] or "unknown"), ("Airframe", info["frame"]),
             ("Log length", _fmt(s["duration_s"], " s")), ("Armed", _fmt(sm.get("armed_time_s"), " s")),
             ("Airborne", _fmt(sm.get("flying_time_s"), " s")),
             ("Takeoff / landing", f"{sm['takeoff_s']} s / {sm['landing_s']} s" if "takeoff_s" in sm else ""),
             ("Distance flown", _fmt(sm.get("distance_flown_m"), " m", 0)),
             ("Max distance from start", _fmt(sm.get("max_distance_from_start_m"), " m", 0)),
             ("Max altitude above home", _fmt(sm.get("max_alt_above_home_m"), " m")),
             ("Max airspeed / ground speed", f"{_fmt(sm.get('max_airspeed_ms'), ' m/s')} / {_fmt(sm.get('max_ground_speed_ms'), ' m/s')}"),
             ("Battery", f"{_fmt(s['battery'].get('start_v'), ' V', 2)} to {_fmt(s['battery'].get('min_v'), ' V', 2)}, "
                         f"{_fmt(s['battery'].get('used_mah'), ' mAh', 0)} used" if s["battery"] else "")]
    facts_html = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in facts if v)
    checks_html = "".join(f"<tr><td>{html.escape(c['check'])}</td><td>{html.escape(str(c['value']))}</td>"
                          f"<td>{html.escape(c['target'])}</td><td>{_badge(c['pass'])}</td></tr>" for c in s["checks"])
    overall = all(c["pass"] for c in s["checks"] if c["pass"] is not None)
    modes_rows = [{"mode": r["mode"], "start": f"{r['start']:.1f}", "end": f"{r['end']:.1f}", "duration": f"{r['duration']:.1f}"}
                  for _, r in s["modes"].iterrows()]
    figs_html = "".join(f"<h2>{html.escape(t)}</h2><img src='data:image/png;base64,{b}' alt='{html.escape(t)}'>" for t, b in figs if b)
    limits = ", ".join(f"{k} = {v}" for k, v in s["limits"].items())

    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1120px;margin:24px auto;padding:0 16px;color:#1d2430}}
h1{{font-size:22px;margin-bottom:4px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:6px 0}} td,th{{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#f3f5f8;font-weight:600}} .pass{{color:#fff;background:#2e7d32;padding:2px 8px;border-radius:4px}}
.fail{{color:#fff;background:#c62828;padding:2px 8px;border-radius:4px}} .info{{color:#fff;background:#546e7a;padding:2px 8px;border-radius:4px}}
.meta{{color:#555;font-size:13px}} img{{max-width:100%}} .cols{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}}
@media (max-width:760px){{.cols{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>{html.escape(title)} {_badge(overall)}</h1>
<p class="meta">{html.escape(name)} · generated {summary['generated_utc']} UTC by flight-log-analyzer</p>
<div class="cols"><div><h2>Flight</h2><table>{facts_html}</table></div>
<div><h2>Ground track</h2>{f"<img src='data:image/png;base64,{track}' alt='Ground track'>" if track else "<p class='meta'>No position data.</p>"}</div></div>
<h2>Checks</h2><table><tr><th>Check</th><th>Result</th><th>Target</th><th>Status</th></tr>{checks_html}</table>
<p class="meta">Limits used: {html.escape(limits)}. Tracking is measured in fixed wing flight more than 5 m above home, outside tuning modes; VTOL phases are excluded.</p>
{autotune_html(s)}
{figs_html}
<div class="cols"><div><h2>Flight modes</h2>{_table(modes_rows, [("mode", "Mode"), ("start", "Start (s)"), ("end", "End (s)"), ("duration", "Duration (s)")])}</div>
<div><h2>Attitude tracking by mode</h2>{_table(s["attitude"], [("mode", "Mode"), ("roll_rms", "Roll RMS °"), ("roll_max", "Roll max °"), ("pitch_rms", "Pitch RMS °"), ("pitch_max", "Pitch max °")])}</div></div>
{"<h2>VTOL transitions</h2>" + _table(s["transitions"], [("start_s", "Start (s)"), ("to_airspeed_s", "To airspeed (s)"), ("duration_s", "To complete (s)"), ("alt_loss_m", "Alt loss (m)"), ("airspeed_end_ms", "Airspeed at end (m/s)")]) if s["info"]["quadplane"] else ""}
<div class="cols"><div><h2>Mission</h2>{_table(s["mission"], [("seq", "#"), ("command", "Command"), ("alt_m", "Alt (m)")])}</div>
<div><h2>Events and messages</h2>{_table(s["events"][:60], [("t", "Time (s)"), ("kind", "Type"), ("text", "Detail")])}</div></div>
<p class="meta">Outputs: summary.json, modes.csv, events.csv, track.geojson.</p>
</body></html>"""
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(page)
    return summary
