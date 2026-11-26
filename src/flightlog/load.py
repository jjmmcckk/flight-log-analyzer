"""Read an ArduPilot DataFlash (.BIN) log into pandas DataFrames, one per message type."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
from pymavlink import DFReader

# Message types the analysis uses. Reading only these keeps large logs fast.
TYPES = [
    "MSG", "MODE", "EV", "ERR", "STAT", "ARM", "PARM", "CMD",
    "ATT", "CTUN", "NTUN", "TECS", "QTUN", "POS", "GPS", "ARSP",
    "BAT", "VIBE", "XKF4", "NKF4", "RCOU", "PIDR", "PIDP",
]


@dataclass
class FlightLog:
    path: str
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)          # latest value of each parameter
    params_initial: dict[str, float] = field(default_factory=dict)  # value when logging started
    messages: list[tuple[float, str]] = field(default_factory=list)

    def get(self, name: str) -> pd.DataFrame:
        return self.frames.get(name, pd.DataFrame())

    def has(self, name: str, column: str | None = None) -> bool:
        df = self.frames.get(name)
        return df is not None and len(df) > 0 and (column is None or column in df.columns)


def load(path: str, types: list[str] | None = None) -> FlightLog:
    """Parse the log (.BIN, or .BIN.gz). Each frame gets a `t` column in seconds from the first timestamp."""
    if path.endswith(".gz"):
        import gzip
        import shutil
        import tempfile
        with gzip.open(path, "rb") as src, tempfile.NamedTemporaryFile(suffix=".BIN", delete=False) as dst:
            shutil.copyfileobj(src, dst)
        log = load(dst.name, types)
        os.unlink(dst.name)
        log.path = path
        return log
    wanted = set(types or TYPES)
    reader = DFReader.DFReader_binary(path, zero_time_base=False)
    rows: dict[str, list[dict]] = {}
    params: dict[str, float] = {}
    params_initial: dict[str, float] = {}
    messages: list[tuple[int, str]] = []
    while True:
        m = reader.recv_match(type=list(wanted))
        if m is None:
            break
        name = m.get_type()
        d = m.to_dict()
        d.pop("mavpackettype", None)
        if name == "PARM":
            params[d["Name"]] = d["Value"]
            params_initial.setdefault(d["Name"], d["Value"])
            continue
        if name == "MSG":
            messages.append((d.get("TimeUS", 0), d["Message"]))
        rows.setdefault(name, []).append(d)

    t0 = None
    for r in rows.values():
        if r and "TimeUS" in r[0]:
            first = r[0]["TimeUS"]
            t0 = first if t0 is None else min(t0, first)
    t0 = t0 or 0

    frames = {}
    for name, r in rows.items():
        df = pd.DataFrame(r)
        if "TimeUS" in df:
            df["t"] = (df["TimeUS"] - t0) / 1e6
        frames[name] = df
    msgs = [((tu - t0) / 1e6, text) for tu, text in messages]
    return FlightLog(path=path, frames=frames, params=params, params_initial=params_initial, messages=msgs)
