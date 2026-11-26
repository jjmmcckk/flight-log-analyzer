"""Command line interface: flightlog-report flight.BIN --out report/ [--limits testcard.json]"""
from __future__ import annotations

import argparse
import json
import sys

from .analyze import analyse
from .load import load
from .report import write_report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flightlog-report",
                                description="Analyse an ArduPilot DataFlash (.BIN) log and write a flight test report.")
    p.add_argument("log", help="ArduPilot DataFlash log (.BIN)")
    p.add_argument("--out", default="flight_report", help="Output folder (default: flight_report)")
    p.add_argument("--limits", help="Test card JSON with pass or fail limits (see examples/)")
    p.add_argument("--title", help="Report title, e.g. 'FT 001 QuadPlane transition'")
    a = p.parse_args(argv)

    limits, title = {}, a.title
    if a.limits:
        with open(a.limits) as f:
            card = json.load(f)
        limits = card.get("limits", card)
        title = title or card.get("title")

    log = load(a.log)
    s = analyse(log, limits)
    write_report(log, s, a.out, title)

    for c in s["checks"]:
        status = "INFO" if c["pass"] is None else ("PASS" if c["pass"] else "FAIL")
        print(f"[{status}] {c['check']}: {c['value']} (target {c['target']})")
    print(f"Report written to {a.out}/report.html")
    return 1 if any(c["pass"] is False for c in s["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
