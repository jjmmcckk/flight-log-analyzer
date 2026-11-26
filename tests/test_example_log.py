"""End to end test on the committed SITL flight log (QuadPlane FT 001 with an AUTOTUNE test point)."""
import json
from pathlib import Path

import pytest

from flightlog import analyse, load
from flightlog.cli import main

LOG = Path(__file__).resolve().parents[1] / "examples" / "logs" / "ft001_quadplane_autotune_sitl.BIN.gz"
CARD = Path(__file__).resolve().parents[1] / "examples" / "ft001_testcard.json"


@pytest.fixture(scope="module")
def result():
    return analyse(load(str(LOG)))


def test_identifies_quadplane(result):
    assert result["info"]["vehicle"] == "ArduPlane"
    assert result["info"]["quadplane"] is True


def test_one_forward_transition_without_altitude_loss(result):
    assert len(result["transitions"]) == 1
    tr = result["transitions"][0]
    assert tr["duration_s"] < 15
    assert tr["alt_loss_m"] <= 5


def test_autotune_speeds_up_roll_response(result):
    roll = result["autotune"]["axes"]["roll"]
    assert roll["after"]["rise90_median_s"] < roll["before"]["rise90_median_s"]
    assert roll["after"]["settled_rms"] < roll["before"]["settled_rms"]


def test_health_checks_pass(result):
    checks = {c["check"]: c for c in result["checks"]}
    for name in ["GPS satellites (minimum)", "EKF innovation test ratios (max)", "Vibration (max axis)", "Error events (ERR)"]:
        assert checks[name]["pass"] is True


def test_cli_writes_report(tmp_path):
    main([str(LOG), "--limits", str(CARD), "--out", str(tmp_path)])
    for name in ["report.html", "summary.json", "modes.csv", "events.csv", "track.geojson"]:
        assert (tmp_path / name).exists(), name
    assert json.loads((tmp_path / "summary.json").read_text())["info"]["frame"] == "QuadPlane (VTOL)"
