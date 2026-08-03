"""Export writers and cross-code comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from codeSpectra import ASCE7_16, NECSEDS2015
from codeSpectra.export import to_csv, to_etabs, to_json, to_opensees, to_sap2000


@pytest.fixture
def spectrum():  # type: ignore[no-untyped-def]
    return ASCE7_16(Ss=1.5, S1=0.6, site_class="D", TL=8.0).design_spectrum()


def _periods(path: Path) -> list[float]:
    """Period column of a written CSV."""
    return [
        float(line.split(",")[0])
        for line in path.read_text(encoding="utf-8").splitlines()[1:]
    ]


class TestCsv:
    def test_writes_header_and_rows(self, spectrum, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = to_csv(spectrum, tmp_path / "s.csv", n=20)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "T [s],Sa [g]"
        assert len(lines) > 20

    def test_unit_conversion(self, spectrum, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = to_csv(spectrum, tmp_path / "s.csv", periods=[1.0], unit="m/s2")
        value = float(path.read_text(encoding="utf-8").splitlines()[1].split(",")[1])
        assert value == pytest.approx(0.68 * 9.80665, rel=1e-4)

    def test_grid_includes_control_periods(self, spectrum, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """The exported curve must reproduce the corners of Fig. 11.4-1."""
        path = to_csv(spectrum, tmp_path / "s.csv", n=50)
        periods = _periods(path)
        assert any(abs(t - 0.68) < 1e-6 for t in periods)   # Ts
        assert any(abs(t - 0.136) < 1e-6 for t in periods)  # T0

    def test_no_duplicate_periods(self, spectrum, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """Corner-straddling points must not collapse into duplicate rows.

        `Spectrum.grid` inserts points a hair either side of each control
        period; at six decimal places those round together, which would be an
        invalid abscissa sequence in every target format.
        """
        periods = _periods(to_csv(spectrum, tmp_path / "s.csv", n=50))
        assert len(periods) == len(set(periods))
        assert periods == sorted(periods)


class TestJson:
    def test_carries_provenance(self, spectrum, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = to_json(spectrum, tmp_path / "s.json", n=10)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["standard"] == "ASCE/SEI 7"
        assert payload["edition"] == "7-16"
        assert payload["parameters"]["SDS"] == pytest.approx(1.0)
        assert payload["control_periods"]["Ts"] == pytest.approx(0.68)
        assert any("11.4" in r for r in payload["references"])

    def test_round_trips_ordinates(self, spectrum, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = to_json(spectrum, tmp_path / "s.json", periods=[0.0, 1.0, 4.0])
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["Sa"] == pytest.approx([0.4, 0.68, 0.17], abs=1e-6)


class TestEtabsAndSap:
    def test_etabs_two_columns_with_comments(self, spectrum, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = to_etabs(spectrum, tmp_path / "s.txt", n=10)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0].startswith("#")
        data = [ln for ln in lines if not ln.startswith("#")]
        assert all(len(ln.split("\t")) == 2 for ln in data)

    def test_sap_matches_etabs_layout(self, spectrum, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        a = to_etabs(spectrum, tmp_path / "a.txt", n=10).read_text(encoding="utf-8")
        b = to_sap2000(spectrum, tmp_path / "b.fnc", n=10).read_text(encoding="utf-8")
        assert a == b


class TestOpenSees:
    def test_tcl_emits_path_series(self, spectrum, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        text = to_opensees(
            spectrum, tmp_path / "s.tcl", n=10, series_tag=7
        ).read_text(encoding="utf-8")
        assert "timeSeries Path 7" in text
        assert "-time" in text and "-values" in text

    def test_python_style(self, spectrum, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        text = to_opensees(
            spectrum, tmp_path / "s.py", n=10, style="python"
        ).read_text(encoding="utf-8")
        assert "ops.timeSeries('Path'" in text

    def test_unknown_style_rejected(self, spectrum, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="style must be"):
            to_opensees(spectrum, tmp_path / "s.txt", style="fortran")


class TestCrossCode:
    def test_same_site_under_two_codes(self) -> None:
        """The shared Spectrum interface lets codes be compared directly."""
        asce = ASCE7_16(Ss=1.5, S1=0.6, site_class="D").design_spectrum()
        nec = NECSEDS2015(zone="V", soil="D", region="sierra").elastic_spectrum()
        T = np.linspace(0.05, 3.0, 50)
        assert asce.at(T).shape == nec.at(T).shape

    def test_envelope_across_codes(self) -> None:
        asce = ASCE7_16(Ss=1.5, S1=0.6, site_class="D").design_spectrum()
        nec = NECSEDS2015(zone="V", soil="D", region="sierra").elastic_spectrum()
        env = asce.envelope(nec)
        for T in (0.2, 1.0, 2.0):
            assert env.at(T) == pytest.approx(max(asce.at(T), nec.at(T)))

    def test_envelope_merges_control_periods(self) -> None:
        asce = ASCE7_16(Ss=1.5, S1=0.6, site_class="D").design_spectrum()
        nec = NECSEDS2015(zone="V", soil="D", region="sierra").elastic_spectrum()
        merged = asce.envelope(nec).control_periods
        assert "Ts" in merged and "Tc" in merged
