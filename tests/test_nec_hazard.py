"""External hazard-source loaders for Ecuador.

Every test here runs against a synthetic contour set. Nothing touches the
network, and no third-party data is vendored into the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeSpectra.codes.nec import (
    PALACIOS_2023,
    ContourHazardMap,
    HazardSource,
    HazardSourceError,
    nec_site_from_hazard,
)
from codeSpectra.core import InvalidInput

TEST_SOURCE = HazardSource(
    name="Synthetic test map",
    authors="Test",
    year=2000,
    hazard_level="475-year return period",
    url="https://example.invalid",
    licence="test fixture",
    caveat="not a real hazard model",
)


def _line(value: float, lon: float) -> dict[str, object]:
    """A meridian contour at constant longitude, spanning 1 deg either side."""
    return {
        "type": "Feature",
        "properties": {"elev": value},
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [[[lon, -1.0], [lon, 0.0], [lon, 1.0]]],
        },
    }


def _collection(*features: dict[str, object]) -> dict[str, object]:
    return {"type": "FeatureCollection", "features": list(features)}


@pytest.fixture
def hazard_map() -> ContourHazardMap:
    """Parallel meridian contours: 0.2 g at lon -79, 0.3 g at lon -78."""
    return ContourHazardMap.from_geojson(
        _collection(_line(0.2, -79.0), _line(0.3, -78.0)),
        source=TEST_SOURCE,
    )


class TestLoading:
    def test_reads_a_feature_collection(self, hazard_map: ContourHazardMap) -> None:
        assert list(hazard_map.contour_values) == [0.2, 0.3]

    def test_contour_interval_is_free_of_float_noise(self) -> None:
        m = ContourHazardMap.from_geojson(
            _collection(_line(0.1, -80.0), _line(0.2, -79.0), _line(0.3, -78.0)),
            source=TEST_SOURCE,
        )
        assert m.contour_interval == 0.1

    def test_bounds(self, hazard_map: ContourHazardMap) -> None:
        lat_min, lat_max, lon_min, lon_max = hazard_map.bounds
        assert (lat_min, lat_max) == (-1.0, 1.0)
        assert (lon_min, lon_max) == (-79.0, -78.0)

    def test_strips_the_qgis2web_js_wrapper(self, tmp_path: Path) -> None:
        payload = json.dumps(_collection(_line(0.2, -79.0), _line(0.3, -78.0)))
        path = tmp_path / "layer.js"
        path.write_text(f"var json_layer_1 = {payload};\n", encoding="utf-8")
        m = ContourHazardMap.from_file(path, source=TEST_SOURCE)
        assert list(m.contour_values) == [0.2, 0.3]

    def test_reads_plain_geojson(self, tmp_path: Path) -> None:
        path = tmp_path / "layer.geojson"
        path.write_text(
            json.dumps(_collection(_line(0.2, -79.0), _line(0.3, -78.0))),
            encoding="utf-8",
        )
        assert ContourHazardMap.from_file(path, source=TEST_SOURCE).contour_interval

    def test_linestring_as_well_as_multilinestring(self) -> None:
        feature = {
            "type": "Feature",
            "properties": {"elev": 0.4},
            "geometry": {"type": "LineString",
                         "coordinates": [[-78.0, -1.0], [-78.0, 1.0]]},
        }
        m = ContourHazardMap.from_geojson(
            _collection(feature, _line(0.3, -79.0)), source=TEST_SOURCE
        )
        assert list(m.contour_values) == [0.3, 0.4]

    def test_features_without_a_numeric_value_are_skipped(self) -> None:
        junk = {"type": "Feature", "properties": {"id": "x"},
                "geometry": {"type": "LineString",
                             "coordinates": [[-78.0, 0.0], [-77.0, 0.0]]}}
        m = ContourHazardMap.from_geojson(
            _collection(junk, _line(0.2, -79.0), _line(0.3, -78.0)),
            source=TEST_SOURCE,
        )
        assert list(m.contour_values) == [0.2, 0.3]

    def test_empty_collection_raises(self) -> None:
        with pytest.raises(HazardSourceError, match="No contour lines"):
            ContourHazardMap.from_geojson(_collection(), source=TEST_SOURCE)

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.js"
        path.write_text("var x = {not json;", encoding="utf-8")
        with pytest.raises(HazardSourceError, match="Could not parse"):
            ContourHazardMap.from_file(path, source=TEST_SOURCE)


class TestQuery:
    def test_on_a_contour_returns_its_value(self, hazard_map: ContourHazardMap) -> None:
        assert hazard_map.pga_at(0.0, -79.0).pga == pytest.approx(0.2, abs=1e-6)

    def test_midway_between_contours(self, hazard_map: ContourHazardMap) -> None:
        assert hazard_map.pga_at(0.0, -78.5).pga == pytest.approx(0.25, abs=1e-6)

    def test_interpolates_linearly_between_parallel_contours(
        self, hazard_map: ContourHazardMap
    ) -> None:
        """One fifth of the way from the 0.2 g line to the 0.3 g line."""
        assert hazard_map.pga_at(0.0, -78.8).pga == pytest.approx(0.22, abs=1e-6)

    def test_reports_the_bracketing_band(self, hazard_map: ContourHazardMap) -> None:
        assert hazard_map.pga_at(0.0, -78.5).band == (0.2, 0.3)

    def test_distance_to_nearest_contour(self, hazard_map: ContourHazardMap) -> None:
        # 0.1 deg of longitude at the equator is about 11.1 km.
        assert hazard_map.pga_at(0.0, -78.9).distance_km == pytest.approx(11.1, rel=0.02)

    def test_result_is_clamped_to_the_contour_range(
        self, hazard_map: ContourHazardMap
    ) -> None:
        est = hazard_map.pga_at(0.0, -77.0)
        assert 0.2 <= est.pga <= 0.3

    def test_point_far_outside_is_flagged_unreliable(
        self, hazard_map: ContourHazardMap
    ) -> None:
        """A confident-looking number from distant geometry is the trap here."""
        est = hazard_map.pga_at(-0.74, -90.31)   # Galapagos, ~1000 km away
        assert est.outside_coverage
        assert not est.reliable
        assert est.distance_km > 500.0

    def test_outside_coverage_note_is_emphatic(
        self, hazard_map: ContourHazardMap
    ) -> None:
        notes = hazard_map.pga_at(-0.74, -90.31).report().notes
        assert any("OUTSIDE THE DATA FOOTPRINT" in n for n in notes)
        assert any("Do not use it" in n for n in notes)

    def test_single_contour_value_is_unreliable(self) -> None:
        m = ContourHazardMap.from_geojson(
            _collection(_line(0.3, -78.0)), source=TEST_SOURCE
        )
        assert not m.pga_at(0.0, -78.5).reliable

    @pytest.mark.parametrize(("lat", "lon"), [(91.0, 0.0), (0.0, 200.0)])
    def test_invalid_coordinates_rejected(
        self, hazard_map: ContourHazardMap, lat: float, lon: float
    ) -> None:
        with pytest.raises(InvalidInput):
            hazard_map.pga_at(lat, lon)


class TestAttribution:
    def test_palacios_citation_names_the_authors_and_url(self) -> None:
        citation = PALACIOS_2023.citation()
        assert "Palacios" in citation and "Celi" in citation and "Poveda" in citation
        assert "github.com/ppalacios92" in citation

    def test_palacios_records_the_hazard_level(self) -> None:
        assert "475" in PALACIOS_2023.hazard_level

    def test_palacios_records_the_licence_position(self) -> None:
        assert "No licence file" in PALACIOS_2023.licence

    def test_palacios_caveat_defers_to_the_code_map(self) -> None:
        assert "NEC-SE-DS" in PALACIOS_2023.caveat

    def test_report_carries_the_citation(self, hazard_map: ContourHazardMap) -> None:
        notes = hazard_map.pga_at(0.0, -78.5).report().notes
        assert any("Source:" in n for n in notes)
        assert any("Licence:" in n for n in notes)

    def test_no_data_is_bundled(self) -> None:
        """The package must ship no third-party hazard layers."""
        import codeSpectra

        root = Path(codeSpectra.__file__).parent
        assert not list(root.rglob("*.geojson"))
        assert not list(root.rglob("*Curvas*"))

    def test_refuses_to_fetch_without_explicit_opt_in(self) -> None:
        with pytest.raises(HazardSourceError, match="does not bundle"):
            ContourHazardMap.from_palacios_2023()

    def test_refusal_message_carries_the_citation(self) -> None:
        with pytest.raises(HazardSourceError, match="Palacios"):
            ContourHazardMap.from_palacios_2023()


class TestNECIntegration:
    def test_builds_a_site_with_Z_from_the_estimate(
        self, hazard_map: ContourHazardMap
    ) -> None:
        site = nec_site_from_hazard(
            hazard_map.pga_at(0.0, -78.5), soil="D", region="sierra"
        )
        assert site.Z_g == pytest.approx(0.25)
        assert site.Fa > 0.0

    def test_report_leads_with_the_provenance_warning(
        self, hazard_map: ContourHazardMap
    ) -> None:
        site = nec_site_from_hazard(hazard_map.pga_at(0.0, -78.5), region="sierra")
        notes = site.report().notes
        assert "NOT from the" in notes[0]
        assert "NEC-SE-DS zone map" in notes[0]

    def test_spectrum_still_builds(self, hazard_map: ContourHazardMap) -> None:
        site = nec_site_from_hazard(hazard_map.pga_at(0.0, -78.5), region="costa")
        assert site.elastic_spectrum().at(0.5) > 0.0

    def test_refuses_an_unreliable_estimate(
        self, hazard_map: ContourHazardMap
    ) -> None:
        with pytest.raises(HazardSourceError, match="not reliable"):
            nec_site_from_hazard(hazard_map.pga_at(-0.74, -90.31))

    def test_unreliable_can_be_forced(self, hazard_map: ContourHazardMap) -> None:
        site = nec_site_from_hazard(
            hazard_map.pga_at(-0.74, -90.31), allow_unreliable=True
        )
        assert site.Z_g > 0.0

    def test_region_is_not_guessed_from_coordinates(
        self, hazard_map: ContourHazardMap
    ) -> None:
        """eta follows provincial boundaries a PGA map cannot resolve."""
        est = hazard_map.pga_at(0.0, -78.5)
        costa = nec_site_from_hazard(est, region="costa")
        sierra = nec_site_from_hazard(est, region="sierra")
        assert costa.eta != sierra.eta

    def test_report_is_cp1252_safe(self, hazard_map: ContourHazardMap) -> None:
        site = nec_site_from_hazard(hazard_map.pga_at(0.0, -78.5), region="sierra")
        site.report().to_text().encode("cp1252")
        hazard_map.pga_at(0.0, -78.5).report().to_text().encode("cp1252")
