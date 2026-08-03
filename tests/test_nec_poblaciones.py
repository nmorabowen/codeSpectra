"""NEC-SE-DS Tabla 19: transcription integrity, name lookup, nearest-town rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeSpectra.codes.nec import (
    AmbiguousPoblacion,
    Gazetteer,
    PoblacionNotFound,
    Tabla19,
    nec_site_from_poblacion,
    region_for_provincia,
)
from codeSpectra.core import InvalidInput

#: Spot values verified against the printed Tabla 19.
KNOWN_Z = {
    "QUITO": 0.40,
    "GUAYAQUIL": 0.40,
    "CUENCA": 0.25,
    "ESMERALDAS": 0.50,
    "PORTOVIEJO": 0.50,
    "MANTA": 0.50,
    "AMBATO": 0.40,
    "MACHALA": 0.40,
    "LOJA": 0.25,
    "TENA": 0.35,
    "PUYO": 0.30,
    "NUEVA LOJA": 0.15,
}


@pytest.fixture(scope="module")
def table() -> Tabla19:
    return Tabla19.load()


@pytest.fixture
def gazetteer() -> Gazetteer:
    return Gazetteer({
        "Quito": (-0.1807, -78.4678),
        "Guayaquil": (-2.1709, -79.9224),
        "Cuenca": (-2.9001, -79.0059),
        "Esmeraldas": (0.9592, -79.6539),
    })


class TestTranscription:
    def test_row_count(self, table: Tabla19) -> None:
        assert len(table) == 515

    def test_every_Z_is_a_nec_zone_value(self, table: Tabla19) -> None:
        """Tabla 19 uses only the six discrete zone values of Tabla 1."""
        assert {p.Z for p in table} == {0.15, 0.25, 0.30, 0.35, 0.40, 0.50}

    def test_every_entry_maps_to_a_zone(self, table: Tabla19) -> None:
        assert all(p.zone is not None for p in table)

    def test_no_field_is_blank(self, table: Tabla19) -> None:
        for p in table:
            assert p.poblacion and p.parroquia and p.canton and p.provincia

    def test_province_count(self, table: Tabla19) -> None:
        # 23 provinces plus the 'ZONA NO DELIMITADA' category.
        assert len(table.provincias) == 24

    @pytest.mark.parametrize(("name", "Z"), sorted(KNOWN_Z.items()))
    def test_known_cities(self, table: Tabla19, name: str, Z: float) -> None:
        assert table.by_name(name).Z == pytest.approx(Z)

    def test_data_file_is_valid_json_with_a_source_note(self) -> None:
        import codeSpectra.codes.nec.poblaciones as mod

        path = Path(mod.__file__).parent / "tables" / "tabla19.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "Tabla 19" in payload["source"]
        assert len(payload["rows"]) == 515
        assert all(len(r) == 5 for r in payload["rows"])


class TestNameLookup:
    def test_is_case_insensitive(self, table: Tabla19) -> None:
        assert table.by_name("quito").Z == table.by_name("QUITO").Z

    def test_is_accent_insensitive(self, table: Tabla19) -> None:
        """Ecuadorian place names are written with and without accents."""
        assert table.by_name("CAÑAR").Z == table.by_name("CANAR").Z

    def test_ignores_parenthetical_aliases(self, table: Tabla19) -> None:
        entries = table.find("LA LIBERTAD")
        assert entries

    def test_unknown_name_raises_with_a_hint(self, table: Tabla19) -> None:
        with pytest.raises(PoblacionNotFound, match="not listed"):
            table.by_name("Springfield")

    def test_unknown_name_points_at_the_nearest_rule(self, table: Tabla19) -> None:
        with pytest.raises(PoblacionNotFound, match="nearest listed"):
            table.by_name("Springfield")

    def test_duplicate_names_raise(self, table: Tabla19) -> None:
        duplicated = next(
            p.poblacion for p in table if len(table.find(p.poblacion)) > 1
        )
        with pytest.raises(AmbiguousPoblacion):
            table.by_name(duplicated)

    def test_ambiguity_can_be_resolved_by_province(self, table: Tabla19) -> None:
        dup = next(p for p in table if len(table.find(p.poblacion)) > 1)
        resolved = table.by_name(
            dup.poblacion, provincia=dup.provincia, canton=dup.canton
        )
        assert resolved.provincia == dup.provincia

    def test_ambiguity_reports_whether_Z_actually_differs(
        self, table: Tabla19
    ) -> None:
        """A duplicate name is harmless when every candidate shares one Z."""
        dup = next(p.poblacion for p in table if len(table.find(p.poblacion)) > 1)
        with pytest.raises(AmbiguousPoblacion) as exc:
            table.by_name(dup)
        same = len({c.Z for c in exc.value.candidates}) == 1
        assert exc.value.same_Z is same

    def test_filters_that_match_nothing_raise(self, table: Tabla19) -> None:
        with pytest.raises(PoblacionNotFound, match="filters"):
            table.by_name("QUITO", provincia="GUAYAS")

    def test_search_returns_substring_matches(self, table: Tabla19) -> None:
        assert all("SAN" in p.key for p in table.search("SAN"))

    def test_by_provincia(self, table: Tabla19) -> None:
        pichincha = table.by_provincia("Pichincha")
        assert pichincha
        assert all(p.provincia == "PICHINCHA" for p in pichincha)


class TestRegionDerivation:
    @pytest.mark.parametrize(
        ("provincia", "region"),
        [
            ("GUAYAS", "costa"), ("MANABI", "costa"), ("EL ORO", "costa"),
            ("SANTA ELENA", "costa"), ("LOS RIOS", "costa"),
            ("PICHINCHA", "sierra"), ("AZUAY", "sierra"), ("TUNGURAHUA", "sierra"),
            ("CAÑAR", "sierra"), ("LOJA", "sierra"),
            ("NAPO", "oriente"), ("PASTAZA", "oriente"), ("SUCUMBIOS", "oriente"),
            ("ORELLANA", "oriente"), ("ZAMORA CHINCHIPE", "oriente"),
            ("MORONA SANTIAGO", "oriente"),
        ],
    )
    def test_province_groups(self, provincia: str, region: str) -> None:
        assert region_for_provincia(provincia) == region

    def test_esmeraldas_is_sierra_despite_being_coastal(self) -> None:
        """§3.3.1 names Esmeraldas with the Sierra group explicitly."""
        assert region_for_provincia("ESMERALDAS") == "sierra"

    def test_galapagos_is_sierra(self) -> None:
        assert region_for_provincia("GALAPAGOS") == "sierra"

    @pytest.mark.parametrize(
        "provincia", ["STO. DOMINGO DE LOS TSACHILAS", "ZONA NO DELIMITADA"]
    )
    def test_unassigned_provinces_raise(self, provincia: str) -> None:
        """The standard does not place these; guessing would be inventing."""
        with pytest.raises(InvalidInput, match="does not assign a region"):
            region_for_provincia(provincia)

    def test_unknown_province_raises(self) -> None:
        with pytest.raises(InvalidInput, match="Unknown Ecuadorian province"):
            region_for_provincia("BAVARIA")

    def test_every_table_province_resolves_or_is_documented(
        self, table: Tabla19
    ) -> None:
        """No province may fail silently or unexpectedly."""
        from codeSpectra.codes.nec.poblaciones import (
            AMBIGUOUS_PROVINCES,
            _province_key,
        )

        for provincia in table.provincias:
            if _province_key(provincia) in AMBIGUOUS_PROVINCES:
                continue
            assert region_for_provincia(provincia) in ("costa", "sierra", "oriente")


class TestGazetteer:
    def test_lookup_ignores_case_and_accents(self) -> None:
        g = Gazetteer({"Cuenca": (-2.9, -79.0)})
        assert g.get("CUENCA") == (-2.9, -79.0)
        assert "cuenca" in g

    def test_rejects_out_of_range_coordinates(self) -> None:
        with pytest.raises(InvalidInput, match="out of range"):
            Gazetteer({"Nowhere": (95.0, 0.0)})

    def test_rejects_empty(self) -> None:
        with pytest.raises(InvalidInput, match="at least one place"):
            Gazetteer({})

    def test_from_geojson_points(self) -> None:
        data = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"id": "Quito"},
                 "geometry": {"type": "MultiPoint",
                              "coordinates": [[-78.4678, -0.1807]]}},
                {"type": "Feature", "properties": {"id": "Manta"},
                 "geometry": {"type": "Point", "coordinates": [-80.7089, -0.9677]}},
            ],
        }
        g = Gazetteer.from_geojson_points(data)
        assert len(g) == 2
        assert g.get("Quito") == pytest.approx((-0.1807, -78.4678))

    def test_from_geojson_without_points_raises(self) -> None:
        with pytest.raises(InvalidInput, match="No point features"):
            Gazetteer.from_geojson_points({"type": "FeatureCollection",
                                           "features": []})

    def test_from_file_strips_js_wrapper(self, tmp_path: Path) -> None:
        data = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"id": "Quito"},
             "geometry": {"type": "Point", "coordinates": [-78.4678, -0.1807]}}]}
        path = tmp_path / "cities.js"
        path.write_text(f"var json_cities = {json.dumps(data)};", encoding="utf-8")
        assert len(Gazetteer.from_file(path)) == 1


class TestNearest:
    def test_finds_the_town_itself(self, table: Tabla19, gazetteer: Gazetteer) -> None:
        m = table.nearest(-0.1807, -78.4678, gazetteer)
        assert m.poblacion.poblacion == "QUITO"
        assert m.distance_km == pytest.approx(0.0, abs=0.5)

    def test_finds_the_nearest_of_several(
        self, table: Tabla19, gazetteer: Gazetteer
    ) -> None:
        m = table.nearest(-2.88, -79.02, gazetteer)   # just outside Cuenca
        assert m.poblacion.poblacion == "CUENCA"
        assert m.Z == pytest.approx(0.25)

    def test_distance_is_reported(
        self, table: Tabla19, gazetteer: Gazetteer
    ) -> None:
        m = table.nearest(-0.30, -78.50, gazetteer)
        assert 5.0 < m.distance_km < 40.0

    def test_max_distance_refuses_a_far_point(
        self, table: Tabla19, gazetteer: Gazetteer
    ) -> None:
        with pytest.raises(PoblacionNotFound, match="beyond the"):
            table.nearest(-2.0, -84.0, gazetteer, max_distance_km=100.0)

    def test_empty_coverage_raises(self, table: Tabla19) -> None:
        g = Gazetteer({"Atlantis": (0.0, -30.0)})
        with pytest.raises(PoblacionNotFound, match="covers none"):
            table.nearest(-0.18, -78.47, g)

    def test_covered_by_reports_overlap(
        self, table: Tabla19, gazetteer: Gazetteer
    ) -> None:
        assert len(table.covered_by(gazetteer)) == 4

    @pytest.mark.parametrize(("lat", "lon"), [(91.0, 0.0), (0.0, 181.0)])
    def test_invalid_coordinates_rejected(
        self, table: Tabla19, gazetteer: Gazetteer, lat: float, lon: float
    ) -> None:
        with pytest.raises(InvalidInput):
            table.nearest(lat, lon, gazetteer)

    def test_report_cites_both_clauses(
        self, table: Tabla19, gazetteer: Gazetteer
    ) -> None:
        report = table.nearest(-0.30, -78.50, gazetteer).report()
        refs = " ".join(str(r) for r in report.refs())
        assert "3.1.1" in refs and "10.2" in refs

    def test_report_warns_about_zone_boundaries(
        self, table: Tabla19, gazetteer: Gazetteer
    ) -> None:
        notes = table.nearest(-0.30, -78.50, gazetteer).report().notes
        assert any("Figura 1" in n for n in notes)


class TestSiteConstruction:
    def test_builds_a_site_with_derived_region(self, table: Tabla19) -> None:
        site = nec_site_from_poblacion(table.by_name("Quito"), soil="D")
        assert site.Z_g == pytest.approx(0.40)
        assert site.eta == pytest.approx(2.48)      # sierra

    def test_costa_province_gives_costa_eta(self, table: Tabla19) -> None:
        site = nec_site_from_poblacion(table.by_name("Guayaquil"))
        assert site.eta == pytest.approx(1.80)

    def test_oriente_province_gives_oriente_eta(self, table: Tabla19) -> None:
        site = nec_site_from_poblacion(table.by_name("Nueva Loja"))
        assert site.eta == pytest.approx(2.60)

    def test_esmeraldas_gives_sierra_eta(self, table: Tabla19) -> None:
        site = nec_site_from_poblacion(table.by_name("Esmeraldas"))
        assert site.eta == pytest.approx(2.48)

    def test_explicit_region_overrides_derivation(self, table: Tabla19) -> None:
        site = nec_site_from_poblacion(table.by_name("Quito"), region="costa")
        assert site.eta == pytest.approx(1.80)

    def test_unassigned_province_requires_explicit_region(
        self, table: Tabla19
    ) -> None:
        sto = table.by_provincia("STO. DOMINGO DE LOS TSACHILAS")[0]
        with pytest.raises(InvalidInput, match="does not assign a region"):
            nec_site_from_poblacion(sto)
        assert nec_site_from_poblacion(sto, region="costa").eta == pytest.approx(1.80)

    def test_provenance_note_cites_the_table(self, table: Tabla19) -> None:
        site = nec_site_from_poblacion(table.by_name("Quito"))
        assert "Tabla 19" in site.report().notes[0]

    def test_note_records_a_nearest_match(
        self, table: Tabla19, gazetteer: Gazetteer
    ) -> None:
        site = nec_site_from_poblacion(table.nearest(-0.30, -78.50, gazetteer))
        note = site.report().notes[0]
        assert "nearest listed poblacion" in note
        assert "Figura 1" in note

    def test_spectrum_builds(self, table: Tabla19) -> None:
        site = nec_site_from_poblacion(table.by_name("Quito"))
        assert site.elastic_spectrum().at(0.5) == pytest.approx(1.1904, rel=1e-4)

    def test_no_interpolation_note_for_tabulated_Z(self, table: Tabla19) -> None:
        """Tabla 19 Z values are all zone values, so Fa/Fd/Fs are exact."""
        site = nec_site_from_poblacion(table.by_name("Quito"))
        assert not any("interpolation" in n for n in site.report().notes)

    def test_report_is_cp1252_safe(self, table: Tabla19) -> None:
        for name in ("Quito", "Guayaquil", "Cuenca", "Nueva Loja"):
            nec_site_from_poblacion(table.by_name(name)).report().to_text().encode(
                "cp1252"
            )


class TestGeoNamesGazetteer:
    """The bundled GeoNames coordinates (CC BY 4.0)."""

    @staticmethod
    @pytest.fixture(scope="class")
    def geo() -> Gazetteer:
        return Gazetteer.geonames()

    def test_loads(self, geo: Gazetteer) -> None:
        assert len(geo) > 400

    def test_coverage_of_the_table(self, table: Tabla19, geo: Gazetteer) -> None:
        covered = table.covered_by(geo)
        assert len(covered) / len(table) > 0.85

    def test_attribution_is_exposed(self) -> None:
        from codeSpectra.codes.nec.poblaciones import GEONAMES_ATTRIBUTION

        assert "GeoNames" in GEONAMES_ATTRIBUTION
        assert "CC BY 4.0" in GEONAMES_ATTRIBUTION

    def test_data_file_records_its_licence(self) -> None:
        import codeSpectra.codes.nec.poblaciones as mod

        path = Path(mod.__file__).parent / "tables" / "gazetteer_geonames.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "CC BY 4.0" in payload["licence"]
        assert "geonames.org" in payload["source"]

    @pytest.mark.parametrize(
        ("name", "provincia", "lat", "lon"),
        [
            ("QUITO", "PICHINCHA", -0.22, -78.51),
            ("GUAYAQUIL", "GUAYAS", -2.19, -79.89),
            ("CUENCA", "AZUAY", -2.90, -79.00),
            ("MANTA", "MANABI", -0.95, -80.73),
            ("NUEVA LOJA", "SUCUMBIOS", 0.09, -76.89),
        ],
    )
    def test_known_coordinates(
        self, geo: Gazetteer, name: str, provincia: str, lat: float, lon: float
    ) -> None:
        point = geo.get(name, provincia)
        assert point is not None
        assert point[0] == pytest.approx(lat, abs=0.1)
        assert point[1] == pytest.approx(lon, abs=0.1)

    def test_every_coordinate_is_inside_ecuador(self, geo: Gazetteer) -> None:
        table = Tabla19.load()
        for row in table.covered_by(geo):
            lat, lon = geo.get(row.poblacion, row.provincia)  # type: ignore[misc]
            assert -5.1 <= lat <= 1.6, f"{row.poblacion} latitude {lat}"
            assert -81.2 <= lon <= -75.1, f"{row.poblacion} longitude {lon}"

    def test_nearest_finds_a_plausible_town(
        self, table: Tabla19, geo: Gazetteer
    ) -> None:
        m = table.nearest(-2.90, -79.00, geo)          # Cuenca
        assert m.poblacion.provincia == "AZUAY"
        assert m.distance_km < 25.0

    def test_nearest_across_the_country(
        self, table: Tabla19, geo: Gazetteer
    ) -> None:
        """Anywhere in continental Ecuador should find a town within ~60 km."""
        for lat, lon in [(-0.22, -78.51), (-2.19, -79.89), (0.96, -79.65),
                         (-3.99, -79.20), (-1.05, -80.45), (0.09, -76.89)]:
            assert table.nearest(lat, lon, geo).distance_km < 60.0


class TestProvinceQualification:
    """Twenty Tabla 19 names occur in several provinces with different Z."""

    def test_qualified_entries_do_not_leak_across_provinces(self) -> None:
        g = Gazetteer({
            ("Bolivar", "CARCHI"): (0.50, -77.90),
            ("Bolivar", "ESMERALDAS"): (0.83, -78.20),
        })
        assert g.get("Bolivar", "CARCHI") == (0.50, -77.90)
        assert g.get("Bolivar", "ESMERALDAS") == (0.83, -78.20)

    def test_unknown_province_does_not_fall_back_to_another(self) -> None:
        g = Gazetteer({("Bolivar", "CARCHI"): (0.50, -77.90)})
        assert g.get("Bolivar", "MANABI") is None

    def test_ambiguous_unqualified_lookup_returns_nothing(self) -> None:
        g = Gazetteer({
            ("Bolivar", "CARCHI"): (0.50, -77.90),
            ("Bolivar", "ESMERALDAS"): (0.83, -78.20),
        })
        assert g.get("Bolivar") is None

    def test_unqualified_entries_still_match_a_province_query(self) -> None:
        """Supplying a plain dict is an explicit choice to ignore provinces."""
        g = Gazetteer({"Quito": (-0.18, -78.47)})
        assert g.get("Quito", "PICHINCHA") == (-0.18, -78.47)

    def test_qualified_key_must_be_a_pair(self) -> None:
        with pytest.raises(InvalidInput, match=r"must be \(name, provincia\)"):
            Gazetteer({("a", "b", "c"): (0.0, -78.0)})  # type: ignore[dict-item]

    def test_duplicated_names_resolve_to_their_own_Z(self, table: Tabla19) -> None:
        """The bug a name-only gazetteer would cause, pinned down."""
        geo = Gazetteer.geonames()
        duplicated = [
            p for p in table
            if len({q.provincia for q in table.find(p.poblacion)}) > 1
            and len({q.Z for q in table.find(p.poblacion)}) > 1
        ]
        assert duplicated, "expected Tabla 19 to contain such names"
        for row in duplicated:
            point = geo.get(row.poblacion, row.provincia)
            if point is None:
                continue
            match = table.nearest(point[0], point[1], geo)
            # The nearest town to a town's own coordinates carries its own Z.
            assert match.Z == pytest.approx(row.Z), (
                f"{row.poblacion} ({row.provincia}) resolved to "
                f"{match.poblacion.poblacion} ({match.poblacion.provincia})"
            )


class TestGlyphRepair:
    """The NEC PDF font mis-encodes three accented capitals."""

    def test_no_stray_glyphs_remain(self, table: Tabla19) -> None:
        import unicodedata

        for p in table:
            for field in (p.poblacion, p.parroquia, p.canton, p.provincia):
                for ch in field:
                    if ch.isascii():
                        continue
                    assert unicodedata.category(ch) == "Lu", f"{ch!r} in {field!r}"
                    assert ch in "ÑÁÉÍÓÚ", f"unexpected {ch!r} in {field!r}"

    def test_repaired_names_are_present(self, table: Tabla19) -> None:
        assert table.find("BAÑOS DE AGUA SANTA")
        assert table.find("AMAGUAÑA")
        assert table.find("SUCÚA")


class TestProvinceAbbreviation:
    """Tabla 19 prints "STO. DOMINGO"; a caller writes it out in full.

    Both must reach the same entry. Before the fix the written-out spelling
    fell through to "Unknown Ecuadorian province", which reads as a typo
    rather than as the standard declining to assign a region.
    """

    @pytest.mark.parametrize(
        "spelling",
        [
            "STO. DOMINGO DE LOS TSACHILAS",      # as Tabla 19 prints it
            "Santo Domingo de los Tsachilas",
            "Santo Domingo de los Tsáchilas",     # with the accent
            "santo domingo de los tsachilas",
        ],
    )
    def test_every_spelling_gives_the_explanatory_error(self, spelling: str) -> None:
        with pytest.raises(InvalidInput, match="does not assign a region"):
            region_for_provincia(spelling)

    def test_a_genuinely_unknown_province_still_says_unknown(self) -> None:
        with pytest.raises(InvalidInput, match="Unknown Ecuadorian province"):
            region_for_provincia("Santo Domingo de Guzman")

    def test_santa_elena_is_unaffected(self) -> None:
        """The STA -> SANTA expansion must not disturb a real province."""
        assert region_for_provincia("SANTA ELENA") == "costa"
        assert region_for_provincia("STA. ELENA") == "costa"


class TestLaConcordiaConflict:
    """Canton La Concordia moved province in 2013; Tabla 19 predates that.

    Reproduced as printed, because the code is what governs. Pinned so the
    disagreement is visible rather than looking like a transcription slip.
    """

    def test_still_listed_under_esmeraldas(self, table: Tabla19) -> None:
        entries = [p for p in table if p.canton == "LA CONCORDIA"]
        assert entries, "expected La Concordia entries in Tabla 19"
        assert {p.provincia for p in entries} == {"ESMERALDAS"}

    def test_the_two_provinces_would_give_different_outcomes(self) -> None:
        """Why the conflict matters rather than being cosmetic."""
        assert region_for_provincia("ESMERALDAS") == "sierra"
        with pytest.raises(InvalidInput, match="does not assign a region"):
            region_for_provincia("Santo Domingo de los Tsachilas")
