"""NEC-SE-DS 2015 (Ecuador): tables, spectrum shape, and base shear."""

from __future__ import annotations

import numpy as np
import pytest

from codeSpectra.codes.nec import NECSEDS2015, Region, SeismicZone, elf
from codeSpectra.codes.nec import _tables as NT
from codeSpectra.core import InvalidInput, SiteSpecificRequired


class TestZoneTable:
    @pytest.mark.parametrize(
        ("zone", "Z"),
        [("I", 0.15), ("II", 0.25), ("III", 0.30),
         ("IV", 0.35), ("V", 0.40), ("VI", 0.50)],
    )
    def test_zone_to_Z(self, zone: str, Z: float) -> None:
        assert SeismicZone(zone).Z == pytest.approx(Z)



class TestSoilTables:
    """Spot values from Tablas 3, 4 and 5."""

    @pytest.mark.parametrize(
        ("soil", "Z", "expected"),
        [("A", 0.15, 0.90), ("A", 0.50, 0.90),
         ("B", 0.30, 1.00),
         ("C", 0.15, 1.40), ("C", 0.40, 1.20), ("C", 0.50, 1.18),
         ("D", 0.15, 1.60), ("D", 0.40, 1.20), ("D", 0.50, 1.12),
         ("E", 0.15, 1.80), ("E", 0.40, 1.00), ("E", 0.50, 0.85)],
    )
    def test_fa(self, soil: str, Z: float, expected: float) -> None:
        assert NT.FA_TABLE.lookup(soil, Z) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("soil", "Z", "expected"),
        [("C", 0.15, 1.36), ("D", 0.40, 1.19), ("E", 0.15, 2.10),
         ("E", 0.50, 1.50)],
    )
    def test_fd(self, soil: str, Z: float, expected: float) -> None:
        assert NT.FD_TABLE.lookup(soil, Z) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("soil", "Z", "expected"),
        [("A", 0.40, 0.75), ("C", 0.50, 1.23), ("D", 0.40, 1.28),
         ("E", 0.50, 2.00)],
    )
    def test_fs(self, soil: str, Z: float, expected: float) -> None:
        assert NT.FS_TABLE.lookup(soil, Z) == pytest.approx(expected)

    def test_soil_F_not_tabulated(self) -> None:
        with pytest.raises(SiteSpecificRequired, match=r"10\.5\.4"):
            NT.FA_TABLE.lookup("F", 0.40)


class TestEta:
    @pytest.mark.parametrize(
        ("region", "eta"),
        [("costa", 1.80), ("sierra", 2.48), ("oriente", 2.60)],
    )
    def test_eta_by_region(self, region: str, eta: float) -> None:
        assert Region(region).eta == pytest.approx(eta)


class TestParameters:
    site = NECSEDS2015(zone="V", soil="D", region="sierra")

    def test_coefficients(self) -> None:
        assert self.site.Fa == pytest.approx(1.20)
        assert self.site.Fd == pytest.approx(1.19)
        assert self.site.Fs == pytest.approx(1.28)

    def test_control_periods(self) -> None:
        cp = self.site.control_periods
        ratio = 1.28 * 1.19 / 1.20
        assert cp.T0 == pytest.approx(0.10 * ratio)

        assert cp.Tc == pytest.approx(0.55 * ratio)

    def test_TL(self) -> None:
        assert self.site.control_periods.TL == pytest.approx(2.4 * 1.19)


    def test_TL_capped_at_4s_for_soil_E(self) -> None:
        """The §3.3.1 note caps TL at 4 s for soil types D and E."""
        site = NECSEDS2015(zone="I", soil="E", region="sierra")
        assert 2.4 * site.Fd > 4.0
        assert site.control_periods.TL == pytest.approx(4.0)

        assert any("capped at 4.0 s" in n for n in site.report().notes)

    def test_r_exponent(self) -> None:
        assert NECSEDS2015(zone="V", soil="D").r == pytest.approx(1.0)
        assert NECSEDS2015(zone="V", soil="E").r == pytest.approx(1.5)

    def test_plateau(self) -> None:
        assert self.site.Sa_plateau == pytest.approx(2.48 * 0.40 * 1.20)

    def test_zone_and_conflicting_Z_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="Supply only one"):
            NECSEDS2015(zone="V", Z=0.35, soil="D")

    def test_zone_or_Z_required(self) -> None:
        with pytest.raises(InvalidInput, match="Supply either zone"):
            NECSEDS2015(soil="D")

    def test_interpolated_Z_is_flagged(self) -> None:
        site = NECSEDS2015(Z=0.325, soil="D", region="sierra")
        assert any("interpolation" in n for n in site.report().notes)

    def test_tabulated_Z_is_not_flagged(self) -> None:
        assert not any("interpolation" in n for n in self.site.report().notes)


class TestSpectrumShape:
    site = NECSEDS2015(zone="V", soil="D", region="sierra")

    def test_plateau_extends_from_zero_by_default(self) -> None:
        """Design spectrum: the plateau runs from T = 0, not from T0."""
        s = self.site.elastic_spectrum()
        plateau = self.site.Sa_plateau
        for T in (0.0, 0.05, 0.1, 0.3, 0.6):
            assert s.at(T) == pytest.approx(plateau)

    def test_decay_branch(self) -> None:
        s = self.site.elastic_spectrum()
        Tc = self.site.control_periods.Tc
        assert s.at(2.0) == pytest.approx(self.site.Sa_plateau * (Tc / 2.0))

    def test_soil_E_decays_faster(self) -> None:
        """r = 1.5 for soil E."""
        site = NECSEDS2015(zone="V", soil="E", region="sierra")
        s = site.elastic_spectrum()
        Tc = site.control_periods.Tc
        assert s.at(2.0) == pytest.approx(site.Sa_plateau * (Tc / 2.0) ** 1.5)

    def test_ascending_branch_is_off_by_default(self) -> None:
        without = self.site.elastic_spectrum()
        with_ramp = self.site.elastic_spectrum(include_ascending_branch=True)
        assert with_ramp.at(0.0) < without.at(0.0)

    def test_ascending_branch_starts_at_Z_Fa(self) -> None:
        s = self.site.elastic_spectrum(include_ascending_branch=True)
        assert s.at(0.0) == pytest.approx(0.40 * 1.20)

    def test_ascending_branch_meets_the_plateau_at_T0(self) -> None:
        s = self.site.elastic_spectrum(include_ascending_branch=True)
        assert s.at(self.site.control_periods.T0) == pytest.approx(
            self.site.Sa_plateau
        )

    def test_continuous_at_Tc(self) -> None:
        s = self.site.elastic_spectrum()
        Tc = self.site.control_periods.Tc
        assert s.at(Tc - 1e-7) == pytest.approx(s.at(Tc + 1e-7), rel=1e-6)

    def test_monotonic_decay_past_Tc(self) -> None:
        s = self.site.elastic_spectrum()
        T = np.linspace(self.site.control_periods.Tc, 4.0, 200)
        assert np.all(np.diff(s.at(T)) <= 1e-12)

    @pytest.mark.parametrize("zone", ["I", "II", "III", "IV", "V", "VI"])
    @pytest.mark.parametrize("soil", ["A", "B", "C", "D", "E"])
    @pytest.mark.parametrize("region", ["costa", "sierra", "oriente"])
    def test_every_combination_is_continuous_and_positive(
        self, zone: str, soil: str, region: str
    ) -> None:
        site = NECSEDS2015(zone=zone, soil=soil, region=region)
        s = site.elastic_spectrum(include_ascending_branch=True)
        T = np.linspace(0.0, 4.0, 400)
        Sa = s.at(T)
        assert np.all(Sa > 0.0)
        # No branch produces a jump larger than the local trend.
        assert np.max(np.abs(np.diff(Sa))) < 0.15 * site.Sa_plateau


class TestInelasticSpectrum:
    site = NECSEDS2015(zone="V", soil="D", region="sierra", occupancy="esencial")

    def test_reduction(self) -> None:
        elastic = self.site.elastic_spectrum()
        inelastic = self.site.inelastic_spectrum(R=8.0, phi_p=0.9, phi_e=1.0)
        expected = elastic.at(1.0) * 1.5 / (8.0 * 0.9 * 1.0)
        assert inelastic.at(1.0) == pytest.approx(expected)

    def test_importance_can_be_omitted(self) -> None:
        s = self.site.inelastic_spectrum(R=8.0, apply_importance=False)
        expected = self.site.elastic_spectrum().at(1.0) / 8.0
        assert s.at(1.0) == pytest.approx(expected)

    def test_phi_above_one_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="phi_p"):
            self.site.inelastic_spectrum(R=8.0, phi_p=1.2)


class TestBaseShear:
    site = NECSEDS2015(zone="V", soil="D", region="sierra")

    def test_approximate_period(self) -> None:
        """Ta = Ct hn^alpha; concrete frames: Ct=0.055, alpha=0.9."""
        assert elf.approximate_period(20.0, "hormigon_porticos") == pytest.approx(
            0.055 * 20.0**0.9
        )

    def test_steel_moment_frame_coefficients(self) -> None:
        assert elf.approximate_period(20.0, "acero_sin_arriostramientos") == (
            pytest.approx(0.072 * 20.0**0.8)
        )

    def test_unknown_structure_type_lists_options(self) -> None:
        with pytest.raises(InvalidInput, match="hormigon_porticos"):
            elf.approximate_period(20.0, "madera")

    @pytest.mark.parametrize(
        ("T", "k"),
        [(0.3, 1.0), (0.5, 1.0), (1.0, 1.25), (2.5, 2.0), (3.0, 2.0)],
    )
    def test_k_exponent(self, T: float, k: float) -> None:
        assert elf.vertical_distribution_exponent(T) == pytest.approx(k)

    def test_base_shear_formula(self) -> None:
        spectrum = self.site.elastic_spectrum()
        result = elf.base_shear(
            spectrum, W=1000.0, R=8.0, I=1.0, phi_p=0.9, phi_e=1.0,
            hn=20.0, structure_type="hormigon_porticos",
        )
        Ta = 0.055 * 20.0**0.9
        expected = 1.0 * spectrum.at(Ta) * 1000.0 / (8.0 * 0.9 * 1.0)
        assert result.Ta == pytest.approx(Ta)
        assert result.V == pytest.approx(expected)


    def test_method_2_capped_at_130_percent(self) -> None:
        """§6.3.3b: Método 2 must not exceed Método 1 by more than 30%."""
        spectrum = self.site.elastic_spectrum()
        Ta1 = elf.approximate_period(20.0, "hormigon_porticos")
        result = elf.base_shear(
            spectrum, W=1000.0, R=8.0, T=10.0 * Ta1, hn=20.0,
            structure_type="hormigon_porticos",
        )
        assert result.Ta == pytest.approx(1.30 * Ta1)

    def test_vertical_distribution_sums_to_V(self) -> None:
        spectrum = self.site.elastic_spectrum()
        result = elf.base_shear(
            spectrum, W=1000.0, R=8.0, hn=20.0,
            structure_type="hormigon_porticos",
            weights=[200.0] * 5, heights=[4.0, 8.0, 12.0, 16.0, 20.0],
        )
        assert result.Fx is not None
        assert float(np.sum(result.Fx)) == pytest.approx(result.V)

    def test_base_storey_shear_equals_V(self) -> None:
        spectrum = self.site.elastic_spectrum()
        result = elf.base_shear(
            spectrum, W=1000.0, R=8.0, hn=20.0,
            structure_type="hormigon_porticos",
            weights=[200.0] * 5, heights=[4.0, 8.0, 12.0, 16.0, 20.0],
        )
        assert result.Vx is not None
        assert float(result.Vx[0]) == pytest.approx(result.V)

    def test_mismatched_weights_and_heights_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="length mismatch"):
            elf.vertical_distribution(100.0, [1.0, 2.0], [1.0], 1.0)


class TestReport:
    def test_report_is_cp1252_safe(self) -> None:
        """Reports contain Spanish text and must print on a Windows console."""
        for zone in ("I", "V", "VI"):
            for soil in ("A", "C", "E"):
                report = NECSEDS2015(zone=zone, soil=soil).report()
                report.to_text().encode("cp1252")

    def test_report_carries_values(self) -> None:
        report = NECSEDS2015(zone="V", soil="D", region="sierra").report()
        assert report["Fa"] == pytest.approx(1.20)
        assert report["eta"] == pytest.approx(2.48)
