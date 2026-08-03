"""ASCE/SEI 7 editions: table values, spectrum shape, and code-mandated rules."""

from __future__ import annotations

import numpy as np
import pytest

from codeSpectra.codes.asce7 import (
    ASCE7_10,
    ASCE7_16,
    ASCE7_22,
    MPRS_PERIODS,
    SeismicDesignCategory,
    SpectrumBasis,
)
from codeSpectra.codes.asce7 import _tables_7_16 as T716
from codeSpectra.core import InvalidInput, SiteSpecificRequired


class TestSiteCoefficientTables716:
    """Spot values straight off ASCE 7-16 Tables 11.4-1 and 11.4-2."""

    @pytest.mark.parametrize(
        ("site_class", "Ss", "expected"),
        [
            ("A", 0.25, 0.8), ("A", 1.5, 0.8),
            ("B", 0.75, 0.9),
            ("C", 0.25, 1.3), ("C", 0.75, 1.2), ("C", 1.5, 1.2),
            ("D", 0.25, 1.6), ("D", 0.5, 1.4), ("D", 1.0, 1.1), ("D", 1.5, 1.0),
            ("E", 0.25, 2.4), ("E", 0.5, 1.7), ("E", 0.75, 1.3),
        ],
    )
    def test_fa(self, site_class: str, Ss: float, expected: float) -> None:
        assert T716.FA_TABLE.lookup(site_class, Ss) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("site_class", "S1", "expected"),
        [
            ("A", 0.1, 0.8), ("B", 0.6, 0.8),
            ("C", 0.1, 1.5), ("C", 0.6, 1.4),
            ("D", 0.1, 2.4), ("D", 0.3, 2.0), ("D", 0.6, 1.7),
            ("E", 0.1, 4.2),
        ],
    )
    def test_fv(self, site_class: str, S1: float, expected: float) -> None:
        assert T716.FV_TABLE.lookup(site_class, S1) == pytest.approx(expected)

    def test_fa_interpolates(self) -> None:
        # Site Class D between Ss=0.5 (1.4) and Ss=0.75 (1.2).
        assert T716.FA_TABLE.lookup("D", 0.625) == pytest.approx(1.3)

    def test_cv_table(self) -> None:
        assert T716.CV_TABLE.lookup("AB", 2.0) == pytest.approx(0.9)
        assert T716.CV_TABLE.lookup("C", 1.0) == pytest.approx(1.1)
        assert T716.CV_TABLE.lookup("DEF", 2.0) == pytest.approx(1.5)
        assert T716.CV_TABLE.lookup("DEF", 0.2) == pytest.approx(0.7)


class TestASCE716Parameters:
    def test_worked_parameters(self) -> None:
        site = ASCE7_16(Ss=1.5, S1=0.6, site_class="D", TL=8.0)
        assert site.Fa == pytest.approx(1.0)
        assert site.Fv == pytest.approx(1.7)
        assert site.SMS == pytest.approx(1.5)

        assert site.SM1 == pytest.approx(1.02)

        assert site.SDS == pytest.approx(1.0)

        assert site.SD1 == pytest.approx(0.68)


    def test_control_periods(self) -> None:
        site = ASCE7_16(Ss=1.5, S1=0.6, site_class="D", TL=8.0)
        cp = site.control_periods
        assert cp.Ts == pytest.approx(0.68)
        assert cp.T0 == pytest.approx(0.136)

        assert cp.TL == pytest.approx(8.0)


    def test_default_site_class_D_floors_Fa_at_1_2(self) -> None:
        """§11.4.4: Fa >= 1.2 where Site Class D is the default."""
        plain = ASCE7_16(Ss=1.5, S1=0.6, site_class="D")
        defaulted = ASCE7_16(Ss=1.5, S1=0.6, site_class="D", default_site_class=True)
        assert plain.Fa == pytest.approx(1.0)
        assert defaulted.Fa == pytest.approx(1.2)

    def test_default_site_class_flag_only_valid_for_D(self) -> None:
        site = ASCE7_16(Ss=1.0, S1=0.4, site_class="C", default_site_class=True)
        with pytest.raises(InvalidInput, match="only applies to Site Class D"):
            _ = site.Fa

    def test_unmeasured_rock_B_uses_unity(self) -> None:
        """§11.4.3: Site Class B rock without velocity measurements."""
        site = ASCE7_16(Ss=1.0, S1=0.4, site_class="B",
                        measured_shear_wave_velocity=False)
        assert site.Fa == pytest.approx(1.0)
        assert site.Fv == pytest.approx(1.0)

    def test_overrides_bypass_the_tables(self) -> None:
        site = ASCE7_16(Ss=1.5, S1=0.6, site_class="E",
                        Fa_override=1.25, Fv_override=3.0)
        assert site.Fa == pytest.approx(1.25)
        assert site.SMS == pytest.approx(1.875)


    def test_s1_greater_than_ss_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="exceeds Ss"):
            ASCE7_16(Ss=0.4, S1=0.6)

    def test_asce722_site_class_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="ASCE 7-22"):
            ASCE7_16(Ss=1.0, S1=0.4, site_class="CD")


class TestSiteSpecificEnforcement:
    def test_site_class_E_high_Ss_raises(self) -> None:
        site = ASCE7_16(Ss=1.2, S1=0.5, site_class="E")
        with pytest.raises(SiteSpecificRequired, match=r"11\.4\.8"):
            _ = site.Fa

    def test_exception_1_substitutes_site_class_C(self) -> None:
        """§11.4.8 Exception 1: use the Site Class C value of Fa."""
        site = ASCE7_16(Ss=1.2, S1=0.1, site_class="E",
                        allow_site_specific_exception=True)
        assert site.Fa == pytest.approx(T716.FA_TABLE.lookup("C", 1.2))

    def test_exception_1_does_not_rescue_Fv(self) -> None:
        """Exception 1 addresses the Ss trigger only; Fv is still undefined.

        Site Class E with S1 >= 0.2 has no tabulated Fv. Exception 3 waives the
        hazard analysis in some cases but supplies no coefficient, so the
        engineer must inject one explicitly rather than have one invented.
        """
        site = ASCE7_16(Ss=1.2, S1=0.5, site_class="E",
                        allow_site_specific_exception=True)
        with pytest.raises(SiteSpecificRequired):
            _ = site.Fv
        rescued = ASCE7_16(Ss=1.2, S1=0.5, site_class="E",
                           allow_site_specific_exception=True, Fv_override=3.2)
        assert rescued.Fv == pytest.approx(3.2)

    def test_site_class_F_always_raises(self) -> None:
        site = ASCE7_16(Ss=0.5, S1=0.2, site_class="F")
        with pytest.raises(SiteSpecificRequired):
            _ = site.Fa

    def test_triggers_are_reported_not_raised(self) -> None:
        site = ASCE7_16(Ss=1.5, S1=0.6, site_class="D")
        texts = [t for t, _ in site.site_specific_triggers]
        assert any("S1 >= 0.2" in t for t in texts)
        # The design spectrum still builds; the trigger is advisory.
        assert site.design_spectrum().at(1.0) > 0.0

    def test_isolated_structure_trigger(self) -> None:
        site = ASCE7_16(Ss=1.0, S1=0.6, site_class="C", seismically_isolated=True)
        assert any("isolated" in t for t, _ in site.site_specific_triggers)


class TestTwoPeriodSpectrumShape:
    site = ASCE7_16(Ss=1.5, S1=0.6, site_class="D", TL=8.0)

    def test_zero_period_is_0_4_SDS(self) -> None:
        assert self.site.design_spectrum().at(0.0) == pytest.approx(0.4)

    def test_plateau_between_T0_and_Ts(self) -> None:
        s = self.site.design_spectrum()
        for T in (0.136, 0.3, 0.5, 0.68):
            assert s.at(T) == pytest.approx(1.0)

    def test_one_over_T_branch(self) -> None:
        s = self.site.design_spectrum()
        assert s.at(1.0) == pytest.approx(0.68)
        assert s.at(4.0) == pytest.approx(0.17)

    def test_one_over_T_squared_beyond_TL(self) -> None:
        s = self.site.design_spectrum()
        assert s.at(10.0) == pytest.approx(0.68 * 8.0 / 100.0)

    def test_continuous_at_every_control_period(self) -> None:
        s = self.site.design_spectrum()
        for T in s.control_periods.values():
            eps = 1e-7
            assert s.at(T - eps) == pytest.approx(s.at(T + eps), rel=1e-5)

    def test_monotonic_decay_past_Ts(self) -> None:
        s = self.site.design_spectrum()
        T = np.linspace(0.68, 10.0, 200)
        Sa = s.at(T)
        assert np.all(np.diff(Sa) <= 1e-12)

    def test_mcer_is_1_5_times_design(self) -> None:
        design = self.site.design_spectrum()
        mcer = self.site.mcer_spectrum()
        for T in (0.0, 0.2, 1.0, 5.0, 9.0):
            assert mcer.at(T) == pytest.approx(1.5 * design.at(T))


class TestVerticalSpectrum:
    site = ASCE7_16(Ss=1.5, S1=0.6, site_class="D", TL=8.0)

    def test_branches(self) -> None:
        Cv = T716.CV_TABLE.lookup("DEF", 1.5)
        SMS = self.site.SMS
        v = self.site.vertical_spectrum(design=False)
        assert v.at(0.01) == pytest.approx(0.3 * Cv * SMS)
        assert v.at(0.1) == pytest.approx(0.8 * Cv * SMS)
        assert v.at(1.0) == pytest.approx(0.8 * Cv * SMS * (0.15 / 1.0) ** 0.75)

    def test_design_is_two_thirds_of_mcer(self) -> None:
        mcer = self.site.vertical_spectrum(design=False)
        design = self.site.vertical_spectrum(design=True)
        assert design.at(0.5) == pytest.approx((2.0 / 3.0) * mcer.at(0.5))

    def test_reported_only_to_2s(self) -> None:
        assert self.site.vertical_spectrum().t_max == pytest.approx(2.0)


class TestSeismicDesignCategory:
    @pytest.mark.parametrize(
        ("SDS", "SD1", "S1", "risk", "expected"),
        [
            (0.10, 0.05, 0.05, "II", "A"),
            (0.25, 0.10, 0.10, "II", "B"),
            (0.40, 0.15, 0.15, "II", "C"),
            (0.80, 0.40, 0.40, "II", "D"),
            (0.25, 0.10, 0.10, "IV", "C"),
            # S1 >= 0.75 overrides both tables.
            (0.80, 0.40, 0.80, "II", "E"),
            (0.80, 0.40, 0.80, "IV", "F"),
        ],
    )
    def test_categories(
        self, SDS: float, SD1: float, S1: float, risk: str, expected: str
    ) -> None:
        from codeSpectra.codes.asce7 import seismic_design_category
        from codeSpectra.codes.asce7._shared import RiskCategory

        result = seismic_design_category(SDS, SD1, S1, RiskCategory(risk))
        assert result is SeismicDesignCategory(expected)

    def test_takes_more_severe_of_the_two_tables(self) -> None:
        """SDS says B, SD1 says D: D governs."""
        from codeSpectra.codes.asce7 import seismic_design_category
        from codeSpectra.codes.asce7._shared import RiskCategory

        assert seismic_design_category(
            0.20, 0.30, 0.30, RiskCategory.II
        ) is SeismicDesignCategory.D


class TestASCE710:
    def test_table_values_differ_from_716(self) -> None:
        """7-10 Site Class C at Ss=0.25 is 1.2, where 7-16 has 1.3."""
        a10 = ASCE7_10(Ss=0.25, S1=0.1, site_class="C")
        a16 = ASCE7_16(Ss=0.25, S1=0.1, site_class="C")
        assert a10.Fa == pytest.approx(1.2)
        assert a16.Fa == pytest.approx(1.3)

    def test_site_class_B_is_unity_in_710(self) -> None:
        site = ASCE7_10(Ss=1.0, S1=0.4, site_class="B")
        assert site.Fa == pytest.approx(1.0)
        assert site.Fv == pytest.approx(1.0)

    def test_site_class_E_fully_tabulated(self) -> None:
        """7-10 tabulates Site Class E everywhere; 7-16 does not."""
        site = ASCE7_10(Ss=1.5, S1=0.5, site_class="E")
        assert site.Fa == pytest.approx(0.9)
        assert site.Fv == pytest.approx(2.4)

    def test_spectrum_shape_matches_716_given_same_parameters(self) -> None:
        a10 = ASCE7_10(Ss=1.5, S1=0.6, site_class="D", TL=8.0)
        a16 = ASCE7_16(Ss=1.5, S1=0.6, site_class="D", TL=8.0,
                       Fa_override=a10.Fa, Fv_override=a10.Fv)
        for T in (0.0, 0.3, 1.0, 9.0):
            assert a10.design_spectrum().at(T) == pytest.approx(
                a16.design_spectrum().at(T)
            )


class TestASCE722:
    def test_two_period_path(self) -> None:
        site = ASCE7_22.from_site_adjusted(SMS=1.5, SM1=1.02, site_class="CD")
        assert site.SDS == pytest.approx(1.0)

        assert site.SD1 == pytest.approx(0.68)

        assert site.basis is SpectrumBasis.TWO_PERIOD

    def test_expanded_site_classes_accepted(self) -> None:
        for sc in ("BC", "CD", "DE"):
            site = ASCE7_22.from_site_adjusted(SMS=1.0, SM1=0.5, site_class=sc)
            assert site.sc == sc

    def test_mprs_path_returns_tabulated_spectrum(self) -> None:
        mcer = [1.5 * v for v in (0.6, 0.8, 1.0, 1.0, 0.5, 0.2)]
        periods = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
        site = ASCE7_22.from_mprs(periods, mcer, site_class="CD", TL=8.0)
        s = site.design_spectrum()
        # Design ordinates are 2/3 of the supplied MCEr values.
        assert s.at(0.2) == pytest.approx(1.0)
        assert s.at(1.0) == pytest.approx(0.5 * 1.5 * 2.0 / 3.0)

    def test_mprs_interpolates_linearly_in_period(self) -> None:
        site = ASCE7_22.from_mprs([0.0, 1.0], [1.5, 0.75], site_class="CD")
        assert site.design_spectrum().at(0.5) == pytest.approx(0.75)

    def test_beyond_10s_rule(self) -> None:
        """§11.4.5.1 item 3: factor by 10/T below TL, 10*TL/T^2 above."""
        periods = list(MPRS_PERIODS)
        mcer = [1.5] * len(periods)
        site = ASCE7_22.from_mprs(periods, mcer, site_class="CD", TL=12.0)
        s = site.design_spectrum()
        at_10 = s.at(10.0)
        assert s.at(11.0) == pytest.approx(at_10 * 10.0 / 11.0)
        assert s.at(16.0) == pytest.approx(at_10 * 10.0 * 12.0 / 16.0**2)

    def test_mprs_and_two_period_diverge(self) -> None:
        """The two bases are not interchangeable; the library exposes both."""
        periods = list(MPRS_PERIODS)
        mcer = [1.5] * len(periods)   # deliberately flat, unlike 1/T
        site = ASCE7_22.from_mprs(periods, mcer, site_class="CD", TL=8.0)
        assert site.design_spectrum().at(5.0) != pytest.approx(
            site.two_period_spectrum().at(5.0)
        )

    def test_mprs_length_mismatch_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="length mismatch"):
            ASCE7_22.from_mprs([0.0, 1.0], [1.0])

    def test_report_warns_when_S1_missing(self) -> None:
        site = ASCE7_22.from_site_adjusted(SMS=1.5, SM1=1.02, site_class="CD")
        assert any("S1 was not supplied" in n for n in site.report().notes)

    def test_report_warns_on_two_period_basis(self) -> None:
        site = ASCE7_22.from_site_adjusted(SMS=1.5, SM1=1.02, site_class="CD")
        assert any("Exception 2" in n for n in site.report().notes)


class TestReports:
    def test_report_carries_citations(self) -> None:
        report = ASCE7_16(Ss=1.5, S1=0.6, site_class="D").report()
        assert report["SDS"] == pytest.approx(1.0)
        assert any("11.4-1" in str(r) for r in report.refs())

    def test_report_renders_to_text_and_markdown(self) -> None:
        report = ASCE7_16(Ss=1.5, S1=0.6, site_class="D").report()
        assert "SDS" in report.to_text()
        assert "| `SDS` |" in report.to_markdown()

    def test_report_text_is_cp1252_safe(self) -> None:
        """Windows consoles default to cp1252; reports must survive printing."""
        for report in (
            ASCE7_16(Ss=1.5, S1=0.6, site_class="D").report(),
            ASCE7_10(Ss=1.5, S1=0.6, site_class="D").report(),
            ASCE7_22.from_site_adjusted(SMS=1.5, SM1=1.0).report(),
        ):
            report.to_text().encode("cp1252")


class TestASCE722IncompleteConstruction:
    """A silently-zero SDS is the failure this library exists to prevent.

    SMS/SM1 default to zero, so a direct construction that omits them used to
    build happily and report SDS = SD1 = 0. The complaint then surfaced much
    later and from somewhere unrelated, by which point a zero design
    acceleration could already have reached a model.
    """

    def test_Ss_S1_alone_is_rejected(self) -> None:
        """The natural wrong call for someone carrying 7-16 habits over."""
        with pytest.raises(InvalidInput, match="would both be 0"):
            ASCE7_22(Ss=1.5, S1=0.6, site_class="D")

    def test_Ss_S1_message_explains_the_7_22_change(self) -> None:
        with pytest.raises(InvalidInput) as exc:
            ASCE7_22(Ss=1.5, S1=0.6, site_class="D")
        message = str(exc.value)
        assert "11.4.3" in message
        assert "Geodatabase" in message
        assert "from_site_adjusted" in message
        assert "ASCE7_16" in message          # where 7-16 Ss/S1 actually belong

    def test_bare_construction_is_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="from_mprs"):
            ASCE7_22()

    def test_SMS_without_mprs_points_at_the_two_period_constructor(self) -> None:
        """Right values, wrong basis: default basis is multi-period."""
        with pytest.raises(InvalidInput, match="from_site_adjusted"):
            ASCE7_22(SMS=1.5, SM1=1.02, site_class="CD")

    def test_two_period_basis_needs_positive_SMS(self) -> None:
        with pytest.raises(InvalidInput, match="positive SMS"):
            ASCE7_22(basis=SpectrumBasis.TWO_PERIOD, SMS=0.0, site_class="CD")

    def test_supported_constructors_are_unaffected(self) -> None:
        assert ASCE7_22.from_site_adjusted(
            SMS=1.5, SM1=1.02, site_class="CD"
        ).SDS == pytest.approx(1.0)
        assert ASCE7_22.from_mprs(
            [0.0, 1.0], [1.5, 0.75], site_class="CD"
        ).design_spectrum().at(0.5) == pytest.approx(0.75)

    def test_no_instance_can_report_zero_SDS(self) -> None:
        """The invariant the guard exists to hold."""
        for site in (
            ASCE7_22.from_site_adjusted(SMS=1.5, SM1=1.02, site_class="CD"),
            ASCE7_22.from_mprs([0.0, 1.0], [1.5, 0.75], site_class="CD"),
        ):
            assert site.SDS > 0.0
