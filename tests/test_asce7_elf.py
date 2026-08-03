"""ASCE 7 §12.8 equivalent lateral force procedure."""

from __future__ import annotations

import numpy as np
import pytest

from codeSpectra.codes.asce7 import ASCE7_16, elf
from codeSpectra.core import InvalidInput


class TestApproximatePeriod:
    @pytest.mark.parametrize(
        ("structure_type", "Ct_m", "x"),
        [
            ("steel_moment_frame", 0.0724, 0.8),
            ("concrete_moment_frame", 0.0466, 0.9),
            ("steel_eccentrically_braced_frame", 0.0731, 0.75),
            ("steel_buckling_restrained_braced_frame", 0.0731, 0.75),
            ("other", 0.0488, 0.75),
        ],
    )
    def test_table_12_8_2_metric(
        self, structure_type: str, Ct_m: float, x: float
    ) -> None:
        assert elf.approximate_period(30.0, structure_type) == pytest.approx(
            Ct_m * 30.0**x
        )

    def test_imperial_coefficients(self) -> None:
        """Ct = 0.028 with hn in feet for steel moment frames."""
        assert elf.approximate_period(
            100.0, "steel_moment_frame", metric=False
        ) == pytest.approx(0.028 * 100.0**0.8)

    def test_metric_and_imperial_agree_on_the_same_building(self) -> None:
        """30.48 m == 100 ft; the two Ct values must give the same period."""
        metric = elf.approximate_period(30.48, "steel_moment_frame", metric=True)
        imperial = elf.approximate_period(100.0, "steel_moment_frame", metric=False)
        assert metric == pytest.approx(imperial, rel=1e-3)

    def test_non_positive_height_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="hn must be positive"):
            elf.approximate_period(0.0)


class TestUpperLimitCoefficient:
    @pytest.mark.parametrize(
        ("SD1", "Cu"),
        [(0.05, 1.7), (0.1, 1.7), (0.15, 1.6), (0.2, 1.5), (0.3, 1.4),
         (0.4, 1.4), (0.8, 1.4)],
    )
    def test_table_12_8_1(self, SD1: float, Cu: float) -> None:
        assert elf.upper_limit_coefficient(SD1) == pytest.approx(Cu)

    def test_interpolates(self) -> None:
        assert elf.upper_limit_coefficient(0.125) == pytest.approx(1.65)


class TestSeismicResponseCoefficient:
    def test_short_period_governed_by_eq_12_8_2(self) -> None:
        Cs, bd = elf.seismic_response_coefficient(
            SDS=1.0, SD1=0.6, S1=0.5, T=0.3, TL=8.0, R=8.0, Ie=1.0
        )
        assert Cs == pytest.approx(1.0 / 8.0)
        assert bd["Cs_eq_12.8-2"] == pytest.approx(0.125)

    def test_eq_12_8_3_cap_value(self) -> None:
        """The T <= TL cap is SD1/(T R/Ie), whether or not it governs."""
        _, bd = elf.seismic_response_coefficient(
            SDS=1.0, SD1=0.6, S1=0.5, T=2.0, TL=8.0, R=8.0, Ie=1.0
        )
        assert bd["Cs_cap_eq_12.8-3/4"] == pytest.approx(0.6 / (2.0 * 8.0))

    def test_eq_12_8_4_cap_value_beyond_TL(self) -> None:
        _, bd = elf.seismic_response_coefficient(
            SDS=1.0, SD1=0.6, S1=0.5, T=10.0, TL=8.0, R=8.0, Ie=1.0
        )
        assert bd["Cs_cap_eq_12.8-3/4"] == pytest.approx(
            0.6 * 8.0 / (100.0 * 8.0)
        )

    def test_cap_governs_where_it_exceeds_the_floor(self) -> None:
        """Low SDS keeps the 0.044 SDS Ie floor out of the way."""
        Cs, bd = elf.seismic_response_coefficient(
            SDS=0.30, SD1=0.25, S1=0.1, T=1.5, TL=8.0, R=4.0, Ie=1.0
        )
        assert bd["Cs_cap_eq_12.8-3/4"] == pytest.approx(0.25 / (1.5 * 4.0))
        assert Cs == pytest.approx(bd["Cs_cap_eq_12.8-3/4"])

    def test_floor_governs_over_the_cap_at_long_period(self) -> None:
        """At T = 2 s with SDS = 1.0 the 0.044 SDS Ie floor beats the cap."""
        Cs, bd = elf.seismic_response_coefficient(
            SDS=1.0, SD1=0.6, S1=0.5, T=2.0, TL=8.0, R=8.0, Ie=1.0
        )
        assert bd["Cs_cap_eq_12.8-3/4"] < bd["Cs_min_eq_12.8-5"]
        assert Cs == pytest.approx(0.044)

    def test_floor_eq_12_8_5(self) -> None:
        Cs, _ = elf.seismic_response_coefficient(
            SDS=1.0, SD1=0.6, S1=0.1, T=40.0, TL=8.0, R=8.0, Ie=1.0
        )
        assert Cs == pytest.approx(0.044 * 1.0 * 1.0)

    def test_absolute_floor_of_0_01(self) -> None:
        Cs, _ = elf.seismic_response_coefficient(
            SDS=0.1, SD1=0.05, S1=0.02, T=30.0, TL=8.0, R=8.0, Ie=1.0
        )
        assert Cs == pytest.approx(0.01)

    def test_near_fault_floor_eq_12_8_6(self) -> None:
        """For S1 >= 0.6g, Cs >= 0.5 S1 / (R/Ie)."""
        Cs, _ = elf.seismic_response_coefficient(
            SDS=1.0, SD1=0.6, S1=0.75, T=20.0, TL=8.0, R=8.0, Ie=1.0
        )
        assert Cs == pytest.approx(0.5 * 0.75 / 8.0)

    def test_eq_12_8_6_does_not_apply_below_0_6(self) -> None:
        _, bd = elf.seismic_response_coefficient(
            SDS=1.0, SD1=0.6, S1=0.59, T=20.0, TL=8.0, R=8.0, Ie=1.0
        )
        assert bd["Cs_min_eq_12.8-6"] == 0.0

    def test_importance_factor_raises_demand(self) -> None:
        low, _ = elf.seismic_response_coefficient(
            SDS=1.0, SD1=0.6, S1=0.3, T=0.3, TL=8.0, R=8.0, Ie=1.0
        )
        high, _ = elf.seismic_response_coefficient(
            SDS=1.0, SD1=0.6, S1=0.3, T=0.3, TL=8.0, R=8.0, Ie=1.5
        )
        assert high == pytest.approx(1.5 * low)

    def test_non_positive_R_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="R must be positive"):
            elf.seismic_response_coefficient(
                SDS=1.0, SD1=0.6, S1=0.3, T=1.0, TL=8.0, R=0.0
            )


class TestDistributionExponent:
    @pytest.mark.parametrize(
        ("T", "k"), [(0.2, 1.0), (0.5, 1.0), (1.5, 1.5), (2.5, 2.0), (4.0, 2.0)]
    )
    def test_k(self, T: float, k: float) -> None:
        assert elf.vertical_distribution_exponent(T) == pytest.approx(k)

    def test_k_is_continuous(self) -> None:
        assert elf.vertical_distribution_exponent(0.5) == pytest.approx(
            elf.vertical_distribution_exponent(0.500001), abs=1e-5
        )


class TestBaseShear:
    site = ASCE7_16(Ss=1.5, S1=0.6, site_class="D", TL=8.0)

    def test_V_equals_Cs_W(self) -> None:
        result = elf.base_shear(
            W=10_000.0, SDS=self.site.SDS, SD1=self.site.SD1, S1=self.site.S1,
            TL=self.site.TL, R=8.0, Ie=1.0, hn=30.0,
            structure_type="concrete_moment_frame",
        )
        assert result.V == pytest.approx(result.Cs * 10_000.0)

    def test_period_capped_at_Cu_Ta(self) -> None:
        """§12.8.2: T shall not exceed Cu * Ta."""
        Ta = elf.approximate_period(30.0, "concrete_moment_frame")
        Cu = elf.upper_limit_coefficient(self.site.SD1)
        result = elf.base_shear(
            W=10_000.0, SDS=self.site.SDS, SD1=self.site.SD1, S1=self.site.S1,
            TL=self.site.TL, R=8.0, T=10.0, hn=30.0,
            structure_type="concrete_moment_frame",
        )
        assert result.T == pytest.approx(Cu * Ta)

    def test_cap_can_be_disabled(self) -> None:
        result = elf.base_shear(
            W=10_000.0, SDS=self.site.SDS, SD1=self.site.SD1, S1=self.site.S1,
            TL=self.site.TL, R=8.0, T=3.0, hn=30.0,
            structure_type="concrete_moment_frame", apply_upper_limit=False,
        )
        assert result.T == pytest.approx(3.0)

    def test_Ta_used_when_no_T_supplied(self) -> None:
        result = elf.base_shear(
            W=10_000.0, SDS=self.site.SDS, SD1=self.site.SD1, S1=self.site.S1,
            TL=self.site.TL, R=8.0, hn=30.0,
            structure_type="concrete_moment_frame",
        )
        assert result.T == pytest.approx(result.Ta)

    def test_requires_T_or_hn(self) -> None:
        with pytest.raises(InvalidInput, match="Supply the fundamental period"):
            elf.base_shear(W=1000.0, SDS=1.0, SD1=0.6, S1=0.3, TL=8.0, R=8.0)

    def test_storey_forces_sum_to_V(self) -> None:
        result = elf.base_shear(
            W=10_000.0, SDS=self.site.SDS, SD1=self.site.SD1, S1=self.site.S1,
            TL=self.site.TL, R=8.0, hn=30.0,
            structure_type="concrete_moment_frame",
            weights=[2000.0] * 5, heights=[6.0, 12.0, 18.0, 24.0, 30.0],
        )
        assert result.Fx is not None
        assert float(np.sum(result.Fx)) == pytest.approx(result.V)

    def test_forces_increase_with_height(self) -> None:
        result = elf.base_shear(
            W=10_000.0, SDS=self.site.SDS, SD1=self.site.SD1, S1=self.site.S1,
            TL=self.site.TL, R=8.0, hn=30.0,
            structure_type="concrete_moment_frame",
            weights=[2000.0] * 5, heights=[6.0, 12.0, 18.0, 24.0, 30.0],
        )
        assert result.Fx is not None
        assert np.all(np.diff(result.Fx) > 0)

    def test_base_storey_shear_equals_V(self) -> None:
        result = elf.base_shear(
            W=10_000.0, SDS=self.site.SDS, SD1=self.site.SD1, S1=self.site.S1,
            TL=self.site.TL, R=8.0, hn=30.0,
            structure_type="concrete_moment_frame",
            weights=[2000.0] * 5, heights=[6.0, 12.0, 18.0, 24.0, 30.0],
        )
        assert result.Vx is not None
        assert float(result.Vx[0]) == pytest.approx(result.V)
        assert float(result.Vx[-1]) == pytest.approx(float(result.Fx[-1]))  # type: ignore[index]

    def test_report_names_the_governing_equation(self) -> None:
        result = elf.base_shear(
            W=10_000.0, SDS=self.site.SDS, SD1=self.site.SD1, S1=self.site.S1,
            TL=self.site.TL, R=8.0, T=0.3, hn=30.0,
            structure_type="concrete_moment_frame", apply_upper_limit=False,
        )
        assert result.governing_equation == "Cs_eq_12.8-2"
        assert "Cs" in result.report().to_text()
