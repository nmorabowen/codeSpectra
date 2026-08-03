"""NCh433 (Chile): tables, alpha, R*, and the migrated behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from codeSpectra.codes.nch433 import NCh433
from codeSpectra.core import InvalidInput


class TestTables:
    @pytest.mark.parametrize(("zone", "Ao"), [("1", 0.20), ("2", 0.30), ("3", 0.40)])
    def test_zone_Ao(self, zone: str, Ao: float) -> None:
        assert NCh433(zone=zone).Ao == pytest.approx(Ao)

    @pytest.mark.parametrize(
        ("category", "I"), [("I", 0.60), ("II", 1.00), ("III", 1.20), ("IV", 1.20)]
    )
    def test_importance(self, category: str, I: float) -> None:
        assert NCh433(category=category).I == pytest.approx(I)


    @pytest.mark.parametrize(
        ("soil", "S", "To", "Tp", "n", "p"),
        [
            ("A", 0.90, 0.15, 0.20, 1.00, 2.0),
            ("B", 1.00, 0.30, 0.35, 1.33, 1.5),
            ("C", 1.05, 0.40, 0.45, 1.40, 1.6),
            ("D", 1.20, 0.75, 0.85, 1.80, 1.0),
            ("E", 1.30, 1.20, 1.35, 1.80, 1.0),
        ],
    )
    def test_soil_parameters(
        self, soil: str, S: float, To: float, Tp: float, n: float, p: float
    ) -> None:
        site = NCh433(soil=soil)
        assert (site.S, site.To, site.Tp, site.n, site.p) == pytest.approx(
            (S, To, Tp, n, p)
        )

    def test_soil_F_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="site-specific"):
            NCh433(soil="F")


class TestSpectrum:
    site = NCh433(zone="2", soil="B", category="II")

    def test_alpha_at_zero_is_one(self) -> None:
        assert float(self.site.alpha(0.0)[0]) == pytest.approx(1.0)

    def test_alpha_matches_equation_6_9(self) -> None:
        T = 0.5
        ratio = T / self.site.To
        expected = (1 + 4.5 * ratio**self.site.p) / (1 + ratio**3)
        assert float(self.site.alpha(T)[0]) == pytest.approx(expected)

    def test_elastic_spectrum_at_zero_is_I_S_Ao(self) -> None:
        s = self.site.elastic_spectrum()
        assert s.at(0.0) == pytest.approx(1.00 * 1.00 * 0.30)

    def test_spectrum_is_positive_and_finite(self) -> None:
        s = self.site.elastic_spectrum()
        Sa = s.at(np.linspace(0.0, 4.0, 500))
        assert np.all(np.isfinite(Sa))
        assert np.all(Sa > 0.0)

    @pytest.mark.parametrize("soil", ["A", "B", "C", "D", "E"])
    @pytest.mark.parametrize("zone", ["1", "2", "3"])
    def test_every_combination_finite(self, soil: str, zone: str) -> None:
        s = NCh433(zone=zone, soil=soil).elastic_spectrum()
        assert np.all(np.isfinite(s.at(np.linspace(0.0, 4.0, 200))))


class TestReductionFactor:
    site = NCh433(zone="2", soil="B", category="II")

    def test_equation_6_10(self) -> None:
        T_star, Ro = 1.0, 7.0
        expected = 1.0 + T_star / (0.10 * self.site.To + T_star / Ro)
        assert self.site.reduction_factor(T_star, Ro) == pytest.approx(expected)

    def test_R_star_exceeds_one(self) -> None:
        assert self.site.reduction_factor(0.01, 2.0) > 1.0

    def test_R_star_increases_with_period(self) -> None:
        values = [self.site.reduction_factor(T, 7.0) for T in (0.2, 0.5, 1.0, 2.0)]
        assert values == sorted(values)

    def test_design_spectrum_divides_by_R_star(self) -> None:
        R_star = self.site.reduction_factor(1.0, 7.0)
        elastic = self.site.elastic_spectrum()
        design = self.site.design_spectrum(T_star=1.0, Ro=7.0)
        assert design.at(0.5) == pytest.approx(elastic.at(0.5) / R_star)

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(InvalidInput, match="T\\* must be positive"):
            self.site.reduction_factor(0.0, 7.0)


class TestCoefficientLimits:
    site = NCh433(zone="2", soil="B", category="II")

    def test_C_min(self) -> None:
        assert self.site.C_min == pytest.approx(1.00 * 0.30 / 6.0)

    @pytest.mark.parametrize(
        ("R", "factor"), [(2.0, 0.90), (3.0, 0.60), (4.0, 0.55), (5.5, 0.40),
                          (6.0, 0.35), (7.0, 0.35)]
    )
    def test_C_max_table(self, R: float, factor: float) -> None:
        assert self.site.C_max(R) == pytest.approx(factor * 1.00 * 0.30)

    def test_Sa_min(self) -> None:
        assert self.site.Sa_min() == pytest.approx(1.00 * 1.00 * 0.30 / 6.0)


class TestReport:
    def test_report_is_cp1252_safe(self) -> None:
        NCh433(zone="2", soil="B").report().to_text().encode("cp1252")

    def test_documents_unimplemented_equation(self) -> None:
        notes = NCh433().report().notes
        assert any("6-11" in n for n in notes)
