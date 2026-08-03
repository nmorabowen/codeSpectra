"""Core spine: Spectrum algebra, tables, control periods, reports."""

from __future__ import annotations

import numpy as np
import pytest

from codeSpectra.core import (
    AccelUnit,
    AnalyticSpectrum,
    ClauseRef,
    ControlPeriods,
    InterpolatedTable,
    InvalidInput,
    SiteSpecificRequired,
    SpectrumMeta,
    TableLookupError,
    TabulatedSpectrum,
)

REF = ClauseRef("Test", "1.0", "1.2.3", description="fixture")


def flat(value: float = 1.0, t_max: float = 5.0, **cp: float) -> AnalyticSpectrum:
    meta = SpectrumMeta(label="flat", control_periods=ControlPeriods(**cp))
    return AnalyticSpectrum(lambda T: np.full_like(T, value), meta=meta, t_max=t_max)


class TestControlPeriods:
    def test_attribute_and_item_access(self) -> None:
        cp = ControlPeriods(T0=0.12, Ts=0.6, TL=8.0)
        assert cp.Ts == 0.6
        assert cp["T0"] == 0.12
        assert list(cp) == ["T0", "Ts", "TL"]

    def test_is_immutable(self) -> None:
        cp = ControlPeriods(T0=0.1)
        with pytest.raises(AttributeError):
            cp.T0 = 0.2  # type: ignore[misc]

    def test_unknown_period_names_itself(self) -> None:
        cp = ControlPeriods(T0=0.1)
        with pytest.raises(AttributeError, match="Tc"):
            _ = cp.Tc

    def test_refine_grid_lands_on_breakpoints(self) -> None:
        cp = ControlPeriods(T0=0.137, Ts=0.683)
        grid = cp.refine_grid(np.linspace(0.0, 4.0, 50))
        assert 0.137 in grid
        assert 0.683 in grid
        assert np.all(np.diff(grid) > 0)

    def test_refine_grid_ignores_breakpoints_beyond_range(self) -> None:
        cp = ControlPeriods(TL=8.0)
        grid = cp.refine_grid(np.linspace(0.0, 4.0, 10))
        assert grid.max() == pytest.approx(4.0)


class TestInterpolatedTable:
    table = InterpolatedTable(
        name="Fa",
        row_label="site class",
        col_label="Ss",
        columns=(0.25, 0.5, 0.75),
        rows={"C": (1.3, 1.3, 1.2), "E": (2.4, 1.7, None)},
        ref=REF,
        site_specific_remedy="do the study",
    )

    def test_exact_column(self) -> None:
        assert self.table.lookup("C", 0.5) == 1.3

    def test_interpolates_between_columns(self) -> None:
        # Halfway from 1.3 at 0.5 to 1.2 at 0.75.
        assert self.table.lookup("C", 0.625) == pytest.approx(1.25)

    def test_clamps_outside_range(self) -> None:
        assert self.table.lookup("C", 0.0) == 1.3
        assert self.table.lookup("C", 99.0) == 1.2

    def test_missing_cell_raises(self) -> None:
        with pytest.raises(SiteSpecificRequired, match="do the study"):
            self.table.lookup("E", 0.75)

    def test_interpolating_toward_missing_cell_raises(self) -> None:
        """Interpolating toward an undefined cell is not a defined operation."""
        with pytest.raises(SiteSpecificRequired):
            self.table.lookup("E", 0.6)

    def test_unknown_row_raises(self) -> None:
        with pytest.raises(TableLookupError, match="site class"):
            self.table.lookup("Z", 0.5)

    def test_is_defined_reports_without_raising(self) -> None:
        assert self.table.is_defined("C", 0.6)
        assert not self.table.is_defined("E", 0.75)

    def test_row_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="expected 3"):
            InterpolatedTable(
                name="X", row_label="r", col_label="c",
                columns=(1.0, 2.0, 3.0), rows={"A": (1.0, 2.0)}, ref=REF,
            )

    def test_non_ascending_columns_rejected(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            InterpolatedTable(
                name="X", row_label="r", col_label="c",
                columns=(2.0, 1.0), rows={"A": (1.0, 2.0)}, ref=REF,
            )


class TestSpectrumEvaluation:
    def test_scalar_in_scalar_out(self) -> None:
        s = flat(0.8)
        assert isinstance(s.at(1.0), float)
        assert s.at(1.0) == pytest.approx(0.8)

    def test_array_in_array_out(self) -> None:
        s = flat(0.8)
        out = s.at([0.5, 1.0, 2.0])
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)

    def test_negative_period_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="non-negative"):
            flat().at(-1.0)

    def test_non_finite_period_rejected(self) -> None:
        with pytest.raises(InvalidInput, match="finite"):
            flat().at(np.inf)


class TestSpectrumAlgebra:
    def test_scaled(self) -> None:
        assert flat(1.0).scaled(1.5).at(1.0) == pytest.approx(1.5)

    def test_scaled_rejects_non_positive(self) -> None:
        with pytest.raises(InvalidInput, match="positive"):
            flat().scaled(0.0)

    def test_reduced_divides_by_all_factors(self) -> None:
        s = flat(1.2).reduced(R=6.0, Ie=1.5, phi_p=0.9, phi_e=0.9)
        assert s.at(1.0) == pytest.approx(1.2 * 1.5 / (6.0 * 0.9 * 0.9))

    def test_reduced_rejects_non_positive(self) -> None:
        with pytest.raises(InvalidInput, match="R must be positive"):
            flat().reduced(R=0.0)

    def test_envelope_takes_pointwise_maximum(self) -> None:
        env = flat(1.0).envelope(flat(1.5), flat(0.5))
        assert env.at(1.0) == pytest.approx(1.5)

    def test_floored_by_applies_ratio(self) -> None:
        """The ASCE 21.3 80% floor, expressed as a spectrum operation."""
        site_specific = flat(0.5)
        code = flat(1.0)
        floored = site_specific.floored_by(code, 0.80)
        assert floored.at(1.0) == pytest.approx(0.80)

    def test_floored_by_leaves_higher_values_alone(self) -> None:
        assert flat(2.0).floored_by(flat(1.0), 0.80).at(1.0) == pytest.approx(2.0)

    def test_capped_by(self) -> None:
        assert flat(2.0).capped_by(flat(1.0)).at(1.0) == pytest.approx(1.0)

    def test_operations_do_not_mutate_the_original(self) -> None:
        original = flat(1.0)
        original.scaled(3.0).reduced(2.0)
        assert original.at(1.0) == pytest.approx(1.0)

    def test_peak(self) -> None:
        meta = SpectrumMeta(label="bump")
        s = AnalyticSpectrum(lambda T: 1.0 - (T - 1.0) ** 2, meta=meta, t_max=2.0)
        T_peak, Sa_peak = s.peak()
        assert T_peak == pytest.approx(1.0, abs=1e-2)
        assert Sa_peak == pytest.approx(1.0, abs=1e-3)


class TestTabulatedSpectrum:
    def test_linear_interpolation(self) -> None:
        s = TabulatedSpectrum([0.0, 1.0, 2.0], [0.0, 1.0, 0.5])
        assert s.at(0.5) == pytest.approx(0.5)
        assert s.at(1.5) == pytest.approx(0.75)

    def test_sorts_unordered_input(self) -> None:
        s = TabulatedSpectrum([2.0, 0.0, 1.0], [0.5, 0.0, 1.0])
        assert s.at(0.5) == pytest.approx(0.5)

    def test_beyond_last_period_decays_as_one_over_T(self) -> None:
        s = TabulatedSpectrum([1.0, 10.0], [1.0, 0.1])
        assert s.at(20.0) == pytest.approx(0.1 * 10.0 / 20.0)

    def test_beyond_TL_decays_as_one_over_T_squared(self) -> None:
        meta = SpectrumMeta(control_periods=ControlPeriods(TL=12.0))
        s = TabulatedSpectrum([1.0, 10.0], [1.0, 0.1], meta=meta)
        assert s.at(20.0) == pytest.approx(0.1 * 10.0 * 12.0 / 20.0**2)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(InvalidInput, match="length mismatch"):
            TabulatedSpectrum([0.0, 1.0], [1.0])

    def test_rejects_duplicate_periods(self) -> None:
        with pytest.raises(InvalidInput, match="strictly increasing"):
            TabulatedSpectrum([1.0, 1.0], [1.0, 2.0])

    def test_rejects_negative_ordinates(self) -> None:
        with pytest.raises(InvalidInput, match="non-negative"):
            TabulatedSpectrum([0.0, 1.0], [1.0, -0.1])

    def test_scaling_stays_tabulated(self) -> None:
        s = TabulatedSpectrum([0.0, 1.0], [1.0, 0.5]).scaled(2.0)
        assert isinstance(s, TabulatedSpectrum)


class TestUnits:
    def test_g_is_identity(self) -> None:
        assert AccelUnit.G.factor_from_g == 1.0

    def test_conversion_to_ms2(self) -> None:
        s = flat(1.0)
        arr = s.to_numpy(periods=[1.0], unit=AccelUnit.M_S2)
        assert arr[0, 1] == pytest.approx(9.80665)

    def test_conversion_to_in_s2(self) -> None:
        s = flat(1.0)
        arr = s.to_numpy(periods=[1.0], unit="in/s2")
        assert arr[0, 1] == pytest.approx(386.0886, rel=1e-5)


class TestDisplacement:
    def test_sd_from_sa(self) -> None:
        """Sd = Sa g (T/2pi)^2."""
        s = flat(1.0)
        expected = 9.80665 * (2.0 / (2 * np.pi)) ** 2
        assert s.displacement(2.0) == pytest.approx(expected)


class TestClauseRef:
    def test_renders_section_and_equation(self) -> None:
        r = ClauseRef("ASCE/SEI 7", "7-16", "11.4.4", equation="11.4-1",
                      description="SMS")
        assert str(r) == "ASCE/SEI 7 7-16 §11.4.4, Eq. 11.4-1 — SMS"

    def test_is_hashable(self) -> None:
        assert len({REF, REF}) == 1
