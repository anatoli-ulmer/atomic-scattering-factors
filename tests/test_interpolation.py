"""Interpolation regression tests against the actual periodictable Henke tables."""

import numpy as np
import periodictable as pt
from periodictable.constants import avogadro_number
import pytest

from atomic_scattering_factors import (
    effective_electron_density,
    get_complex_scattering_factor,
    get_effective_Z,
    get_effective_Z_formula,
    get_scattering_factor_formula,
    get_scattering_factors,
)


def test_default_keeps_existing_periodictable_results():
    energy = np.array([[8000., 9000.], [11000., 11900.]])
    f1, f2 = get_scattering_factors("Au", energy)
    ref1, ref2 = pt.Au.xray.scattering_factors(energy=energy.ravel() / 1000.)
    np.testing.assert_array_equal(f1, np.asarray(ref1).reshape(energy.shape))
    np.testing.assert_array_equal(f2, np.asarray(ref2).reshape(energy.shape))


def test_spline_matches_its_source_table_exactly():
    pytest.importorskip("scipy")
    # Finite, exactly tabulated gold energies include the pair across Au L3.
    table = pt.Au.xray.sftable
    valid = np.isfinite(table[1]) & np.isfinite(table[2])
    energy = table[0, valid] * 1000.
    f1, f2 = get_scattering_factors("Au", energy, interpolation="pchip")
    np.testing.assert_allclose(f1, table[1, valid], rtol=0, atol=1e-12)
    np.testing.assert_allclose(f2, table[2, valid], rtol=0, atol=1e-12)


def test_spline_alias_scalar_and_nd_arrays():
    pytest.importorskip("scipy")
    energies = np.array([[11800., 11898.5], [11918.7, 12000.]])
    a, b = get_scattering_factors("Au", energies, interpolation="pchip")
    a2, b2 = get_scattering_factors("Au", energies, interpolation="spline")
    assert a.shape == b.shape == energies.shape
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)
    f1, f2 = get_scattering_factors("Au", 11898.5, interpolation="pchip")
    assert isinstance(f1, float) and isinstance(f2, float)
    assert f1 == pytest.approx(a[0, 1])
    assert f2 == pytest.approx(b[0, 1])
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert np.all(b >= 0)


def test_spline_can_differ_from_default_off_grid():
    pytest.importorskip("scipy")
    f1_default, _ = get_scattering_factors("Au", 11898.5)
    f1_pchip, _ = get_scattering_factors("Au", 11898.5, interpolation="pchip")
    assert abs(f1_default - f1_pchip) > 1e-4


def test_interpolation_propagates_into_complex_factors_and_densities():
    pytest.importorskip("scipy")
    energy = np.array([11000., 11898.5, 11920., 12000.])
    f1, f2 = get_scattering_factors("Au", energy, interpolation="pchip")
    np.testing.assert_allclose(
        get_complex_scattering_factor("Au", energy, interpolation="pchip"),
        f1 + 1j * f2,
    )
    np.testing.assert_allclose(
        get_effective_Z("Au", energy, interpolation="pchip"),
        np.hypot(f1, f2),
    )
    np.testing.assert_allclose(
        get_scattering_factor_formula("AuCl3", energy, interpolation="pchip"),
        get_complex_scattering_factor("Au", energy, interpolation="pchip")
        + 3 * get_complex_scattering_factor("Cl", energy, interpolation="pchip"),
    )
    np.testing.assert_allclose(
        get_effective_Z_formula("H2O", energy, interpolation="pchip"),
        2 * get_effective_Z("H", energy, interpolation="pchip")
        + get_effective_Z("O", energy, interpolation="pchip"),
    )
    rho = effective_electron_density("Au", energies_eV=energy, complex=True,
                                     interpolation="pchip")
    n = pt.Au.density * 1e6 * avogadro_number / pt.Au.mass
    np.testing.assert_allclose(rho, n * (f1 + 1j * f2), rtol=1e-12)
    assert np.iscomplexobj(rho)
    legacy = effective_electron_density("Au", energies_eV=energy,
                                        interpolation="pchip")
    np.testing.assert_allclose(legacy, n * np.hypot(f1, f2), rtol=1e-12)


def test_invalid_interpolation_or_energy_rejected():
    with pytest.raises(ValueError, match="interpolation"):
        get_scattering_factors("Au", 8000, interpolation="cubic_magic")
    with pytest.raises(ValueError):
        get_scattering_factors("Au", 31000, interpolation="pchip")
