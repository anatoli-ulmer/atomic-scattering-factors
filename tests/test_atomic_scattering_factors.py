"""Regression tests for the NumPy-2-compatible ASAXS interface.

These tests supplement (do not replace) tests/test_atomic_scattering_factors.py.
"""

import numpy as np
import periodictable as pt
import pytest
from periodictable.constants import avogadro_number, electron_radius

from atomic_scattering_factors import (
    effective_electron_density,
    get_complex_scattering_factor,
    get_effective_Z_formula,
    get_scattering_factor_formula,
    get_scattering_factors,
    molecular_weight,
)


def test_direct_periodictable_factors_match_and_energy_in_ev():
    energies_eV = np.array([5000.0, 8000.0, 12000.0])
    f1, f2 = get_scattering_factors("Au", energies_eV)
    f1_ref, f2_ref = pt.Au.xray.scattering_factors(energy=energies_eV / 1000)
    np.testing.assert_allclose(f1, f1_ref)
    np.testing.assert_allclose(f2, f2_ref)


def test_multidimensional_scattering_factors_keep_shape():
    energies = np.array([[6000.0, 7000.0], [8000.0, 9000.0]])
    f1, f2 = get_scattering_factors("Si", energies)
    assert f1.shape == f2.shape == energies.shape
    f1_flat, f2_flat = get_scattering_factors("Si", energies.reshape(-1))
    np.testing.assert_allclose(f1.reshape(-1), f1_flat)
    np.testing.assert_allclose(f2.reshape(-1), f2_flat)


def test_coherent_formula_factor_is_sum_before_modulus():
    e = np.array([8000.0, 9000.0])
    actual = get_scattering_factor_formula("H2O", e)
    expected = 2 * get_complex_scattering_factor("H", e) + get_complex_scattering_factor("O", e)
    np.testing.assert_allclose(actual, expected)
    assert np.iscomplexobj(actual)
    assert actual.shape == e.shape


def test_chemical_aliases_and_fractional_counts():
    e = 8000.0
    assert get_scattering_factor_formula("HOH", e) == pytest.approx(get_scattering_factor_formula("H2O", e))
    expected = (get_complex_scattering_factor("Na", e)
                + 0.5 * get_complex_scattering_factor("Y", e)
                + 0.3 * get_complex_scattering_factor("Gd", e)
                + 4 * get_complex_scattering_factor("F", e))
    assert get_scattering_factor_formula("NaY0.5Gd0.3F4", e) == pytest.approx(expected)


def test_complex_density_is_number_density_times_coherent_amplitude():
    energy = np.array([[8000., 8500.], [9000., 9500.]])
    rho = effective_electron_density("H2O", mass_density_g_per_cm3=1.0,
                                     energies_eV=energy, complex=True)
    number_density = 1e6 * avogadro_number / molecular_weight("H2O")
    np.testing.assert_allclose(rho, number_density * get_scattering_factor_formula("H2O", energy))
    assert rho.shape == energy.shape
    assert np.iscomplexobj(rho)
    assert np.all(np.isfinite(rho.imag))


def test_real_and_imaginary_components_and_no_premature_absolute_value():
    e = 8000.0
    rho = effective_electron_density("SiO2", 2.2, e, complex=True)
    n = 2.2e6 * avogadro_number / molecular_weight("SiO2")
    f1_si, f2_si = get_scattering_factors("Si", e)
    f1_o, f2_o = get_scattering_factors("O", e)
    assert isinstance(rho, complex)
    assert rho.real == pytest.approx(n * (f1_si + 2 * f1_o))
    assert rho.imag == pytest.approx(n * (f2_si + 2 * f2_o))
    # In general |sum(f_i)| != sum(|f_i|). Legacy is kept but separate.
    legacy = effective_electron_density("SiO2", 2.2, e)
    assert legacy == pytest.approx(n * get_effective_Z_formula("SiO2", e))
    assert legacy >= abs(rho) * (1 - 1e-12)


def test_complex_density_is_consistent_with_periodictable_xray_sld():
    e = 8000.0
    rho = effective_electron_density("SiO2", 2.2, e, complex=True)
    # periodictable's xray_sld uses 1e-6 Å^-2 units. Convert to m^-2.
    sld_real, sld_imag = pt.xray_sld("SiO2", density=2.2, energy=e / 1000)
    np.testing.assert_allclose(electron_radius * rho, (sld_real + 1j * sld_imag) * 1e14, rtol=1e-6)


def test_complex_contrast_is_difference_of_complex_densities():
    e = np.array([11500.0, 11900.0, 12000.0])
    core = effective_electron_density("Au", energies_eV=e, complex=True)
    solvent = effective_electron_density("H2O", 1.0, e, complex=True)
    contrast = core - solvent
    assert contrast.shape == e.shape
    assert np.iscomplexobj(contrast)
    intensity_multiplier = np.abs(contrast) ** 2
    assert np.all(intensity_multiplier >= 0)
    assert np.all(np.isfinite(intensity_multiplier))


@pytest.mark.parametrize("energy", [0, -1, 31000, np.nan, np.inf, [8000, 31000]])
def test_invalid_or_out_of_table_energy_is_rejected(energy):
    with pytest.raises(ValueError):
        get_scattering_factors("Si", energy)


def test_implicit_element_density_and_explicit_density_match():
    e = np.array([8000., 11920.])
    a = effective_electron_density("Au", energies_eV=e, complex=True)
    b = effective_electron_density("Au", pt.Au.density, e, complex=True)
    np.testing.assert_allclose(a, b)


def test_density_validation_also_applies_to_complex_mode():
    with pytest.raises(ValueError, match="mass_density_g_per_cm3"):
        effective_electron_density("H2O", energies_eV=8000, complex=True)
    with pytest.raises(ValueError, match="positive"):
        effective_electron_density("H2O", 0, 8000, complex=True)
