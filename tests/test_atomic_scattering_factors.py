import numpy as np
import pytest
import periodictable as pt

from atomic_scattering_factors.atomic_scattering_factors import (
    parse_formula,
    molecular_weight,
    get_Z_formula,
    get_scattering_factors,
    get_effective_Z,
    get_effective_Z_formula,
    effective_electron_density,
)


# ============================================================================
# Formula parsing
# ============================================================================

class TestParseFormula:

    def test_simple_formula(self):
        elements, counts = parse_formula("H2O")

        assert elements == ["H", "O"]
        np.testing.assert_allclose(counts, [2.0, 1.0])

    def test_formula_without_explicit_counts(self):
        elements, counts = parse_formula("HOH")

        assert elements == ["H", "O"]
        np.testing.assert_allclose(counts, [2.0, 1.0])

    def test_glucose(self):
        elements, counts = parse_formula("C6H12O6")

        assert elements == ["C", "H", "O"]
        np.testing.assert_allclose(counts, [6.0, 12.0, 6.0])

    def test_decimal_stoichiometry(self):
        elements, counts = parse_formula("NaY0.5Gd0.3F4")

        assert elements == ["Na", "Y", "Gd", "F"]
        np.testing.assert_allclose(
            counts,
            [1.0, 0.5, 0.3, 4.0],
        )

    def test_dopant_notation(self):
        elements, counts = parse_formula(
            "NaY0.5Gd0.3F4:Yb0.02,Er0.01"
        )

        assert elements == ["Na", "Y", "Gd", "F", "Yb", "Er"]
        np.testing.assert_allclose(
            counts,
            [1.0, 0.5, 0.3, 4.0, 0.02, 0.01],
        )

    def test_repeated_elements_are_combined(self):
        elements, counts = parse_formula("HOH")

        assert elements == ["H", "O"]
        np.testing.assert_allclose(counts, [2.0, 1.0])

    def test_repeated_elements_are_combined_with_dopants(self):
        elements, counts = parse_formula("H2O:H0.5")

        assert elements == ["H", "O"]
        np.testing.assert_allclose(counts, [2.5, 1.0])

    def test_integer_count(self):
        elements, counts = parse_formula("Si2")

        assert elements == ["Si"]
        np.testing.assert_allclose(counts, [2.0])

    def test_decimal_with_trailing_zero(self):
        elements, counts = parse_formula("H2.0O")

        assert elements == ["H", "O"]
        np.testing.assert_allclose(counts, [2.0, 1.0])

    def test_decimal_without_leading_zero(self):
        elements, counts = parse_formula("H.5O")

        assert elements == ["H", "O"]
        np.testing.assert_allclose(counts, [0.5, 1.0])

    def test_element_symbol_case(self):
        elements, counts = parse_formula("NaCl")

        assert elements == ["Na", "Cl"]
        np.testing.assert_allclose(counts, [1.0, 1.0])

    def test_whitespace_is_stripped(self):
        elements, counts = parse_formula("  H2O  ")

        assert elements == ["H", "O"]
        np.testing.assert_allclose(counts, [2.0, 1.0])

    @pytest.mark.parametrize(
        "formula",
        [
            "",
            " ",
            "H2Oxyz",
            "123",
            "H2O!",
            "H2O@",
            "H2O+",
            "H2O(",
            "H2O)",
            "foo",
        ],
    )
    def test_invalid_formula(self, formula):
        with pytest.raises(ValueError):
            parse_formula(formula)

    def test_non_string_formula(self):
        with pytest.raises(TypeError):
            parse_formula(123)

    @pytest.mark.parametrize(
        "formula",
        [
            "H0O",
            "H-1O",
            "H0.0O",
            "H.0O",
        ],
    )
    def test_non_positive_atom_count(self, formula):
        with pytest.raises(ValueError):
            parse_formula(formula)

    def test_unknown_element(self):
        with pytest.raises(ValueError):
            parse_formula("Xx2")


# ============================================================================
# Molecular weight
# ============================================================================

class TestMolecularWeight:

    def test_water(self):
        mass = molecular_weight("H2O")

        expected = (
            pt.H.mass * 2
            + pt.O.mass
        )

        assert mass == pytest.approx(expected)

    def test_silica(self):
        mass = molecular_weight("SiO2")

        expected = (
            pt.Si.mass
            + 2 * pt.O.mass
        )

        assert mass == pytest.approx(expected)

    def test_glucose(self):
        mass = molecular_weight("C6H12O6")

        expected = (
            6 * pt.C.mass
            + 12 * pt.H.mass
            + 6 * pt.O.mass
        )

        assert mass == pytest.approx(expected)

    def test_decimal_formula(self):
        mass = molecular_weight("NaY0.5Gd0.3F4")

        expected = (
            pt.Na.mass
            + 0.5 * pt.Y.mass
            + 0.3 * pt.Gd.mass
            + 4 * pt.F.mass
        )

        assert mass == pytest.approx(expected)

    def test_dopant_formula(self):
        mass = molecular_weight(
            "NaY0.5Gd0.3F4:Yb0.02,Er0.01"
        )

        expected = (
            pt.Na.mass
            + 0.5 * pt.Y.mass
            + 0.3 * pt.Gd.mass
            + 4 * pt.F.mass
            + 0.02 * pt.Yb.mass
            + 0.01 * pt.Er.mass
        )

        assert mass == pytest.approx(expected)

    def test_invalid_formula(self):
        with pytest.raises(ValueError):
            molecular_weight("H2Oxyz")


# ============================================================================
# Total atomic number
# ============================================================================

class TestGetZFormula:

    def test_water(self):
        # 2 * Z(H) + Z(O) = 2 + 8 = 10
        assert get_Z_formula("H2O") == pytest.approx(10.0)

    def test_silica(self):
        # Z(Si) + 2 * Z(O) = 14 + 16 = 30
        assert get_Z_formula("SiO2") == pytest.approx(30.0)

    def test_glucose(self):
        expected = (
            6 * 6
            + 12 * 1
            + 6 * 8
        )

        assert get_Z_formula("C6H12O6") == pytest.approx(expected)

    def test_decimal_formula(self):
        expected = (
            11
            + 0.5 * 39
            + 0.3 * 64
            + 4 * 9
        )

        assert get_Z_formula("NaY0.5Gd0.3F4") == pytest.approx(expected)

    def test_dopant_formula(self):
        expected = (
            11
            + 0.5 * 39
            + 0.3 * 64
            + 4 * 9
            + 0.02 * 70
            + 0.01 * 68
        )

        assert get_Z_formula(
            "NaY0.5Gd0.3F4:Yb0.02,Er0.01"
        ) == pytest.approx(expected)


# ============================================================================
# Energy handling
# ============================================================================

class TestEnergyHandling:

    def test_python_float_scalar(self):
        result = get_effective_Z("Si", 8000.0)

        assert np.isscalar(result)
        assert np.isfinite(result)

    def test_python_int_scalar(self):
        result = get_effective_Z("Si", 8000)

        assert np.isscalar(result)
        assert np.isfinite(result)

    def test_numpy_float_scalar(self):
        result = get_effective_Z(
            "Si",
            np.float64(8000.0),
        )

        assert np.isscalar(result)
        assert np.isfinite(result)

    def test_numpy_int_scalar(self):
        result = get_effective_Z(
            "Si",
            np.int64(8000),
        )

        assert np.isscalar(result)
        assert np.isfinite(result)

    def test_list_of_energies(self):
        energies = [5000, 8000, 12000]

        result = get_effective_Z("Si", energies)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_tuple_of_energies(self):
        energies = (5000, 8000, 12000)

        result = get_effective_Z("Si", energies)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_numpy_array(self):
        energies = np.array([5000, 8000, 12000])

        result = get_effective_Z("Si", energies)

        assert isinstance(result, np.ndarray)
        assert result.shape == energies.shape

    def test_single_element_array(self):
        energies = np.array([8000])

        result = get_effective_Z("Si", energies)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_multidimensional_array(self):
        energies = np.array([
            [5000, 6000],
            [7000, 8000],
        ])

        result = get_effective_Z("Si", energies)

        assert isinstance(result, np.ndarray)
        assert result.shape == energies.shape

    @pytest.mark.parametrize(
        "energies",
        [
            0,
            -1,
            [0, 8000],
            [-100, 8000],
        ],
    )
    def test_non_positive_energy(self, energies):
        with pytest.raises(ValueError):
            get_effective_Z("Si", energies)

    @pytest.mark.parametrize(
        "energies",
        [
            np.nan,
            np.inf,
            -np.inf,
            [5000, np.nan],
            [5000, np.inf],
        ],
    )
    def test_non_finite_energy(self, energies):
        with pytest.raises(ValueError):
            get_effective_Z("Si", energies)


# ============================================================================
# Scattering factors
# ============================================================================

class TestScatteringFactors:

    def test_scalar(self):
        f1, f2 = get_scattering_factors("Si", 8000)

        assert np.isscalar(f1)
        assert np.isscalar(f2)

        assert np.isfinite(f1)
        assert np.isfinite(f2)

    def test_array(self):
        energies = np.array([5000, 8000, 12000])

        f1, f2 = get_scattering_factors("Si", energies)

        assert isinstance(f1, np.ndarray)
        assert isinstance(f2, np.ndarray)

        assert f1.shape == energies.shape
        assert f2.shape == energies.shape

        assert np.all(np.isfinite(f1))
        assert np.all(np.isfinite(f2))

    def test_multidimensional_array(self):
        energies = np.array([
            [5000, 6000],
            [7000, 8000],
        ])

        f1, f2 = get_scattering_factors("Si", energies)

        assert f1.shape == energies.shape
        assert f2.shape == energies.shape

    def test_factors_are_real(self):
        energies = np.array([5000, 8000, 12000])

        f1, f2 = get_scattering_factors("Si", energies)

        assert np.isrealobj(f1)
        assert np.isrealobj(f2)


# ============================================================================
# Effective Z for an individual element
# ============================================================================

class TestEffectiveZ:

    def test_definition(self):
        energies = np.array([5000, 8000, 12000])

        f1, f2 = get_scattering_factors("Si", energies)

        result = get_effective_Z("Si", energies)

        expected = np.sqrt(f1**2 + f2**2)

        np.testing.assert_allclose(
            result,
            expected,
        )

    def test_scalar_and_array_agree(self):
        energy = 8000

        scalar_result = get_effective_Z(
            "Si",
            energy,
        )

        array_result = get_effective_Z(
            "Si",
            np.array([energy]),
        )

        assert scalar_result == pytest.approx(
            array_result[0]
        )


# ============================================================================
# Effective Z for chemical formulas
# ============================================================================

class TestEffectiveZFormula:

    def test_water_scalar(self):
        result = get_effective_Z_formula(
            "H2O",
            8000,
        )

        assert np.isscalar(result)
        assert np.isfinite(result)

    def test_water_array(self):
        energies = np.array([
            5000,
            8000,
            12000,
        ])

        result = get_effective_Z_formula(
            "H2O",
            energies,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == energies.shape
        assert np.all(np.isfinite(result))

    def test_formula_is_sum_of_elements(self):
        energies = np.array([
            5000,
            8000,
            12000,
        ])

        result = get_effective_Z_formula(
            "H2O",
            energies,
        )

        expected = (
            2 * get_effective_Z("H", energies)
            + get_effective_Z("O", energies)
        )

        np.testing.assert_allclose(
            result,
            expected,
        )

    def test_repeated_formula_is_equivalent(self):
        energies = np.array([
            5000,
            8000,
            12000,
        ])

        result_1 = get_effective_Z_formula(
            "H2O",
            energies,
        )

        result_2 = get_effective_Z_formula(
            "HOH",
            energies,
        )

        np.testing.assert_allclose(
            result_1,
            result_2,
        )

    def test_scalar_and_array_agree(self):
        energy = 8000

        scalar_result = get_effective_Z_formula(
            "H2O",
            energy,
        )

        array_result = get_effective_Z_formula(
            "H2O",
            np.array([energy]),
        )

        assert scalar_result == pytest.approx(
            array_result[0]
        )

    def test_multidimensional_input(self):
        energies = np.array([
            [5000, 6000],
            [7000, 8000],
        ])

        result = get_effective_Z_formula(
            "H2O",
            energies,
        )

        assert result.shape == energies.shape

    def test_decimal_formula(self):
        energies = np.array([
            5000,
            8000,
            12000,
        ])

        result = get_effective_Z_formula(
            "NaY0.5Gd0.3F4",
            energies,
        )

        expected = (
            get_effective_Z("Na", energies)
            + 0.5 * get_effective_Z("Y", energies)
            + 0.3 * get_effective_Z("Gd", energies)
            + 4 * get_effective_Z("F", energies)
        )

        np.testing.assert_allclose(
            result,
            expected,
        )

    def test_dopant_formula(self):
        energies = np.array([
            5000,
            8000,
            12000,
        ])

        result = get_effective_Z_formula(
            "NaY0.5Gd0.3F4:Yb0.02,Er0.01",
            energies,
        )

        expected = (
            get_effective_Z("Na", energies)
            + 0.5 * get_effective_Z("Y", energies)
            + 0.3 * get_effective_Z("Gd", energies)
            + 4 * get_effective_Z("F", energies)
            + 0.02 * get_effective_Z("Yb", energies)
            + 0.01 * get_effective_Z("Er", energies)
        )

        np.testing.assert_allclose(
            result,
            expected,
        )


# ============================================================================
# Effective electron density
# ============================================================================

class TestEffectiveElectronDensity:

    def test_pure_element_without_density(self):
        result = effective_electron_density(
            "Si",
            energies_eV=8000,
        )

        assert np.isscalar(result)
        assert np.isfinite(result)
        assert result > 0

    def test_compound_requires_density(self):
        with pytest.raises(
            ValueError,
            match="mass_density_g_per_cm3",
        ):
            effective_electron_density(
                "SiO2",
                energies_eV=8000,
            )

    def test_compound_with_density(self):
        result = effective_electron_density(
            "SiO2",
            mass_density_g_per_cm3=2.2,
            energies_eV=8000,
        )

        assert np.isscalar(result)
        assert np.isfinite(result)
        assert result > 0

    def test_array_energy(self):
        energies = np.array([
            5000,
            8000,
            12000,
        ])

        result = effective_electron_density(
            "SiO2",
            mass_density_g_per_cm3=2.2,
            energies_eV=energies,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == energies.shape
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    def test_multidimensional_energy(self):
        energies = np.array([
            [5000, 6000],
            [7000, 8000],
        ])

        result = effective_electron_density(
            "SiO2",
            mass_density_g_per_cm3=2.2,
            energies_eV=energies,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == energies.shape

    def test_scalar_and_array_agree(self):
        energy = 8000

        scalar_result = effective_electron_density(
            "SiO2",
            mass_density_g_per_cm3=2.2,
            energies_eV=energy,
        )

        array_result = effective_electron_density(
            "SiO2",
            mass_density_g_per_cm3=2.2,
            energies_eV=np.array([energy]),
        )

        assert scalar_result == pytest.approx(
            array_result[0]
        )

    def test_density_scaling(self):
        energies = np.array([8000])

        rho_1 = effective_electron_density(
            "SiO2",
            mass_density_g_per_cm3=1.0,
            energies_eV=energies,
        )

        rho_2 = effective_electron_density(
            "SiO2",
            mass_density_g_per_cm3=2.0,
            energies_eV=energies,
        )

        np.testing.assert_allclose(
            rho_2,
            2 * rho_1,
        )

    def test_pure_element_density_matches_explicit_density(self):
        energy = 8000

        si_density = pt.Si.density

        result_implicit = effective_electron_density(
            "Si",
            energies_eV=energy,
        )

        result_explicit = effective_electron_density(
            "Si",
            mass_density_g_per_cm3=si_density,
            energies_eV=energy,
        )

        assert result_implicit == pytest.approx(
            result_explicit,
        )

    def test_multicomponent_compound_requires_density(self):
        with pytest.raises(ValueError):
            effective_electron_density(
                "NaY0.5Gd0.3F4",
                energies_eV=8000,
            )

    def test_doped_compound_with_density(self):
        result = effective_electron_density(
            "NaY0.5Gd0.3F4:Yb0.02,Er0.01",
            mass_density_g_per_cm3=4.5,
            energies_eV=8000,
        )

        assert np.isscalar(result)
        assert np.isfinite(result)
        assert result > 0


# ============================================================================
# Density validation
# ============================================================================

class TestDensityValidation:

    @pytest.mark.parametrize(
        "density",
        [
            0,
            -1,
            -0.1,
        ],
    )
    def test_non_positive_density(self, density):
        with pytest.raises(ValueError):
            effective_electron_density(
                "SiO2",
                mass_density_g_per_cm3=density,
                energies_eV=8000,
            )

    def test_zero_density(self):
        """
        This test documents the desired behavior that a zero density
        should be rejected rather than returning zero electron density.
        """
        with pytest.raises(ValueError):
            effective_electron_density(
                "SiO2",
                mass_density_g_per_cm3=0.0,
                energies_eV=8000,
            )


# ============================================================================
# Invalid input
# ============================================================================

class TestInvalidInput:

    def test_invalid_formula_for_effective_Z(self):
        with pytest.raises(ValueError):
            get_effective_Z_formula(
                "H2Oxyz",
                8000,
            )

    def test_invalid_formula_for_density(self):
        with pytest.raises(ValueError):
            effective_electron_density(
                "H2Oxyz",
                mass_density_g_per_cm3=1.0,
                energies_eV=8000,
            )

    def test_unknown_element_for_scattering_factors(self):
        with pytest.raises(Exception):
            get_scattering_factors(
                "Xx",
                8000,
            )

    def test_unknown_element_for_effective_Z(self):
        with pytest.raises(Exception):
            get_effective_Z(
                "Xx",
                8000,
            )