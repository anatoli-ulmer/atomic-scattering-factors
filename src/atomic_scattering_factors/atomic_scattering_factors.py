import re
from collections import OrderedDict

from xraylabtool.calculators.core import create_scattering_factor_interpolators

import periodictable as pt
import numpy as np
import scipy.constants as const


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _prepare_energies(energies_eV):
    """
    Convert energies to a NumPy array and validate the values.

    Parameters
    ----------
    energies_eV : float or array-like
        X-ray photon energy or energies in eV.

    Returns
    -------
    numpy.ndarray
        Energies as a NumPy array.
    """
    energies_eV = np.asarray(energies_eV, dtype=float)

    if np.any(~np.isfinite(energies_eV)):
        raise ValueError("energies_eV must contain only finite values.")

    if np.any(energies_eV <= 0):
        raise ValueError("energies_eV must be positive.")

    return energies_eV


def _restore_scalar(value, input_was_scalar):
    """Return a scalar if the original input was scalar."""
    if input_was_scalar:
        return float(np.asarray(value))

    return np.asarray(value)


# ---------------------------------------------------------------------------
# Scattering factors
# ---------------------------------------------------------------------------

def get_scattering_factors(element, energies_eV):
    """
    Get the atomic scattering factors f1 and f2 for a given element
    and X-ray energies.

    Parameters
    ----------
    element : str
        Element symbol, e.g. 'H', 'C', 'O'.
    energies_eV : float or array-like
        X-ray photon energy or energies in eV.

    Returns
    -------
    f1 : float or numpy.ndarray
        Scattering factor f1. A scalar is returned for scalar input;
        otherwise a NumPy array with the same shape as `energies_eV`.
    f2 : float or numpy.ndarray
        Scattering factor f2. A scalar is returned for scalar input;
        otherwise a NumPy array with the same shape as `energies_eV`.
    """
    input_was_scalar = np.ndim(energies_eV) == 0
    energies_eV = _prepare_energies(energies_eV)

    f1_interp, f2_interp = create_scattering_factor_interpolators(element)

    f1 = f1_interp(energies_eV)
    f2 = f2_interp(energies_eV)

    return (
        _restore_scalar(f1, input_was_scalar),
        _restore_scalar(f2, input_was_scalar),
    )


def get_effective_Z(element, energies_eV):
    """
    Calculate the effective atomic number for a given element
    and X-ray energies.

    Parameters
    ----------
    element : str
        Element symbol, e.g. 'H', 'C', 'O'.
    energies_eV : float or array-like
        X-ray photon energy or energies in eV.

    Returns
    -------
    float or numpy.ndarray
        Effective atomic number. A scalar is returned for scalar input;
        otherwise a NumPy array with the same shape as `energies_eV`.
    """
    input_was_scalar = np.ndim(energies_eV) == 0

    f1, f2 = get_scattering_factors(element, energies_eV)

    effective_Z = np.sqrt(f1**2 + f2**2)

    return _restore_scalar(effective_Z, input_was_scalar)


# ---------------------------------------------------------------------------
# Chemical formula parsing
# ---------------------------------------------------------------------------

_FORMULA_PATTERN = re.compile(
    r"([A-Z][a-z]?)(?:(\d+(?:\.\d*)?|\.\d+))?"
)


def parse_formula(formula):
    """
    Parse a chemical formula and return elements and atom counts.

    Supports decimal stoichiometries and optional dopant notation
    separated by a colon.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. 'H2O', 'NaY0.5Gd0.3F4', or
        'NaY0.5Gd0.3F4:Yb0.02,Er0.01'.

    Returns
    -------
    elements : list of str
        Element symbols.

    num_atoms : list of float
        Corresponding atom counts.

    Raises
    ------
    TypeError
        If `formula` is not a string.

    ValueError
        If the formula is empty, contains invalid syntax, contains an
        unknown element, or contains a non-positive atom count.
    """
    if not isinstance(formula, str):
        raise TypeError("formula must be a string.")

    formula = formula.strip()

    if not formula:
        raise ValueError("formula must not be empty.")

    # Split host and dopants at ':'.
    # Commas are simply treated as separators.
    formula = formula.replace(":", ",")
    formula = formula.replace(",", "")

    counts = OrderedDict()
    position = 0

    for match in _FORMULA_PATTERN.finditer(formula):

        # Make sure there is no unparsed text between valid tokens.
        if match.start() != position:
            raise ValueError(
                f"Invalid chemical formula: {formula!r}"
            )

        element = match.group(1)
        count_str = match.group(2)

        # Validate the element symbol.
        try:
            pt.elements.symbol(element)
        except Exception as exc:
            raise ValueError(
                f"Unknown element symbol: {element!r}"
            ) from exc

        if count_str is None:
            count = 1.0
        else:
            count = float(count_str)

        if count <= 0:
            raise ValueError(
                f"Atom count must be positive for element {element!r}."
            )

        counts[element] = counts.get(element, 0.0) + count

        position = match.end()

    # Detect trailing or otherwise invalid text.
    if position != len(formula):
        raise ValueError(
            f"Invalid chemical formula: {formula!r}"
        )

    if not counts:
        raise ValueError(
            f"Could not parse chemical formula: {formula!r}"
        )

    return list(counts.keys()), list(counts.values())


# ---------------------------------------------------------------------------
# Molecular properties
# ---------------------------------------------------------------------------

def molecular_weight(formula):
    """
    Compute the molecular mass in g/mol.

    Parameters
    ----------
    formula : str
        Chemical formula.

    Returns
    -------
    float
        Molecular mass in g/mol.
    """
    elements, counts = parse_formula(formula)

    mass = 0.0

    for element, count in zip(elements, counts):
        mass += pt.elements.symbol(element).mass * count

    return mass


def get_effective_Z_formula(chemical_formula, energies_eV):
    """
    Calculate the effective atomic number for a chemical formula.

    Parameters
    ----------
    chemical_formula : str
        Chemical formula, e.g. 'H2O', 'HOH', or 'C6H12O6'.

    energies_eV : float or array-like
        X-ray photon energy or energies in eV.

    Returns
    -------
    float or numpy.ndarray
        Effective atomic number. A scalar is returned for scalar input;
        otherwise a NumPy array with the same shape as `energies_eV`.
    """
    input_was_scalar = np.ndim(energies_eV) == 0
    energies_eV = _prepare_energies(energies_eV)

    elements, num_atoms = parse_formula(chemical_formula)

    effective_Z = np.zeros_like(energies_eV, dtype=float)

    for element, count in zip(elements, num_atoms):
        effective_Z += count * get_effective_Z(element, energies_eV)

    return _restore_scalar(effective_Z, input_was_scalar)


def get_Z_formula(chemical_formula):
    """
    Calculate the total atomic number for a chemical formula.

    Parameters
    ----------
    chemical_formula : str
        Chemical formula, e.g. 'H2O', 'HOH', or 'C6H12O6'.

    Returns
    -------
    float
        Total atomic number per formula unit.
    """
    elements, num_atoms = parse_formula(chemical_formula)

    return sum(
        count * pt.elements.symbol(element).number
        for element, count in zip(elements, num_atoms)
    )


# ---------------------------------------------------------------------------
# Effective electron density
# ---------------------------------------------------------------------------

def effective_electron_density(
    sample_str,
    mass_density_g_per_cm3=None,
    energies_eV=8000,
):
    """
    Calculate the effective electron density of a material.

    The effective electron density is calculated from the molecular number
    density and the energy-dependent effective atomic number:

        rho_e(E) = n * Z_eff(E)

    Parameters
    ----------
    sample_str : str
        Chemical formula of the material, e.g. 'H2O', 'SiO2', or
        'C6H12O6'. For a pure element, provide its element symbol,
        e.g. 'Si' or 'Au'.

    mass_density_g_per_cm3 : float, optional
        Mass density of the material in g/cm^3. This parameter is required
        for chemical compounds. For a pure element, it may be omitted,
        in which case the tabulated elemental density is used.

    energies_eV : float or array-like
        X-ray photon energy or energies in eV.

    Returns
    -------
    float or numpy.ndarray
        Effective electron density in electrons/m^3. A scalar is returned
        for scalar input; otherwise a NumPy array with the same shape as
        `energies_eV`.

    Raises
    ------
    ValueError
        If `mass_density_g_per_cm3` is omitted for a chemical compound.
    """
    elements, num_atoms = parse_formula(sample_str)

    if mass_density_g_per_cm3 is None:
        # A density can only be obtained automatically for a pure element.
        if len(elements) != 1:
            raise ValueError(
                "mass_density_g_per_cm3 must be provided for chemical compounds."
            )

        mass_density_g_per_cm3 = pt.elements.symbol(elements[0]).density

    if not np.isfinite(mass_density_g_per_cm3):
        raise ValueError(
            "mass_density_g_per_cm3 must be finite."
        )

    if mass_density_g_per_cm3 <= 0:
        raise ValueError(
            "mass_density_g_per_cm3 must be positive."
        )

    mass_density = mass_density_g_per_cm3 * 1e3  # kg/m^3
    mass = molecular_weight(sample_str) * const.u  # kg

    number_density = mass_density / mass  # formula units/m^3

    Z_eff = get_effective_Z_formula(sample_str, energies_eV)

    return number_density * Z_eff