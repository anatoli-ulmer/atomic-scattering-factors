"""Forward X-ray scattering factors and effective electron densities.

All energies are in eV; all electron densities are in electrons/m³.
The complex convention is f = f1 + 1j*f2, consistent with periodictable.
"""

import re
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import periodictable as pt
from periodictable.constants import avogadro_number


def _prepare_energies(energies_eV):
    """Convert energies to an array and require finite, positive energies."""
    energies_eV = np.asarray(energies_eV, dtype=float)
    if np.any(~np.isfinite(energies_eV)):
        raise ValueError("energies_eV must contain only finite values.")
    if np.any(energies_eV <= 0):
        raise ValueError("energies_eV must be positive.")
    return energies_eV


def _restore_scalar(value, input_was_scalar):
    """Return a Python float for scalar input, or an ndarray otherwise."""
    if input_was_scalar:
        return float(np.asarray(value))
    return np.asarray(value)


def _restore_complex(value, input_was_scalar):
    """Return a Python complex for scalar input, or an ndarray otherwise."""
    if input_was_scalar:
        return complex(np.asarray(value))
    return np.asarray(value, dtype=complex)


@lru_cache(maxsize=128)
def _get_pchip_interpolators(element):
    """Build cached shape-preserving cubic interpolators from Henke tables.

    SciPy is imported lazily so default calculations have no SciPy dependency.
    Source values and tabulated energy grid are unchanged.
    """
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError as exc:
        raise ImportError(
            "PCHIP interpolation requires the optional SciPy dependency. "
            "Install it with 'pip install atomic-scattering-factors[spline]' "
            "or use interpolation='default'."
        ) from exc

    table = pt.elements.symbol(element).xray.sftable
    if table is None:
        raise ValueError(f"No X-ray scattering factors available for {element!r}.")

    # periodictable's sftable energies are in keV. In its table loader,
    # missing f1 entries (originally -9999) are converted into NaNs.
    table = np.asarray(table, dtype=float)
    x = table[0] * 1000.0  # eV
    interpolators = []
    for y in (table[1], table[2]):
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            raise ValueError(f"Insufficient scattering-factor data for {element!r}.")
        # Keep individual valid ranges for f1 and f2; never extrapolate.
        interpolators.append(PchipInterpolator(x[valid], y[valid], extrapolate=False))
    return tuple(interpolators)


def get_scattering_factors(element, energies_eV, *, interpolation="default"):
    """Return atomic forward factors f1 and f2 at photon energies in eV.

    Parameters
    ----------
    element : str
        Atomic symbol, e.g. ``'Au'``.
    energies_eV : float or array-like
        Photon energies in eV; scalar or arrays of arbitrary shape.
    interpolation : {"default", "pchip", "spline"}, optional
        ``"default"`` preserves periodictable's native linear f1 and
        log-log f2 interpolation. ``"pchip"`` or ``"spline"`` uses
        SciPy's shape-preserving piecewise cubic Hermite interpolation for
        both f1 and f2 on the *same* periodictable source tables. SciPy is
        optional and loaded only for the spline modes.

    Notes
    -----
    All interpolation is within the available atomic tables, without
    extrapolation. Piecewise cubic interpolation may differ near absorption
    edges but does not add experimentally measured near-edge structure.
    """
    if interpolation not in ("default", "pchip", "spline"):
        raise ValueError("interpolation must be 'default', 'pchip', or 'spline'.")
    input_was_scalar = np.ndim(energies_eV) == 0
    energies = _prepare_energies(energies_eV)
    try:
        atom = pt.elements.symbol(element)
    except Exception as exc:
        raise ValueError(f"Unknown element symbol: {element!r}") from exc

    if interpolation == "default":
        # periodictable wants energies in keV and expects a flat vector.
        f1, f2 = atom.xray.scattering_factors(energy=energies.reshape(-1) / 1000.0)
        if f1 is None or f2 is None:
            raise ValueError(f"No X-ray scattering factors available for {element!r}.")
    else:
        f1_interp, f2_interp = _get_pchip_interpolators(atom.symbol)
        f1, f2 = f1_interp(energies.reshape(-1)), f2_interp(energies.reshape(-1))

    f1 = np.asarray(f1, dtype=float).reshape(energies.shape)
    f2 = np.asarray(f2, dtype=float).reshape(energies.shape)
    if not (np.all(np.isfinite(f1)) and np.all(np.isfinite(f2))):
        raise ValueError(
            f"Scattering factors for {element!r} are unavailable at one or "
            "more supplied energies; check the tabulated energy range."
        )
    return _restore_scalar(f1, input_was_scalar), _restore_scalar(f2, input_was_scalar)


def get_complex_scattering_factor(element, energies_eV, *, interpolation="default"):
    """Return complex atomic forward factor f1 + 1j*f2."""
    input_was_scalar = np.ndim(energies_eV) == 0
    f1, f2 = get_scattering_factors(element, energies_eV, interpolation=interpolation)
    return _restore_complex(np.asarray(f1) + 1j * np.asarray(f2), input_was_scalar)


def get_effective_Z(element, energies_eV, *, interpolation="default"):
    """Legacy effective atomic number: ``abs(f1 + 1j*f2)``."""
    input_was_scalar = np.ndim(energies_eV) == 0
    f1, f2 = get_scattering_factors(element, energies_eV, interpolation=interpolation)
    return _restore_scalar(np.hypot(f1, f2), input_was_scalar)


_FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(?:(\d+(?:\.\d*)?|\.\d+))?")


def parse_formula(formula):
    """Parse a composition with fractional counts and optional colon dopants.

    Examples: ``'H2O'``, ``'HOH'``, ``'NaY0.5Gd0.3F4:Yb0.02,Er0.01'``.
    Parentheses are not supported (unchanged from v0.2.0).
    """
    if not isinstance(formula, str):
        raise TypeError("formula must be a string.")
    formula = formula.strip()
    if not formula:
        raise ValueError("formula must not be empty.")
    formula = formula.replace(":", ",").replace(",", "")
    counts = OrderedDict()
    position = 0
    for match in _FORMULA_PATTERN.finditer(formula):
        if match.start() != position:
            raise ValueError(f"Invalid chemical formula: {formula!r}")
        element = match.group(1)
        count_str = match.group(2)
        try:
            pt.elements.symbol(element)
        except Exception as exc:
            raise ValueError(f"Unknown element symbol: {element!r}") from exc
        count = float(count_str) if count_str is not None else 1.0
        if count <= 0:
            raise ValueError(f"Atom count must be positive for element {element!r}.")
        counts[element] = counts.get(element, 0.0) + count
        position = match.end()
    if position != len(formula):
        raise ValueError(f"Invalid chemical formula: {formula!r}")
    if not counts:
        raise ValueError(f"Could not parse chemical formula: {formula!r}")
    return list(counts.keys()), list(counts.values())


def molecular_weight(formula):
    """Return the molar mass for a formula in g/mol."""
    elements, counts = parse_formula(formula)
    return sum(pt.elements.symbol(el).mass * n for el, n in zip(elements, counts))


def get_effective_Z_formula(chemical_formula, energies_eV, *, interpolation="default"):
    """Legacy formula descriptor ``sum(N_i * abs(f_i))``.

    CAUTION: this is *not* a coherent molecular scattering amplitude and
    must not be used as the scattering contrast for ASAXS. Kept for backward
    compatibility. Use ``get_scattering_factor_formula`` for ASAXS.
    """
    input_was_scalar = np.ndim(energies_eV) == 0
    energies = _prepare_energies(energies_eV)
    elements, counts = parse_formula(chemical_formula)
    result = np.zeros_like(energies, dtype=float)
    for element, count in zip(elements, counts):
        result += count * get_effective_Z(element, energies, interpolation=interpolation)
    return _restore_scalar(result, input_was_scalar)


def get_scattering_factor_formula(chemical_formula, energies_eV, *, interpolation="default"):
    """Return the *coherent complex* forward factor of a formula unit.

    F(E) = sum_i N_i [f1_i(E) + 1j*f2_i(E)].
    This is the correct composition-weighted factor for an ASAXS contrast.
    """
    input_was_scalar = np.ndim(energies_eV) == 0
    energies = _prepare_energies(energies_eV)
    elements, counts = parse_formula(chemical_formula)
    result = np.zeros_like(energies, dtype=complex)
    for element, count in zip(elements, counts):
        result += count * get_complex_scattering_factor(element, energies, interpolation=interpolation)
    return _restore_complex(result, input_was_scalar)


def get_Z_formula(chemical_formula):
    """Return the total proton count per formula unit."""
    elements, counts = parse_formula(chemical_formula)
    return sum(count * pt.elements.symbol(element).number for element, count in zip(elements, counts))


def effective_electron_density(
    sample_str,
    mass_density_g_per_cm3=None,
    energies_eV=8000,
    *,
    complex=False,
    interpolation="default",
):
    """Return an effective electron density in electrons/m³.

    When ``complex=True`` (recommended for ASAXS), return

        rho_tilde(E) = n * sum_i N_i * [f1_i(E) + 1j*f2_i(E)]

    with n the number density of formula units (1/m³). The corresponding
    complex scattering length density is r_e * rho_tilde (1/m²), with r_e
    the classical electron radius.

    When ``complex=False`` (default), retain the v0.2 legacy definition
    n*sum_i N_i*abs(f_i) for backward compatibility. This legacy result
    is NOT generally equal to abs(rho_tilde) or real(rho_tilde), and must
    NOT be used to compute physical ASAXS contrast. For the real coherent
    electron density use ``effective_electron_density(..., complex=True).real``.

    Parameters
    ----------
    sample_str : str
        Formula, e.g. ``'Si'`` or ``'SiO2'``.
    mass_density_g_per_cm3 : float, optional
        Bulk mass density in g/cm³. Required for compounds. If omitted for
        a pure element, its periodictable tabulated density is used.
    energies_eV : float or array-like
        Photon energies in eV, scalar or array of any shape.
    complex : bool, optional
        Select coherent complex electron density. Defaults to False for
        backward compatibility with the v0.2 API.

    interpolation : {"default", "pchip", "spline"}, optional
        Interpolation for each atomic factor. SciPy is needed only for
        spline/PCHIP; default behavior is unchanged.

    Returns
    -------
    float, complex, or ndarray
        Effective electron density in electrons/m³, with a scalar or an
        array shaped like ``energies_eV``.
    """
    elements, _ = parse_formula(sample_str)
    if mass_density_g_per_cm3 is None:
        if len(elements) != 1:
            raise ValueError("mass_density_g_per_cm3 must be provided for chemical compounds.")
        mass_density_g_per_cm3 = pt.elements.symbol(elements[0]).density
    if mass_density_g_per_cm3 is None or not np.isfinite(mass_density_g_per_cm3):
        raise ValueError("mass_density_g_per_cm3 must be finite.")
    if mass_density_g_per_cm3 <= 0:
        raise ValueError("mass_density_g_per_cm3 must be positive.")

    # [g/cm³] -> [g/m³]; divided by [g/mol], multiplied by [1/mol].
    number_density = (mass_density_g_per_cm3 * 1e6 / molecular_weight(sample_str)) * avogadro_number
    if complex:
        return number_density * get_scattering_factor_formula(
            sample_str, energies_eV, interpolation=interpolation
        )
    return number_density * get_effective_Z_formula(
        sample_str, energies_eV, interpolation=interpolation
    )
