# Atomic Scattering Factors

A lightweight Python package to calculate energy-dependent X-ray **atomic forward scattering factors** and **real or complex effective electron densities** for SAXS and anomalous SAXS (ASAXS).

All energies supplied to the API are in **eV**, and effective electron densities are returned in **electrons/m³**.

## Features

- Retrieve energy-dependent atomic scattering factors, `f1(E)` and `f2(E)`, or the complex factor `f1(E) + 1j*f2(E)`.
- Calculate coherent, composition-weighted complex scattering factors for chemical formulas.
- Calculate real-valued legacy or complex effective electron densities, with explicit particle–solvent contrasts for ASAXS.
- Accept scalar energies, arrays, and multidimensional NumPy arrays.
- Parse formulas with fractional stoichiometries, repeated elements, and optional colon-separated dopant notation.

## Installation

Requires Python >= 3.12, NumPy >= 1.26, and `periodictable` >= 2.1.0. No NumPy-major-version upper bound is imposed by this package.

```bash
git clone https://github.com/anatoli-ulmer/atomic-scattering-factors.git
cd atomic-scattering-factors
python -m pip install -e .
```

To check what an installation would change **without modifying the environment**:

```bash
python -m pip install --dry-run -e .
```

If the required dependencies are already installed and compatible, `python -m pip install --no-deps -e .` installs only this project. Note that `--no-deps` does not verify compatibility.

## Usage

### Atomic scattering factors

```python
import numpy as np
from atomic_scattering_factors import get_scattering_factors, get_complex_scattering_factor

energies = np.linspace(8000, 14000, 300)  # eV
f1, f2 = get_scattering_factors("Au", energies)
f = get_complex_scattering_factor("Au", energies)
assert np.allclose(f, f1 + 1j*f2)
```

The complex convention is $f(E)=f_1(E)+i f_2(E)$. These dimensionless factors describe **forward scattering**, approximately $q=0$; a complete finite-$q$ atomistic form factor also requires the momentum-transfer dependence.

### Coherent factor for a chemical formula

```python
from atomic_scattering_factors import get_scattering_factor_formula

f_silica = get_scattering_factor_formula("SiO2", energies)
# f_silica(E) = f_Si(E) + 2*f_O(E)
```

The complex contributions are summed **before** taking a magnitude. This is different from the historical `get_effective_Z_formula()`, which sums individual magnitudes.

### Complex effective electron density

```python
from atomic_scattering_factors import effective_electron_density

rho_gold = effective_electron_density("Au", energies_eV=energies, complex=True)
rho_water = effective_electron_density(
    "H2O", mass_density_g_per_cm3=1.0, energies_eV=energies, complex=True
)

rho_real = rho_gold.real
rho_imag = rho_gold.imag
rho_absolute = np.abs(rho_gold)
rho_gold_in_electrons_per_A3 = rho_gold * 1e-30
```

The package uses the tabulated elemental mass density for a pure element when no density is specified. For a chemical compound, provide its bulk mass density in **g/cm³**; the chemical formula alone does not determine this quantity.

### ASAXS particle–solvent contrast

```python
delta_rho = rho_gold - rho_water
contrast_squared = np.abs(delta_rho)**2
# For a homogeneous particle: I(q,E) is proportional to V**2 * P(q) * contrast_squared
```

This yields a contrast term, **not a full intensity model**: concentration, structure factor, instrumental effects, and any relevant additional physical contributions must be addressed separately.

## Definitions and units

For a chemical formula with $N_j$ atoms of element $j$ per formula unit:

$$
F(E)=\sum_j N_j\,[f_{1,j}(E)+i f_{2,j}(E)] .
$$

The coherent effective electron density is

$$
\widetilde{\rho}_e(E)=n F(E),\qquad n=\frac{\rho_m N_A}{M},
$$

where $n$ is the number density of formula units, $\rho_m$ is mass density, $N_A$ is Avogadro's constant, and $M$ is molar mass (using consistent units). The function returns $\widetilde{\rho}_e$ in **electrons/m³**.

For a particle in a solvent with potentially complex electron density:

$$
\Delta\widetilde{\rho}_e(E)=\widetilde{\rho}_{e,\mathrm{particle}}(E)-\widetilde{\rho}_{e,\mathrm{solvent}}(E).
$$

The complex X-ray scattering-length density is related by $\widetilde{\rho}_{\mathrm{SLD}}=r_e\widetilde{\rho}_e$, with $r_e$ the classical electron radius (giving SLD in m⁻²). Other libraries may use different imaginary-part sign conventions.

### Backward compatibility

The default `effective_electron_density(..., complex=False)` maintains the earlier magnitude-based definition:

$$
\rho_{e,\mathrm{legacy}}(E)=n\sum_j N_j\,\left|f_j(E)\right|.
$$

**This is not, in general, the coherent electron-density magnitude** $|\widetilde{\rho}_e(E)|$, because $\sum_j N_j |f_j|$ and $|\sum_j N_j f_j|$ are different for compounds. For SAXS and ASAXS calculations use `complex=True` and, if needed, apply `np.abs()` **after** the coherent summation. The legacy functions `get_effective_Z()` and `get_effective_Z_formula()` remain available for compatibility.

## API overview

| Function | Description |
| --- | --- |
| `get_scattering_factors(element, energies_eV)` | Return atomic `f1` and `f2` |
| `get_complex_scattering_factor(element, energies_eV)` | Return `f1 + 1j*f2` |
| `get_scattering_factor_formula(formula, energies_eV)` | Sum coherent complex factors over a formula |
| `effective_electron_density(formula, mass_density_g_per_cm3=None, energies_eV=8000, *, complex=False)` | Legacy or complex effective electron density |
| `get_effective_Z(element, energies_eV)` | Legacy atomic factor magnitude |
| `get_effective_Z_formula(formula, energies_eV)` | Legacy sum of atomic magnitudes |
| `parse_formula(formula)` | Return elemental symbols and stoichiometries |
| `molecular_weight(formula)` | Return molar mass in g/mol |
| `get_Z_formula(formula)` | Return total atomic number per formula unit |

Examples of accepted formula notation: `H2O`, `HOH`, `SiO2`, `NaY0.5Gd0.3F4`, and `NaY0.5Gd0.3F4:Yb0.02,Er0.01`. Parenthesized formulas such as `Ca(OH)2` are not currently supported.

## Scattering-factor data and ASAXS caveats

The `periodictable` backend uses tabulated Henke/CXRO scattering factors. Photon energies passed in **eV** are converted to the **keV** expected by `periodictable`. Its `f1` interpolation is linear in energy, while its `f2` interpolation is log-log. Unavailable factors or energies outside the tabulated range result in a `ValueError`.

Results can differ slightly from the older `xraylabtool` backend because of **interpolation differences** and, for some elements, differences in the underlying tables. Around absorption edges these differences deserve special attention. Atomic tabulations do not necessarily capture chemical shifts or X-ray absorption fine structure in a particular experimental sample. For quantitative near-edge ASAXS, experimentally determined absorption spectra and a Kramers–Kronig treatment of the dispersive correction may be required. This package does not currently load custom experimental factor tables.

## Testing

```bash
python -m pip install -e '.[test]'
python -m pip check
python -m pytest -q
```

The test suite is under `tests/`. For a scientific comparison with the former implementation, compare the real and imaginary atomic factors separately; also distinguish changes due to the backend from the different mathematics of the legacy and coherent density definitions.

## License

See [LICENSE](LICENSE).

## Data sources and acknowledgments

Atomic X-ray scattering factors are obtained using the
`periodictable` Python package, which uses tabulated data
from the Center for X-Ray Optics, Lawrence Berkeley
National Laboratory.

Reference:
B. L. Henke, E. M. Gullikson, and J. C. Davis,
Atomic Data and Nuclear Data Tables 54, 181–342 (1993).