# Atomic Scattering Factors

A Python package for calculating atomic X-ray scattering factors and
energy-dependent effective electron densities of materials.

The package provides utilities for:

- calculating atomic scattering factors `f1` and `f2`
- calculating an energy-dependent effective atomic number
- parsing chemical formulas with fractional stoichiometries
- calculating molecular masses
- calculating the total atomic number of a chemical formula
- calculating effective electron densities
- handling both scalar and NumPy-array X-ray energies

## Features

### Atomic scattering factors

For a given element and X-ray photon energy, the package returns the
real and imaginary parts of the atomic scattering factor:

\[
f(E) = f_1(E) + i f_2(E)
\]

```python
from atomic_scattering_factors import get_scattering_factors

f1, f2 = get_scattering_factors("Si", 8000)
````

Multiple energies can be supplied as a NumPy array:

```python
import numpy as np

energies = np.array([5000, 6000, 7000, 8000])

f1, f2 = get_scattering_factors("Si", energies)
```

The returned arrays have the same shape as the input energy array.

---

## Effective atomic number

The package defines an energy-dependent effective atomic number as

\[
Z_\mathrm{eff}(E) =
\sqrt{f_1(E)^2 + f_2(E)^2}.
\]

For a single element:

```python
from atomic_scattering_factors import get_effective_Z

Z_eff = get_effective_Z("Si", 8000)
```

or for multiple energies:

```python
energies = np.array([5000, 6000, 7000, 8000])

Z_eff = get_effective_Z("Si", energies)
```

---

## Chemical formulas

Chemical formulas can contain integer and fractional stoichiometries.

For example:

```python
from atomic_scattering_factors import parse_formula

elements, counts = parse_formula("NaY0.5Gd0.3F4")

print(elements)
# ['Na', 'Y', 'Gd', 'F']

print(counts)
# [1.0, 0.5, 0.3, 4.0]
```

Dopants can be specified after a colon:

```python
elements, counts = parse_formula(
    "NaY0.5Gd0.3F4:Yb0.02,Er0.01"
)
```

The parser combines repeated elements automatically.

For example:

```python
parse_formula("HOH")
```

returns the equivalent stoichiometry of:

```text
H2O
```

### Supported notation

Examples of supported formulas include:

```text
H2O
SiO2
C6H12O6
NaY0.5Gd0.3F4
NaY0.5Gd0.3F4:Yb0.02,Er0.01
```

Parenthesized formulas such as `Ca(OH)2` are currently not supported.

---

## Effective atomic number of a compound

The effective atomic number of a chemical formula is calculated by summing
the energy-dependent effective atomic numbers of its constituent elements,
weighted by their stoichiometric coefficients:

\[
Z_\mathrm{eff,formula}(E)
=========================

\sum_i n_i Z_{\mathrm{eff},i}(E).
\]

For example:

```python
from atomic_scattering_factors import get_effective_Z_formula

energies = np.array([5000, 6000, 7000, 8000])

Z_eff = get_effective_Z_formula(
    "SiO2",
    energies,
)
```

A scalar energy returns a scalar:

```python
Z_eff = get_effective_Z_formula("SiO2", 8000)
```

An array of energies returns a NumPy array with the same shape:

```python
energies = np.array([
    [5000, 6000],
    [7000, 8000],
])

Z_eff = get_effective_Z_formula("SiO2", energies)

print(Z_eff.shape)
# (2, 2)
```

---

## Molecular weight

Molecular masses are calculated using the element masses provided by
`periodictable`.

```python
from atomic_scattering_factors import molecular_weight

mass = molecular_weight("H2O")

print(mass)
# approximately 18.015 g/mol
```

Fractional stoichiometries are supported:

```python
mass = molecular_weight("NaY0.5Gd0.3F4")
```

---

## Total atomic number

The total atomic number per formula unit can be obtained with:

```python
from atomic_scattering_factors import get_Z_formula

Z = get_Z_formula("SiO2")

print(Z)
# 30
```

For `SiO2`:

\[
Z = 14 + 2 \times 8 = 30.
\]

---

## Effective electron density

The package can calculate an energy-dependent effective electron density:

\[
\rho_e(E) =
n,Z_\mathrm{eff}(E),
\]

where (n) is the number density of formula units.

The returned quantity has units of electrons per cubic metre (`electrons/m³`).

### Pure elements

For a pure element, the tabulated elemental density can be used automatically:

```python
from atomic_scattering_factors import effective_electron_density

rho_e = effective_electron_density(
    "Si",
    energies_eV=8000,
)
```

### Compounds

For chemical compounds, the mass density must be provided explicitly:

```python
rho_e = effective_electron_density(
    "SiO2",
    mass_density_g_per_cm3=2.2,
    energies_eV=8000,
)
```

This is intentional: there is no unique material density that can be
inferred from a chemical formula alone.

Multiple energies are supported:

```python
energies = np.array([
    5000,
    6000,
    7000,
    8000,
])

rho_e = effective_electron_density(
    "SiO2",
    mass_density_g_per_cm3=2.2,
    energies_eV=energies,
)
```

The returned array has the same shape as `energies`.

Mass densities must be finite and strictly positive.

---

# Installation

## Requirements

The package requires:

* Python
* NumPy
* SciPy
* periodictable

The package also depends on the X-ray scattering-factor interpolation
functionality provided by `xraylabtool`.

## Development installation

Clone the repository and install it in editable mode:

```bash
git clone git@itgit.bs.ptb.de:ulmer01/atomic_scattering_factors.git
cd atomic_scattering_factors
python -m pip install -e .
```

Using an editable installation means that changes to the source code are
immediately available without reinstalling the package.

---

# Usage

After installation, functions can be imported directly from the package:

```python
from atomic_scattering_factors import (
    get_scattering_factors,
    get_effective_Z,
    get_effective_Z_formula,
    get_Z_formula,
    molecular_weight,
    effective_electron_density,
)
```

For example:

```python
import numpy as np

energies = np.linspace(5000, 12000, 100)

rho_e = effective_electron_density(
    "SiO2",
    mass_density_g_per_cm3=2.2,
    energies_eV=energies,
)
```

---

# Testing

The project uses `pytest`.

Install the test dependencies if necessary:

```bash
python -m pip install pytest
```

Run the complete test suite from the project root:

```bash
python -m pytest -v
```

The tests cover:

* chemical formula parsing
* fractional stoichiometries
* dopant notation
* invalid formulas
* molecular masses
* atomic numbers
* scalar energy input
* NumPy array energy input
* multidimensional energy arrays
* scattering factors
* effective atomic numbers
* effective electron densities
* density validation
* consistency between scalar and array calculations

The test suite is located in:

```text
tests/
└── test_atomic_scattering_factors.py
```

---

# Project structure

```text
atomic_scattering_factors/
├── .gitignore
├── pyproject.toml
├── README.md
├── src/
│   └── atomic_scattering_factors/
│       ├── __init__.py
│       └── atomic_scattering_factors.py
└── tests/
    └── test_atomic_scattering_factors.py
```

The source code follows the standard Python `src` layout.

---

# API overview

| Function                       | Description                                           |
| ------------------------------ | ----------------------------------------------------- |
| `get_scattering_factors()`     | Calculate `f1` and `f2` for an element                |
| `get_effective_Z()`            | Calculate energy-dependent effective Z for an element |
| `parse_formula()`              | Parse a chemical formula                              |
| `molecular_weight()`           | Calculate molecular mass                              |
| `get_effective_Z_formula()`    | Calculate effective Z for a chemical formula          |
| `get_Z_formula()`              | Calculate total atomic number                         |
| `effective_electron_density()` | Calculate energy-dependent effective electron density |

---

# Energy input

All functions accepting X-ray energies support both scalar and array-like
input.

For example:

```python
get_effective_Z("Si", 8000)
```

returns a scalar, while:

```python
get_effective_Z(
    "Si",
    np.array([5000, 8000, 12000]),
)
```

returns a NumPy array.

Multidimensional arrays are also supported:

```python
energies = np.array([
    [5000, 6000],
    [7000, 8000],
])

result = get_effective_Z("Si", energies)
```

The output has the same shape as the input.

Energies must be finite and strictly positive.

---

# License

This project is licensed under the terms of the license specified in
`LICENSE`.
