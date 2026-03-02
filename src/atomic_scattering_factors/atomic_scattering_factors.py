import re
from collections import OrderedDict
from xraylabtool.calculators.core import create_scattering_factor_interpolators
import periodictable as pt
import numpy as np

def get_scattering_factors(element, energies_eV):
    """
    Get the atomic scattering factors f1 and f2 for a given element and X-ray energies.
    
    Parameters
    ----------
    element : str
        Element symbol, e.g. 'H', 'C', 'O'
    energies_eV : array-like
        X-ray energies in eV
    
    Returns
    -------
    f1 : array-like
        Scattering factor f1 for each energy in `energies_eV`
    f2 : array-like
        Scattering factor f2 for each energy in `energies_eV`
    """
    f1_interp, f2_interp = create_scattering_factor_interpolators(element)    
    f1 = f1_interp(energies_eV)
    f2 = f2_interp(energies_eV)
    return f1, f2

def get_effective_Z(element, energies_eV):
    """
    Calculate the effective atomic number for a given element and X-ray energies.
    
    Parameters
    ----------
    element : str
        Element symbol, e.g. 'H', 'C', 'O'
    energies_eV : array-like
        X-ray energies in eV
    
    Returns
    -------
    effective_Z : array-like
        Effective atomic number for each energy in `energies_eV`
    """
    f1, f2 = get_scattering_factors(element, energies_eV)
    return np.sqrt(f1**2 + f2**2)

def parse_formula(formula):
    """
    Parse a molecular formula and return elements and atom counts.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. 'H2O', 'HOH', 'C6H12O6'

    Returns
    -------
    elements : list of str
        Element symbols
    num_atoms : list of int
        Corresponding atom counts
    """
    # match element symbols and optional numbers
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)

    counts = OrderedDict()
    for element, count in tokens:
        count = int(count) if count else 1
        counts[element] = counts.get(element, 0) + count

    return list(counts.keys()), list(counts.values())

def molecular_weight(formula):
    """Compute the molecular mass (g/mol) using periodictable."""
    elements, counts = parse_formula(formula)
    mass = 0.0
    for el, n in zip(elements, counts):
        try:
            mass += pt.elements.symbol(el).mass * n
        except KeyError:
            raise ValueError(f"Unknown element symbol: {el}")
    return mass

def get_effective_Z_formula(chemical_formula, energies):
    """
    Calculate the effective atomic number for a chemical formula by summing the contributions of each element.
    
    Parameters
    ----------
    chemical_formula : str
        Chemical formula, e.g. 'H2O', 'HOH', 'C6H12O6'
    energies : array-like
        X-ray energies in eV
    
    Returns
    -------
    effective_Z : array-like
        Effective atomic number for each energy in `energies`
    """
    elements, num_atoms = parse_formula(chemical_formula)
    return sum(num_atoms[i] * get_effective_Z(elements[i], energies) for i in range(len(elements)))

def get_Z_formula(chemical_formula, energies):
    """
    Calculate the total atomic number for a chemical formula by summing the contributions of each element.
    
    Parameters
    ----------
    chemical_formula : str
        Chemical formula, e.g. 'H2O', 'HOH', 'C6H12O6'
    energies : array-like
        X-ray energies in eV
    
    Returns
    -------
    total_Z : array-like
        Total atomic number for each energy in `energies`
    """
    elements, num_atoms = parse_formula(chemical_formula)
    return sum(num_atoms[i] * pt.elements.symbol(elements[i]).number for i in range(len(elements)))