
from .structure import *
from .struct_dict import *
from ase.data import atomic_masses_iupac2016,atomic_numbers


def volume2density(struct,volume):
    """
    Transforms a given solid-form volume of the molecule to a crystal density
    
    Arguments
    ---------
    struct: pymove.Structure
        Structure object to use for element masses
    volume: float
        Solid-form volume of the given molecule
        
    """
    
    mass = sum([atomic_masses_iupac2016[atomic_numbers[x]]
                   for x in struct.geometry["element"]])
    
    ## Conversion factor for converting amu/angstrom^3 to g/cm^3
    ## Want to just apply factor to avoid any numerical errors to due float 
    factor = 1.66053907

    density = (mass / volume)*factor
    
    return density