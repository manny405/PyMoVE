
### Pymatgen gives warnings about occupancies which can be safely ignored for 
### these examples
import warnings
warnings.filterwarnings("ignore")

from pymove import SDS
from pymove.io import read,write
from pymove.crystals import FindMolecules

"""
Find molecular units in periodic and non-periodic Structures. In addition, 
identifies if the molecules that are found are duplicates of one another.

Arguments
---------
mult: float
    Multiplicative factor to multiply ase.neighborlist.natural_cutoff.
residues: int
    Number of unqiue molecules the user would like to find. If the value is 
    zero, then default settings are used. However, if ean integer value is 
    supplied, then the mult parameter will be varied to try to achieve an 
    identical number of unique residues in the system. 
conformation: bool
    If True, will account for conformational degrees of freedom when
    identifying unique molecules. If False, then molecules will be 
    duplicates if they have the exact same covalent bonds present. 
formula_check: bool
    If True, overwrites the conformation setting. Only the formula will 
    be checked to determine uniqueness.

"""
###############################################################################
#### Example for finding molecules from a single crystal structure         ####
###############################################################################

fm = FindMolecules(
                 mult=1.05, 
                 residues=1,
                 conformation=False, 
                 formula_check=False)


struct = read("../Example_Structures/crystals/BENZEN.cif")
fm.calc_struct(struct)

print("{} molecules found: {}".format(struct.struct_id, len(fm.molecules)))
print("{} unique molecules: {}".format(struct.struct_id, len(fm.unique)))

write(fm.unique[0].struct_id, fm.unique[0], file_format="xyz", overwrite=True)

###############################################################################
#### Example to iteratively process multiple crystal structures            ####
###############################################################################

fm = FindMolecules(
                 mult=1.05, 
                 conformation=False, 
                 formula_check=False,
                 residues=1)
struct_dict = read("../Example_Structures/crystals")
output_stream = SDS("Unique_Molecules", file_format="xyz", overwrite=True)

for struct_id,struct in struct_dict.items():
    fm.calc_struct(struct)
    
    print("{} molecules found: {}".format(struct.struct_id, len(fm.molecules)))
    print("{} unique molecules: {}".format(struct.struct_id, len(fm.unique)))

    if len(fm.unique) == 1:
        output_stream.update(fm.unique[0])
    else:
        raise Exception("Multiple unique molecules found for {}"
                        .format(struct_id))

