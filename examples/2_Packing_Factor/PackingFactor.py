
### Pymatgen gives warnings about occupancies which can be safely ignored for 
### these examples
import warnings
warnings.filterwarnings("ignore")

from pymove.io import read,write
from pymove.crystals import PackingFactor

"""
    Calculatees the geometric packing factor using the vdW radii of the 
    structure and a user specified grid spacing. The algorithm is as follows:
        1. Generate Supercell of user specified size. This is to ensure that all
           of the necessary atoms are within the unit cell.
        2. Keep only atoms that are within the unit cell plus a correction 
           equal to the largest van der Waals radius in the system.
        3. Generate a grid using the specified grid spacing. This is done by 
           computing the grid spacing that should be used based on the lattice
           vector norm in every direction. Then, this grid is generated in 
           fraction space and converted finally to real space. 
        5. For each location of the grid spacing, calculate how far it is from 
           each atom. 
        6. For the distance to each atom, divide by the vdW radius of  the
           respective atom. 
        7. All values less than 1 are occupied and all values greater than 1 
           are considered empty. 
        8. Divide filled by total to get the packing factor.
        
    Arguments
    ---------
    spacing: float
        Spacing is given in Angstroms
    vdw: iterable
        Iterable that can be indexed where the index is equal to the atomic 
        number and the value at the index is the radius to use. 
    supercell_mult: int
        Size of supercell to build in every lattice direction.
    low_memory: bool
        If True, an implementation that requires a smaller, fixed amount of 
        system memory is used at the expense of additional compute time. This
        should be set to True if the user would like to use grid spacings
        below 0.25 Angstrom.

"""


pf = PackingFactor(spacing=0.25, 
                   supercell_mult=2,
                   low_memory=False)
struct_dict = read("../Example_Structures/crystals")

for struct_id,struct in struct_dict.items():
    temp_pf_result = pf.calc_struct(struct)
    print("{} Packing Factor: {}".format(struct_id, temp_pf_result))

### If structures are written as JSON files, then the calculated value will
### be stored with the geometry, and other properties, of the structure
write("Structures_With_Packing_Factor", struct_dict, file_format="json", overwrite=True)

### Results can be read back in
print("---------------------------------------------------")
print("----------------- READING RESULTS -----------------")
print("---------------------------------------------------")

result_struct_dict = read("Structures_With_Packing_Factor")
for struct_id,struct in result_struct_dict.items():
    stored_pf = struct.properties["Packing_Factor"]
    print("{} Stored Packing Factor: {}".format(struct_id, stored_pf))
