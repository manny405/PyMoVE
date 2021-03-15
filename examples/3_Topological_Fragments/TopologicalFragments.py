

from pymove.io import read,write
from pymove.molecules import TopologicalFragments,FragmentImage

"""
Returns the topological fragments of each atom for a structure. User is 
allowed to define a radius that the algorithm traverses to build 
the neighborhood for each atom. If the radius is 0, this would 
correspond to just return the atoms im the system.

Arguments
---------
bond_kw: dict
    Keyword arguments for the molecule bonding module. The default values
    are the recommended settings. A user may potentially want to decrease
    the mult. This value is multiplied by covalent bond 
    radii in the MoleculeBonding class. It's highly recommended that the
    skin value is kept at 0.
    
"""

tf = TopologicalFragments(
                    bond_kw={"mult": 1.20,
                            "skin": 0,
                            "update": True})

struct_dict = read("../Example_Structures/molecules")

for struct_id,struct in struct_dict.items():
    tf.calc_struct(struct)
    print("{} Fragments and Counts: {}"
          .format(struct_id, struct.properties["topological_fragments"]))


###############################################################################
#### Plotting fragment image results                                      #####
###############################################################################

fi = FragmentImage()
fi.calc_dict(struct_dict, figname="Fragment_Image.pdf")