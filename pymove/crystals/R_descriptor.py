
import time
import numpy as np
import torch

from pymove import Structure,StructDict
from pymove.crystals.motif_utils import construct_supercell_by_molecule


def atomic_R_descriptor(struct_dict, cutoff_R=-1,
                        supercell=2, include_negative=True):
    """ 
    Construct the R lists (inter-atomic distance lists) for all atoms in 
    each structure in the struct_dict.
    
    Arguments
    ---------
    struct_dict: dict
        Standard Structure Dictionary
    cutoff_R: float
        If the cutoff is less than 0, then no cutoff will be applied. If 
        cutoff is greater than zero, then all distances greater than this 
        value or identified and removed. This comes at a computational cost.
    supercell: int
        Number of supercells to use in the construction of R 
    include_negative:
        Builds supercell about the origin by also including negative 
        translation vectors. This is highly advisable for the calculation 
        of the R descriptor. 
    
    """
    struct_dict_R = {}
    for struct_id,struct in struct_dict.items():
        R = calc_R(struct, cutoff_R=cutoff_R, supercell=supercell, 
                   include_negative=include_negative)
        struct.set_property("R", R)
        struct_dict_R[struct_id] = struct
    return struct_dict_R


def calc_R(struct, cutoff_R=-1, supercell=2, include_negative=True):
    """ Performs computation of atomic R descriptor for input structure
    """
    
    original_geo = struct.get_geo_array()
    original_ele = struct.geometry['element']
    ele_types = np.unique(original_ele)
    
    # R computation for periodic structure
    if len(struct.get_lattice_vectors_better()) > 0:
        supercell = construct_supercell_by_molecule(
                       struct, supercell=supercell, 
                       include_negative=include_negative)
        supercell_ele = supercell.geometry['element']
        supercell_geo = supercell.get_geo_array()
        dist_geo = supercell_geo
        dist_ele = supercell_ele
    # R for non-periodic structures
    else:
        dist_geo = original_geo
        dist_ele = original_ele
    
    # Compute R distances    
    all_dist = np.linalg.norm(dist_geo - original_geo[:,None,:],axis=2)
    ele_take = [np.argwhere(dist_ele == x).ravel() for x in ele_types]
    taken = [np.take(all_dist,x,axis=1) for x in ele_take]
    
    
    result_dict = {}
    atom_list = [original_ele[i]+"_{}".format(i) 
                 for i in range(original_ele.shape[0])]
    
    for i,atom_index in enumerate(atom_list):
        result_dict[atom_index] = {}
        temp = result_dict[atom_index]
        for j,ele in enumerate(ele_types):
            contact_name = original_ele[i]+"-{}".format(ele)
            if cutoff_R > 0:
                values = taken[j][i,:]
                temp[contact_name] = values[values < cutoff_R].tolist()
            else:
                temp[contact_name] = taken[j][i,:].tolist()
            
    return result_dict


def ele_R(R):
    """ 
    Concatenates an R descriptor which is indexed by each individual atom in 
    the structure to an R descriptor indexed by element.
    
    Arguments
    ---------
    R: dict
        Atom indexed dictionary with interatomic distances index by 
        interactomic distance type.
        
    """
    keys = [x for x in R.keys()]
    elements = [x.split("_")[0] for x in keys]
    unique_ele = np.unique(elements)

    interactions = [[] for x in range(len(unique_ele))]
    for i,ele_1 in enumerate(unique_ele):
        for ele_2 in unique_ele:
            interactions[i].append("{}-{}".format(ele_1, ele_2))
    
    ele_R = {}
    for i,ele in enumerate(unique_ele):
        ele_R[ele] = {}
        for interaction in interactions[i]:
            ele_R[ele][interaction] = torch.tensor([])[:,None]
    
    for key in keys:
        ele = key.split("_")[0]
        temp_R = R[key]
        for inter_name,value in temp_R.items():
            value = torch.FloatTensor(value).view(1,-1)
            temp_inter = ele_R[ele][inter_name]
            if temp_inter.nelement() == 0:
                ele_R[ele][inter_name] = value
            else:
                ele_R[ele][inter_name] = torch.cat(
                    (ele_R[ele][inter_name],value), dim=0)
    
    return ele_R


def struct_R(R):
    """ From ele_R, concatenate and sort all Rij for each element
    """
    final_R = {}
    for element,R_dict in R.items():
        final_R[element] = {}
        for interaction,Rij in R_dict.items():
            final = Rij[0,:]
            for row in Rij[1:,:]:
                final = torch.cat((final,row), dim=0)
            final,indices = torch.sort(final)
            final_R[element][interaction] = final
    return final_R
    


if __name__ == "__main__":
    pass
#    from ibslib.io import read,write
##    dimer_path = "C:\\Users\\manny\\Research\\Datasets\\Hab_Dimers\\FUQJIK_4mpc_tight\\dimers"
##    dimer_dict = read(dimer_path)
##    test_struct = dimer_dict["00e3c7641c_d_07RaEM"]
#    
#    test = AtomicPairDistance(p_type="structure")
#    test.calculate(test_struct)
#    R = test_struct.get_property("R")
#    temp = {}
#    for key,value in R.items():
#        temp[key] = {}
#        for nkey,nvalue in value.items():
#            temp[key][nkey] = nvalue.tolist()
#    test_struct.set_property("R",temp)
#    write("C:\\Users\\manny\\Research\\Datasets\\Hab_Dimers\\FUQJIK_4mpc_tight\\test.json", test_struct,
#          overwrite=True)
#    
