# -*- coding: utf-8 -*-



"""
File that describes some relatively low level operations on Structures. 
This is different from utils because utils generally provides relatively 
high level operations on Structures. For example, finding molecules. 
This file provides lower level operations such as rotations, combining, and 
aligning. 

"""


import copy
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation 

from ase.data import atomic_numbers,atomic_masses_iupac2016

from pymove import Structure
    

def com(struct):
    """
    Calculates center of mass of the system. 

    """
    geo_array = struct.get_geo_array()
    element_list = struct.geometry['element'] 
    mass = np.array([atomic_masses_iupac2016[atomic_numbers[x]] 
                     for x in element_list]).reshape(-1)
    total = np.sum(mass)
    com = np.sum(geo_array*mass[:,None], axis=0)
    com = com / total
    return com


def cartesian_com(struct):
    geo_array = struct.get_geo_array()
    return np.mean(geo_array,axis=0)


def combine(struct1,struct2,lat=False):
    """
    Combines two structures. 
    
    Arguments
    ---------
    lat: bool
        If True, keeps lattice vectors of first structure. 
    """
    geo1 = struct1.get_geo_array()
    ele1 = struct1.geometry["element"]
    
    combined = Structure.from_geo(geo1,ele1)
    
    if lat == True:
        lattice = struct1.get_lattice_vectors()
        combined.set_lattice_vectors(lattice)
    
    geo2 = struct2.get_geo_array()
    ele2 = struct2.geometry["element"]
    
    for idx,coord in enumerate(geo2):
        combined.append(coord[0],coord[1],coord[2],ele2[idx])
    
    return combined
    

def center_struct_cartesian(struct):
    """
    Centers the structure on the (0,0,0) origin. Uses the point center of mass
    of the structure, which does not take into account atomic masses. 
    
    """
    cart_com = cartesian_com(struct)
    geo = struct.get_geo_array()
    geo -= cart_com
    
    struct.from_geo_array(geo, struct.geometry["element"])
    
    return struct
    

def get_rot(struct1,struct2,force_d=0):
    """
    Gets rotation matrix to overlap struct1 and struct2. Rotation matrix can 
    be applied to either geometry by np.dot(rot,geo.T).T
    
    """
    geo1 = struct1.get_geo_array()
    geo2 = struct2.get_geo_array()
    
    h = np.dot(geo1.T,geo2)
    u,s,v = np.linalg.svd(h)
    
    d = np.linalg.det(np.dot(v,u.T)) 
    if force_d != 0:
        d = force_d
    d = np.array([[1,0,0],[0,1,0],[0,0,d]])
    rot = np.dot(u,np.dot(d,v))
    
#    d = np.linalg.det(u)*np.linalg.det(v)
#    if d < 0:
#        u[:,-1] = -u[:,-1]
#        print("MIRROR")
#    rot = np.dot(u,v)
    
    return rot


def rot_struct(rot, struct, frac=False):
    """
    Applys rotation matrix to the geometry of the structure. 
    
    Arguments
    ---------
    frac: bool
        If frac is True, then the lattice of the Structure will be considered
        and the rotation will take place in fractional space before being
        converted back to cartesian space. 
    
    """
    if frac == False:
        geo = struct.get_geo_array()
        rot_geo = np.dot(geo, rot)
        
        struct.from_geo_array(rot_geo, struct.geometry["element"])
        
        return struct
    else:
        lat = np.dot(rot, np.vstack(struct.get_lattice_vectors()))
        frac_coords = get_frac(struct)
        frac_coords = np.dot(frac_coords, rot)
        
        cart = np.dot(frac_coords, lat)
        
        struct.from_geo_array(cart, struct.geometry["element"])
        struct.properties["lattice_vector_a"] = lat[0].tolist()
        struct.properties["lattice_vector_b"] = lat[1].tolist()
        struct.properties["lattice_vector_c"] = lat[2].tolist()
        return struct
    

def remove_H(struct):
    """
    Removes hydrogens from structure.
    
    """
    geo = struct.get_geo_array()
    ele = struct.geometry["element"]
    h_idx = np.where(ele != "H")[0]
    geo = geo[h_idx]
    ele = ele[h_idx]
    struct.from_geo_array(geo,ele)
    
    return struct
    
    
def align_structs(struct1,struct2,lat=False,H=True,weight=False):
    """
    Brings structure 1 into alignment with struct2
    
    Arguments
    ---------
    struct1: Structure
        Structure to align struct2 with
    struct2: Structure
        Structure that is rotated to match struct1
    lat: bool
        If True, uses the lattice of struct1 in the return Structure
    H: bool 
        If False, hydrogens will not be considered when creating the rotation 
        matrix. Typically, hydrogen positions from the CSD may be incorrect
        and can negatively impact the performance of the algorithm. Also, 
        it is common that matching hydrogen positions may not be of interest to 
        the user. 
    force_d: 0,1,-1
        For testing purposes. 
    
    """
    original_pos = cartesian_com(struct1)
    
    temp1 = copy.deepcopy(struct1)
    temp2 = copy.deepcopy(struct2)
    
    center_struct_cartesian(temp1)
    center_struct_cartesian(temp2)
    
    if H:
        pass
    else:
        remove_H(temp1)
        remove_H(temp2)
        
    geo1 = temp1.get_geo_array()
    geo2 = temp2.get_geo_array()
    
    ### Need to reorder geometries to match by index
    dist = cdist(geo1,geo2)
    idx1,idx2 = linear_sum_assignment(dist)
    
    geo1 = geo1[idx1,:]
    geo2 = geo2[idx2,:]
    
    temp1.from_geo_array(geo1, temp1.geometry["element"][idx1])
    temp2.from_geo_array(geo2, temp2.geometry["element"][idx2])
    
    if weight:
        ## First sort geometries by elements to get same weight vectors
        ele1 = temp1.geometry["element"]
        ele2 = temp2.geometry["element"]
        
        s1_idx = np.argsort(ele1)
        s2_idx = np.argsort(ele2)
        
        geo1 = geo1[s1_idx]
        geo2 = geo2[s2_idx]
        
        ele1 = ele1[s1_idx]
        ele2 = ele2[s2_idx]
        
        ## Create weight list from elements
        weights = [atomic_masses_iupac2016[atomic_numbers[x]] for x in 
                   ele1]
        
        ## Get rotation matrix from scipy
        Rot,rmsd = Rotation.align_vectors(geo1,geo2,weights=weights)
        rot = Rot.as_matrix()
        
    else:
        ## Get rotation object from scipy without weights
        Rot,rmsd = Rotation.align_vectors(geo1,geo2)
        
        rot = Rot.as_matrix()
        
    rot_struct(rot,temp1)
    rmsd = calc_rmsd(temp1,temp2)
    
    temp1.geometry["element"] = "O"
    temp2.geometry["element"] = "N"
    
    combined = combine(temp1,temp2,lat=lat)
    
    ## If using a lattice, then translation the geometry to the original 
    ## position so that the original structure preserved
    if lat:
        geo = combined.get_geo_array()
        geo += original_pos
        combined.from_geo_array(geo, combined.geometry["element"])
        
    combined.struct_id = "Combined_{}_{}".format(
            struct1.struct_id,struct2.struct_id)
    combined.properties["Combined"] = [struct1.struct_id,
                                       struct2.struct_id]
    combined.properties["RMSD"] = rmsd
        
    return combined 


def align_structs_iter(struct1,struct2,lat=False,H=True,weight=False):
    """
    Same as align_structs but modify structures so that this function can 
    be called over and over again until convergence.
    
    Arguments
    ---------
    struct1: Structure
        Structure to align struct2 with
    struct2: Structure
        Structure that is rotated to match struct1
    lat: bool
        If True, uses the lattice of struct1 in the return Structure
    H: bool 
        If False, hydrogens will not be considered when creating the rotation 
        matrix. Typically, hydrogen positions from the CSD may be incorrect
        and can negatively impact the performance of the algorithm. Also, 
        it is common that matching hydrogen positions may not be of interest to 
        the user. 
    force_d: 0,1,-1
        For testing purposes. 
    
    """
    original_pos = cartesian_com(struct1)
    
    temp1 = struct1
    temp2 = struct2
    
    center_struct_cartesian(temp1)
    center_struct_cartesian(temp2)
    
    if H:
        pass
    else:
        remove_H(temp1)
        remove_H(temp2)
        
    geo1 = temp1.get_geo_array()
    geo2 = temp2.get_geo_array()
    
    ### Need to reorder geometries to match by index
    dist = cdist(geo1,geo2)
    idx1,idx2 = linear_sum_assignment(dist)
    
    geo1 = geo1[idx1,:]
    geo2 = geo2[idx2,:]
    
    temp1.from_geo_array(geo1, temp1.geometry["element"][idx1])
    temp2.from_geo_array(geo2, temp2.geometry["element"][idx2])
    
    if weight:
        ## First sort geometries by elements to get same weight vectors
        ele1 = temp1.geometry["element"]
        ele2 = temp2.geometry["element"]
        
        s1_idx = np.argsort(ele1)
        s2_idx = np.argsort(ele2)
        
        geo1 = geo1[s1_idx]
        geo2 = geo2[s2_idx]
        
        ele1 = ele1[s1_idx]
        ele2 = ele2[s2_idx]
        
        ## Create weight list from elements
        weights = [atomic_masses_iupac2016[atomic_numbers[x]] for x in 
                   ele1]
        
        ## Get rotation matrix from scipy
        Rot,rmsd = Rotation.align_vectors(geo1,geo2,weights=weights)
        rot = Rot.as_matrix()
        
    else:
        ## Get rotation object from scipy without weights
        Rot,rmsd = Rotation.align_vectors(geo1,geo2)
        
        rot = Rot.as_matrix()
        
    rot_struct(rot,temp1)
    rmsd = calc_rmsd(temp1,temp2)
    
    temp1.geometry["element"] = "O"
    temp2.geometry["element"] = "N"
    
    combined = combine(temp1,temp2,lat=lat)
    
    ## If using a lattice, then translation the geometry to the original 
    ## position so that the original structure preserved
    if lat:
        geo = combined.get_geo_array()
        geo += original_pos
        combined.from_geo_array(geo, combined.geometry["element"])
        
    combined.struct_id = "Combined_{}_{}".format(
            struct1.struct_id,struct2.struct_id)
    combined.properties["Combined"] = [struct1.struct_id,
                                       struct2.struct_id]
    combined.properties["RMSD"] = rmsd
        
    return struct1,struct2,rmsd 


def calc_rmsd(struct1,struct2):
    """
    Calculates the rmsd between two structures. 
    
    """
    geo1 = struct1.get_geo_array()
    geo2 = struct2.get_geo_array()
    
    dist = cdist(geo1,geo2)
    
    idx1,idx2 = linear_sum_assignment(dist)
    
    geo1 = geo1[idx1]
    geo2 = geo2[idx2]
    
    return np.mean(np.linalg.norm(geo1 - geo2,axis=-1))


def get_frac(struct, points=[]):
    """
    Obtains fraction coordinates for the crystal structure. If points are 
    provided, then it will convert these points into fractional coordinates. 
    
    Verified against Pymatgen fractional coordinates. 
    
    """
    lat = np.vstack(struct.get_lattice_vectors())
    linv = np.linalg.inv(lat)
    
    geo = struct.get_geo_array()
    frac = np.dot(geo,linv)
    
    return frac
    
    
    
    