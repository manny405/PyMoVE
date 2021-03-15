#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:13:29 2020

@author: manny
"""
import numpy as np
import numba

from pymove.molecules.marching_cubes_lookup import tri_connectivity

from numba import typed,types
from numba.pycc import CC


cc = CC("marching_cubes_numba_functions")

@numba.njit
@cc.export("numba_norm", "float64[:](float64[:,:])")
def numba_norm(matrix):
    result = np.zeros((matrix.shape[0]))
    for idx,entry in enumerate(matrix):
        result[idx] = np.sqrt(np.sum(np.square(entry)))
    return result

@numba.njit
@cc.export("numba_norm_projected", "float64[:,:](float64[:,:,:])")
def numba_norm_projected(matrix):
    result = np.zeros((matrix.shape[0],matrix.shape[1]))
    for idx1,entry1 in enumerate(matrix):
        for idx2,entry2 in enumerate(entry1):
            result[idx1,idx2] = np.sqrt(np.sum(np.square(entry2)))
    return result

@numba.njit
@cc.export("numba_proj_edge", "float64[:](float64[:], int64, float64[:,:], float64, float64[:])")
def numba_proj_edge(edge, edge_idx, vertices, radius, center):
    x = edge[0]
    y = edge[1]
    z = edge[2]
    a = center[0]
    b = center[1]
    c = center[2]
    ## Each edge idx only has one degree of freedom to project onto surface
    if edge_idx == 0:
        ## Z 
        proj2 = radius*radius - np.square(x-a) - np.square(y-b)
        proj_dir_value = z
        proj_dir_center = c
        original = z
    elif edge_idx == 1:
        ## Y
        proj2 = radius*radius - np.square(x-a) - np.square(z-c)
        proj_dir_value = y
        proj_dir_center = b
        original = y
    elif edge_idx == 2:
        ## Z
        proj2 = radius*radius - np.square(x-a) - np.square(y-b)
        proj_dir_value = z
        proj_dir_center = c
        original = z
    elif edge_idx == 3:
        ## Y
        proj2 = radius*radius - np.square(x-a) - np.square(z-c)
        proj_dir_value = y
        proj_dir_center = b
        original = y
    elif edge_idx == 4: 
        ## X
        proj2 = radius*radius - np.square(z-c) - np.square(y-b)
        proj_dir_value = x
        proj_dir_center = a
        original = x
    elif edge_idx == 5:
        ## X
        proj2 = radius*radius - np.square(z-c) - np.square(y-b)
        proj_dir_value = x
        proj_dir_center = a
        original = x
    elif edge_idx == 6:
        ## X
        proj2 = radius*radius - np.square(z-c) - np.square(y-b)
        proj_dir_value = x
        proj_dir_center = a
        original = x
    elif edge_idx == 7: 
        ## X
        proj2 = radius*radius - np.square(z-c) - np.square(y-b)
        proj_dir_value = x
        proj_dir_center = a
        original = x
    elif edge_idx == 8:
        ## Z
        proj2 = radius*radius - np.square(x-a) - np.square(y-b)
        proj_dir_value = z
        proj_dir_center = c
        original = z
    elif edge_idx == 9:
        ## Y 
        proj2 = radius*radius - np.square(x-a) - np.square(z-c)
        proj_dir_value = y
        proj_dir_center = b
        original = y
    elif edge_idx == 10:
        ## Z
        proj2 = radius*radius - np.square(x-a) - np.square(y-b)
        proj_dir_value = z
        proj_dir_center = c
        original = z
    elif edge_idx == 11:
        ## Y
        proj2 = radius*radius - np.square(x-a) - np.square(z-c)
        proj_dir_value = y
        proj_dir_center = b
        original = y
    if proj2 < 0:
        proj2 = proj2*-1
    
    proj = np.sqrt(proj2)
    
    ### 20200429 Fix decision function
    temp_pos_dir = abs((proj + proj_dir_center) - proj_dir_value)
    temp_neg_dir = abs((-proj + proj_dir_center) - proj_dir_value)
#    temp_pos_dir = np.linalg.norm((proj + proj_dir_center) - proj_dir_value)
#    temp_neg_dir = np.linalg.norm((-proj + proj_dir_center) - proj_dir_value)
    
    if temp_neg_dir < temp_pos_dir:
        proj = proj*-1 + proj_dir_center
    else:
        proj = proj + proj_dir_center
    
    ## Check if projection is within the spacing of the grid. 
    ## If it's outside, then this cannot be a valid projection.
    ## And the value is set back to original edge position. 
    if edge_idx == 0:
        ## Z, 0,1 
        if proj < vertices[0][2]:
            proj = z
        elif proj > vertices[1][2]:
            proj = z
    elif edge_idx == 1:
        if proj < vertices[0][1]:
            proj = y
        elif proj > vertices[3][1]:
            proj = y
    elif edge_idx == 2:
        ## Z 2,3
        if proj < vertices[3][2]:
            proj = z
        elif proj > vertices[2][2]:
            proj = z
    elif edge_idx == 3:
        if proj < vertices[1][1]:
            proj = y
        elif proj > vertices[2][1]:
            proj = y
    elif edge_idx == 4: 
        ## X 0,4
        if proj < vertices[0][0]:
            proj = x
        elif proj > vertices[4][0]:
            proj = x
    elif edge_idx == 5:
        ## X 3,7
        if proj < vertices[3][0]:
            proj = x
        elif proj > vertices[7][0]:
            proj = x
    elif edge_idx == 6:
        ## X 2,6
        if proj < vertices[2][0]:
            proj = x
        elif proj > vertices[6][0]:
            proj = x
    elif edge_idx == 7: 
        ## X, 1,5
        if proj < vertices[1][0]:
            proj = x
        elif proj > vertices[5][0]:
            proj = x
    elif edge_idx == 8:
        ## Z, 4.5
        if proj < vertices[4][2]:
            proj = z
        elif proj > vertices[5][2]:
            proj = z
    elif edge_idx == 9:
        ## Y 4,7
        if proj < vertices[4][1]:
            proj = y
        elif proj > vertices[7][1]:
            proj = y
    elif edge_idx == 10:
        ## Z, 6,7
        if proj < vertices[7][2]:
            proj = z
        elif proj > vertices[6][2]:
            proj = z
    elif edge_idx == 11:
        ## Y, 5,6
        if proj < vertices[5][1]:
            proj = y
        elif proj > vertices[6][1]:
            proj = y
    
    
    ### Return final projection
    ret_edge = edge.copy()
    if edge_idx == 0:
        ## Z 
        ret_edge[2] = proj
    elif edge_idx == 1:
        ## Y
        ret_edge[1] = proj
    elif edge_idx == 2:
        ## Z
        ret_edge[2] = proj
    elif edge_idx == 3:
        ## Y
        ret_edge[1] = proj
    elif edge_idx == 4: 
        ## X
        ret_edge[0] = proj
    elif edge_idx == 5:
        ## X
        ret_edge[0] = proj
    elif edge_idx == 6:
        ## X
        ret_edge[0] = proj
    elif edge_idx == 7: 
        ## X
        ret_edge[0] = proj
    elif edge_idx == 8:
        ## Z
        ret_edge[2] = proj
    elif edge_idx == 9:
        ## Y 
        ret_edge[1] = proj
    elif edge_idx == 10:
        ## Z
        ret_edge[2] = proj
    elif edge_idx == 11:
        ## Y
        ret_edge[1] = proj
    
    return ret_edge

@numba.njit
@cc.export("tostring_numba", "unicode_type(float64[:])")
def tostring_numba(array):
    """
    1D array to string
    """
    temp_string = ""
    
    for value in array[:-1]:
        if value == 0:
            temp_string += "0"
        else:
            temp_string += "1"
        
        temp_string += ","
    
    if array[-1] == 0:
        temp_string += "0"
    else:
        temp_string += "1"
        
    return temp_string
            

################################################################################
### Preparing lookup dictionaries in numba typed dictionaries
################################################################################
numba_tri_connectivity = typed.Dict.empty(
        key_type=types.unicode_type, 
        value_type=types.int_[:,:])
for key,value in tri_connectivity.items():
    numba_tri_connectivity[key] = value.astype(int)


## Only load once
pair_idx = np.array([
            [0,1],
            [0,3],
            [2,3],
            [1,2],
            
            [0,4],
            [3,7],
            [2,6],
            [1,5],
            
            [4,5],
            [4,7],
            [6,7],
            [5,6],
            
            ])
            
@numba.njit
@cc.export("numba_compute_edge_sites", "float64[:,:](float64[:,:])")
def numba_compute_edge_sites(cube_vertex):
    """
    Tested to give same values as original.
    
    """
    edges = np.zeros((12,3)).astype(np.float64)
    
    for idx,temp_pair_idx in enumerate(pair_idx):
        pass
        v1 = cube_vertex[temp_pair_idx[0]]
        v2 = cube_vertex[temp_pair_idx[1]]
        temp_edge = v1 + v2 
        temp_edge = temp_edge / 2
        edges[idx,0] = temp_edge[0]
        edges[idx,1] = temp_edge[1]
        edges[idx,2] = temp_edge[2]
        
    return edges


#### https://numba.pydata.org/numba-doc/dev/reference/types.html
#### FOR TESTING SIGNATURES sigutils.normalize_signature
#### FOR GETTING SIGNATURE <jitted_function>.inspect_types()
@cc.export("surface_numba", "Tuple((List(float64[:,::1]),List(float64[:,::1]),List(float64[:,::1]),float64[:,::1]))(int64[:,:], int64[:,:], float64[:,:], float64, float64[:], float64[:,:])") #, DictType(unicode_type,float64[:,:,::1]))")
def surface_numba(surface_voxel, 
                  surface_voxel_vert,
                  grid_point_reference,
                  spacing,
                  radii,
                  centers):
    
    ## Keep track of the populated surface voxel positions
    voxel_coords = []
    ## Keep track of surface voxel cube vertex coordinates
    cube_coords = []
    ## Keep track of the edge positions that will be used for plotting surface
    coords = []
    ## Keep track of which atom each edge belongs to
    atom_idx = np.zeros((surface_voxel.shape[0],12))
    atom_idx -= 1
    
    ## Is it possible to convert this to entirely vectorized algorithm? 
    
    
    for idx,entry in enumerate(surface_voxel):      
        
        ## Get Cartesian Coordinates index
        temp_ref_idx = surface_voxel_vert[idx]
        ### Get populated coordinates
        keep_idx = np.where(entry == 1)[0]
        temp_voxel_coords = grid_point_reference[temp_ref_idx[keep_idx]]
        voxel_coords.append(temp_voxel_coords)
    
        ### Get Cart Cube vertex and edges
        temp_vertices = grid_point_reference[temp_ref_idx]
        temp_edges = numba_compute_edge_sites(temp_vertices)
        
        ### Performing projections onto sphere surfaces for each edge point
        for edge_idx,edge in enumerate(temp_edges):
            ### Project onto surface of each sphere present
            
            ### First choose relevant spheres
            edge_to_center = numba_norm(edge - centers)
            edge_to_center_inside = edge_to_center - radii
            proj_sphere_idx = np.where(np.abs(edge_to_center_inside) <=
                                       (spacing*2))[0]
            
            ### Allocate storage
            temp_projected_edge_list = np.zeros((len(proj_sphere_idx),3))
            temp_projected_centers = np.zeros((len(proj_sphere_idx),3))
          
            for r_iter_idx,r_idx in enumerate(proj_sphere_idx):
                ## Also, need center of the atom for proper projection
                temp_center = centers[r_idx]
                radius = radii[r_idx]
                temp_proj_edge = numba_proj_edge(edge, 
                                               edge_idx, 
                                               temp_vertices, 
                                               radius, 
                                               temp_center)
                
                ## If there was no change, do not append
                if np.linalg.norm(temp_proj_edge - edge) < 1e-6:
                    continue
              
                ## Store result
                temp_projected_edge_list[r_iter_idx] = temp_proj_edge
                temp_projected_centers[r_iter_idx] = temp_center
            
            keep_idx = np.where(np.sum(np.abs(temp_projected_edge_list),axis=-1) != 0)[0]
            temp_projected_edge_list = temp_projected_edge_list[keep_idx]
            temp_projected_centers = temp_projected_centers[keep_idx]
                               
            ## Let's see if this problem can be solved in a different way
            if len(temp_projected_edge_list) == 0:
                continue
            elif len(temp_projected_edge_list) == 1:
                choice_idx = 0
            else:
                temp = temp_projected_edge_list - np.expand_dims(temp_projected_centers,1)
                cdist_distances = numba_norm_projected(temp).T
                
                ## Choose the one that maximizes distances
                cdist_sum = np.sum(cdist_distances,axis=-1)
                choice_idx = np.argmax(cdist_sum)
            
            temp_edges[edge_idx,0] = temp_projected_edge_list[choice_idx,0]
            temp_edges[edge_idx,1] = temp_projected_edge_list[choice_idx,1]
            temp_edges[edge_idx,2] = temp_projected_edge_list[choice_idx,2]
            
            ## Store atom choice
            if len(proj_sphere_idx) > 0:
                atom_idx[idx,edge_idx] = proj_sphere_idx[choice_idx]
        
        ### Save results for plotting
        cube_coords.append(temp_vertices)
        coords.append(temp_edges)
    
    return voxel_coords,cube_coords,coords,atom_idx

#@numba.njit
#@cc.export("generate_edge", "float64[:,::1](float64[:],float64[:],float64)")
#def generate_edge(point1,point2,spacing):
#    edge_vec = point1 - point2
#    dist = np.linalg.norm(edge_vec)
#    num_points = np.round(dist / (spacing*0.25))
#    values = np.arange(0,num_points+1)
#    if len(values) == 0 or len(values) == 1:
#        return_list = np.zeros((1,3))
#        return_list[0] = point1
#        return return_list
#    values = values / np.max(values)
#    return_list = np.zeros((values.shape[0],3))
#    for idx,entry in enumerate(values):
#        temp_value = point1 - entry*edge_vec
#        return_list[idx] = temp_value
#    return return_list 
#
#@cc.export("fill_face", "List(float64[::1])(float64[:,::1],float64)")
#def fill_face(face, spacing):
#    edge1 = generate_edge(face[0], face[1], spacing)
#    edge2 = generate_edge(face[0], face[2], spacing)
#    edge3 = generate_edge(face[1], face[2], spacing)
#
#    all_points = []
#    for point1 in edge1:
#        for point2 in edge2:
#            for point3 in edge3:
#                temp_edge = generate_edge(point1,point1, spacing)
#                for entry in temp_edge:
#                    all_points.append(entry)
#                
#                temp_edge = generate_edge(point1,point2, spacing)
#                for entry in temp_edge:
#                    all_points.append(entry)
#                    
#                temp_edge = generate_edge(point2,point3, spacing)
#                for entry in temp_edge:
#                    all_points.append(entry)
#    
##    return_points = np.zeros((len(all_points),3))
##    for idx,entry in enumerate(all_points):
##        return_points[idx] = entry
#
#    return all_points


if __name__ == "__main__":
    cc.compile()