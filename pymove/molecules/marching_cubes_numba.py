
import numpy as np
from ase.data import vdw_radii,atomic_numbers,covalent_radii

from pymove import Structure
from pymove.io import read,write
from pymove.driver import BaseDriver_
from pymove.molecules.utils import com

import numpy as np
from scipy.spatial.distance import cdist
import scipy

from pymove.io import read,write
from pymove.molecules.utils import align
from pymove.molecules.marching_cubes_lookup import *

import numba
from numba import jit,njit, typeof, typed, types 
from numba.extending import overload
import time 

from pymove.molecules.marching_cubes import MarchingCubes
from pymove.molecules.marching_cubes_numba_functions import *

numba_tri_connectivity = typed.Dict.empty(
        key_type=types.unicode_type, 
        value_type=types.int_[:,:])
for key,value in tri_connectivity.items():
    numba_tri_connectivity[key] = value.astype(int)

all_radii = []
for idx,value in enumerate(vdw_radii):
    if np.isnan(value):
        value = covalent_radii[idx]
    all_radii.append(value)
all_radii = np.array(all_radii)

def equal_axis_aspect(ax):
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    
    xrange = xticks[-1] - xticks[0]
    yrange = yticks[-1] - yticks[0]
    zrange = zticks[-1] - zticks[0]
    max_range = max([xrange,yrange,zrange]) / 2
    
    xmid = np.mean(xticks)
    ymid = np.mean(yticks)
    zmid = np.mean(zticks)
    
    ax.set_xlim(xmid - max_range, xmid + max_range)
    ax.set_ylim(ymid - max_range, ymid + max_range)
    ax.set_zlim(zmid - max_range, zmid + max_range)
    
def equal_axis_aspect_2D(ax):
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    
    xrange = xticks[-1] - xticks[0]
    yrange = yticks[-1] - yticks[0]
    max_range = max([xrange,yrange]) / 2
    
    xmid = np.mean(xticks)
    ymid = np.mean(yticks)
    
    ax.set_xlim(xmid - max_range, xmid + max_range)
    ax.set_ylim(ymid - max_range, ymid + max_range)
    
   
def compute_edge_sites(cube_vertex):
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
    pairs = cube_vertex[pair_idx]
    edge = np.mean(pairs, axis=1)
    return edge


class MarchingCubesNumba(MarchingCubes):
    
    def __init__(self, vdw=all_radii, update=True,
                 cache=0.25, spacing=0.25, timing=False):
        self.vdw = vdw
        self.update = update
        self.struct = None
        self.spacing = spacing
        self.cache = cache
        self.timing = timing
        
        ## Storage
        self.offset_combination_dict = self.create_offset_dict_fast()
        self.x_vals = []
        self.y_vals = []
        self.z_vals = []
        

    def create_offset_dict(self):
        ## Find all combinations of small values that lead to less than or equal
        ## to the largest value. This is equivalent to finding all grid points 
        ## within a certain radius
        offset_combination_dict = {}
        max_offset_value = np.round(np.max(self.vdw) / self.cache) + 1
        idx_range = np.arange(-max_offset_value , max_offset_value+1)[::-1]
        sort_idx = np.argsort(np.abs(idx_range))
        idx_range = idx_range[sort_idx]
        all_idx = np.array(
                    np.meshgrid(idx_range,idx_range,idx_range)).T.reshape(-1,3)
        all_idx = all_idx.astype(int)
        all_norm = np.linalg.norm(all_idx, axis=-1)
        
        for value in range(int(max_offset_value+1)):
            min_norm = value
            take_idx = np.where(all_norm <= value)[0]
            
            final_idx = all_idx[take_idx]
            offset_combination_dict[value] = final_idx
            
        return offset_combination_dict
    
    
    def create_offset_dict_fast(self):
        """
        Current offset dict version is rigorous but slow.
        """
        offset_combination_dict = {}
        max_offset_value = np.round(np.max(self.vdw) / self.cache) + 1
        idx_range = np.arange(-max_offset_value , max_offset_value+1)[::-1]
        sort_idx = np.argsort(np.abs(idx_range))
        idx_range = idx_range[sort_idx]
        all_idx = np.array(
                    np.meshgrid(idx_range,idx_range,idx_range)).T.reshape(-1,3)
        all_idx = all_idx.astype(int)
        all_norm = np.linalg.norm(all_idx, axis=-1)
        
        sort_idx = np.argsort(all_norm, kind="mergesort")
        
        self.sort_idx = sort_idx
        
        all_idx = all_idx[sort_idx]
        all_norm = all_norm[sort_idx]
        
        prev_idx = 0
        for value in range(int(max_offset_value+1)):
            idx = np.searchsorted(all_norm[prev_idx:], value, side="right")
            idx += prev_idx
            offset_combination_dict[value] = all_idx[0:idx]
            prev_idx = idx

            
        return offset_combination_dict

        
    def center_molecule(self, struct):
        """
        Simple centering operation.
        
        """
        mol_com = com(struct)
        geo = struct.get_geo_array()
        geo = geo - mol_com
        struct.from_geo_array(geo, struct.geometry["element"])
    
    
    def get_grid(self, struct=None, spacing=0, expand=0):
        """
        Prepares the grid points in a numerically stable way about the origin. 
        If the molecule is not centered at the origin, this will be corrected 
        automatically.
        
        Arguments
        ---------
        expand: float
            Use to construct a volume that is at least this value in each 
            direction. Useful fo resizing the volume for any reason. 
        
        """
        geo = struct.get_geo_array()
        ele = struct.geometry["element"]
        struct_radii = np.array([self.vdw[atomic_numbers[x]] for x in ele])
        struct_centers = self.centers
        
        ### Get minimum and maximum positions that the grid should have
        min_pos = []
        for idx,radius in enumerate(struct_radii):
            temp_pos = struct_centers[idx] - radius - self.spacing
            temp_pos = (temp_pos / self.spacing - 1).astype(int)*self.spacing
            min_pos.append(temp_pos)
        
        max_pos = []
        for idx,radius in enumerate(struct_radii):
            temp_pos = struct_centers[idx] + radius + self.spacing
            temp_pos = (temp_pos / self.spacing + 1).astype(int)*self.spacing
            max_pos.append(temp_pos)

        min_pos = np.min(np.vstack(min_pos), axis=0)
        max_pos = np.max(np.vstack(max_pos), axis=0)
        
        if expand > 0:
            expand_min_pos = np.array([-expand,-expand,-expand])
            expand_max_pos = np.array([expand,expand,expand])
            
            if (expand_min_pos > min_pos).any():
                raise Exception("Cannot have expand less than the minimum "+
                        "position required by the structure")
            if (expand_max_pos < max_pos).any():
                raise Exception("Cannot have expand less than the maximum "+
                        "position required by the structure")
            
            min_pos = expand_min_pos
            max_pos = expand_max_pos
        
        ### Build grid out from the origin
        x_pos_num = np.abs(np.round(max_pos[0] / self.spacing).astype(int))
        x_neg_num = np.abs(np.round(min_pos[0] / self.spacing).astype(int))
        
        y_pos_num = np.abs(np.round(max_pos[1] / self.spacing).astype(int))
        y_neg_num = np.abs(np.round(min_pos[1] / self.spacing).astype(int))
        
        z_pos_num = np.abs(np.round(max_pos[2] / self.spacing).astype(int))
        z_neg_num = np.abs(np.round(min_pos[2] / self.spacing).astype(int))
        
        ### Using linspace instead of arange because arange is not 
        ### numerically stable. 
        x_grid_pos = np.linspace(0,max_pos[0],x_pos_num+1)
        x_grid_neg = np.linspace(min_pos[0], 0-self.spacing, x_neg_num)
        x_grid = np.hstack([x_grid_neg, x_grid_pos])
        
        y_grid_pos = np.linspace(0,max_pos[1],y_pos_num+1)
        y_grid_neg = np.linspace(min_pos[1], 0-self.spacing, y_neg_num)
        y_grid = np.hstack([y_grid_neg, y_grid_pos])
        
        z_grid_pos = np.linspace(0,max_pos[2],z_pos_num+1)
        z_grid_neg = np.linspace(min_pos[2], 0-self.spacing, z_neg_num)
        z_grid = np.hstack([z_grid_neg, z_grid_pos])
        
        self.x_vals = x_grid
        self.y_vals = y_grid
        self.z_vals = z_grid
        
        X,Y,Z = np.meshgrid(self.x_vals, self.y_vals, self.z_vals,
                            indexing="ij")
        
        self.grid_coords = np.c_[X.ravel(),
                                 Y.ravel(),
                                 Z.ravel()]
        
    
    def place_atom_centers(self, struct):
        """
        Places the centers of the atoms onto the grid. This is necessary to 
        ensure numerical stability of the algorithm. While this is an approximation,
        using even a course grid, such as 0.05 this will introduce only a minimum 
        amount of error. Stores radii and centers. 
        
        """
        centers = struct.get_geo_array()
        ele = struct.geometry["element"]
        struct_radii = np.array([self.vdw[atomic_numbers[x]] for x in ele])
        
        ## Compute centers on grid
        grid_centers = []
        for idx,center in enumerate(centers):
            centered_on_grid = np.round(centers[idx] / self.spacing)*self.spacing
            grid_centers.append(centered_on_grid)
        
        ## Store radii and centers
        self.radii = struct_radii
        self.centers = np.vstack(grid_centers)
        self.centers = self.centers.astype(np.float64)
    
    
    def struct_to_volume(self, struct=None, spacing=0, center_com=True,
                         expand=0):
        if spacing == 0:
            spacing = self.spacing
            
        if struct == None:
            struct = self.struct
        
        if center_com:
            self.center_molecule(struct)
        self.place_atom_centers(struct)
        self.get_grid(struct,expand=expand)
        
        min_loc = np.array([self.x_vals[0],self.y_vals[0],self.z_vals[0]])
        volume = np.zeros((self.x_vals.shape[0], 
                           self.y_vals.shape[0], 
                           self.z_vals.shape[0]))
        
        for idx,center in enumerate(self.centers):            
            ## Now compute idx to also populate x,y,z directions for given radius
            rad = self.radii[idx]
            rad_spacing = np.round(rad / spacing).astype(int)
            
            #### THIS SUFFERS FROM NUMERICAL ERRORS
#            all_idx = self.offset_combination_dict[rad_spacing]
#            temp_grid_coords = all_idx*spacing
            
            #### GET ONE SPACING LARGER
            all_idx = self.offset_combination_dict[rad_spacing+1]
            temp_grid_coords = all_idx*spacing
            temp_norm = np.linalg.norm(temp_grid_coords,axis=-1)
            final_idx = np.where(temp_norm < rad)[0]
            temp_grid_coords = temp_grid_coords[final_idx]

            ### 20200429 Trying to correct grid filling
            temp_grid_coords = temp_grid_coords+self.centers[idx]-min_loc
            
            grid_region_idx = np.round(temp_grid_coords / spacing)
            grid_region_idx = grid_region_idx.astype(int)
            
            volume[grid_region_idx[:,0], grid_region_idx[:,1], grid_region_idx[:,2]] = 1
            
        return volume
    
    
    
    def marching_cubes(self, volume):
        
        start = time.time()
        
        X,Y,Z = np.meshgrid(self.x_vals, self.y_vals, self.z_vals,
                    indexing="ij")
        grid_point_reference = np.c_[X.ravel(),
                                     Y.ravel(),
                                     Z.ravel()]
        
        x_num,y_num,z_num = volume.shape
        
        ## Start by projecting down Z direction because this is easiest based on the 
        ## indexing scheme
        z_proj = np.arange(0,z_num-1)
        front_plane_top_left_idx = z_proj
        front_plane_bot_left_idx = front_plane_top_left_idx + 1
        
        ## Have to move 1 in the Y direction which is the same as z_num
        back_plane_top_left_idx = z_proj + z_num
        back_plane_bot_left_idx = back_plane_top_left_idx + 1
        
        ## Have to move 1 in the X direction which is the same as z_num*y_num 
        front_plane_top_right_idx = z_proj + y_num*z_num
        front_plane_bot_right_idx = front_plane_top_right_idx + 1
        
        ## Have to move 1 in the y direction which is the same as z_num
        back_plane_top_right_idx = front_plane_top_right_idx + z_num
        back_plane_bot_right_idx = back_plane_top_right_idx + 1
        
        #### Now project over the Y direction
        y_proj = np.arange(0,y_num-1)[:,None]*(z_num)
        front_plane_top_left_idx = front_plane_top_left_idx + y_proj
        front_plane_bot_left_idx = front_plane_bot_left_idx+ y_proj
        back_plane_top_left_idx = back_plane_top_left_idx+ y_proj
        back_plane_bot_left_idx = back_plane_bot_left_idx+ y_proj
        front_plane_top_right_idx = front_plane_top_right_idx+ y_proj
        front_plane_bot_right_idx = front_plane_bot_right_idx+ y_proj
        back_plane_top_right_idx = back_plane_top_right_idx+ y_proj
        back_plane_bot_right_idx = back_plane_bot_right_idx+ y_proj
        
        
        #### Lastly project in X direction
        x_proj = np.arange(0,x_num-1)[:,None,None]*(y_num*z_num)
        front_plane_top_left_idx = front_plane_top_left_idx + x_proj
        front_plane_bot_left_idx = front_plane_bot_left_idx + x_proj
        back_plane_top_left_idx = back_plane_top_left_idx + x_proj
        back_plane_bot_left_idx = back_plane_bot_left_idx + x_proj
        front_plane_top_right_idx = front_plane_top_right_idx + x_proj
        front_plane_bot_right_idx = front_plane_bot_right_idx + x_proj
        back_plane_top_right_idx = back_plane_top_right_idx + x_proj
        back_plane_bot_right_idx = back_plane_bot_right_idx + x_proj
        #
        voxel_idx = np.c_[front_plane_top_left_idx.ravel(),
                          front_plane_bot_left_idx.ravel(),
                          back_plane_bot_left_idx.ravel(),
                          back_plane_top_left_idx.ravel(),
                          front_plane_top_right_idx.ravel(),
                          front_plane_bot_right_idx.ravel(),
                          back_plane_bot_right_idx.ravel(),
                          back_plane_top_right_idx.ravel(),
                          ]
        
        voxel_mask = np.take(volume, voxel_idx)
        voxel_sum = np.sum(voxel_mask, axis=-1)
        voxel_surface_vertex_idx = np.where(np.logical_and(voxel_sum != 0,
                                             voxel_sum != 8))[0]
        
        self.full_voxels = np.where(voxel_sum == 8)[0]
        
        ## Get only the non-zero points on the surface for visualization
        surface_vertex_idx = voxel_idx[voxel_surface_vertex_idx][
                                voxel_mask[voxel_surface_vertex_idx].astype(bool)]
        surface_vertex = grid_point_reference[surface_vertex_idx]
        
        ## Get the voxels that correspond to the surface of the molecule
        surface_voxel = voxel_mask[voxel_surface_vertex_idx].astype(int)
        ## Get corresponding grid_point_reference idx for each of the surface voxel
        ## verticies
        surface_voxel_vert = voxel_idx[voxel_surface_vertex_idx]
        
        voxel_coords = []
        cube_coords = []
        coords = []
        triangles = []
        total_volume = \
            self.full_voxels.shape[0]*self.spacing*self.spacing*self.spacing
        
        if self.timing:
            print("BEFORE LOOP: {}".format(time.time() - start))
        
        #Convert tri_connectivity to numba typed dictionary     
        ## Make sure every relevant float is float64 before numba operations
        voxel_coords,cube_coords,coords,atom_idx = \
                  surface_numba(
                  surface_voxel, 
                  surface_voxel_vert,
                  grid_point_reference,
                  self.spacing,
                  self.radii,
                  self.centers)
                  
        ## Compute volume with the projected edges
        coords_idx = 0
        for idx,surface_entry in enumerate(surface_voxel):
            key = tostring(surface_entry)
            triangles_bool = tri_connectivity[key]
            tri_idx = np.zeros((triangles_bool.shape[0],3))
            for tri_iter,entry in enumerate(triangles_bool):
                triangles_idx = np.where(entry == 1)[0]
                tri_idx[tri_iter,0] = triangles_idx[0]
                tri_idx[tri_iter,1] = triangles_idx[1]
                tri_idx[tri_iter,2] = triangles_idx[2]
            tri_idx = tri_idx + coords_idx
            triangles.append(tri_idx.astype(int))
            
            temp_vertices = cube_coords[idx]
            temp_edges = coords[idx]
            temp_volume = get_volume(surface_entry, temp_vertices, temp_edges)
            total_volume += temp_volume
            
            ## Keep track of corresponding coords index for tri_idx which 
            ## are using for plotting purposes
            coords_idx += temp_edges.shape[0]
                  
        
        ### For debugging purposes
        self.o_voxel_coords = voxel_coords.copy()
        self.o_cube_coords = cube_coords.copy()
        self.o_coords = coords.copy()
        self.o_triangles = triangles.copy()
        self.surface_voxel = surface_voxel
        self.surface_voxel_vert = surface_voxel_vert
        self.atom_idx = atom_idx
        
        voxel_coords = np.vstack(voxel_coords)
        cube_coords = np.vstack(cube_coords)
        coords = np.vstack(coords)
        triangles = np.vstack(triangles)
        
        if self.timing:
            print("AFTER LOOP: {}".format(time.time() - start))
        
        return total_volume,voxel_coords,cube_coords,coords,triangles


    
if __name__ == "__main__":
    import json
    from scipy.optimize import linear_sum_assignment
    import time
    from pymove import Structure
#    
##    s = Structure.from_geo(np.array([[0,0,0], [1,1,1]]), ["H","H"]) 
#    s = read("/Users/ibier/Research/Volume_Estimation/Datasets/PAHS_geo/BIFUOR.in")
##    s = read("/Users/ibier/Research/Volume_Estimation/Datasets/Polymorph_AVG/PUDXES02.json")
#    
#    spacing=0.1
#    m = MarchingCubes(spacing=spacing,cache=spacing)
#    volume = m.struct_to_volume(s)
#    total_volume,voxel_coords,cube_coords,coords,triangles = m.marching_cubes(volume)
#    print(total_volume)
    
#    vert = unit_cube()
#    numba_edges = numba_compute_edge_sites(vert)
#    edges = compute_edge_sites(vert)
    
    
#    from numba import njit, typeof, typed, types 
#    numba_tri_connectivity = typed.Dict.empty(types.string, types.int_[:,:])
#    for key,value in tri_connectivity.items():
#        numba_tri_connectivity[key] = value.astype(int)
    
    