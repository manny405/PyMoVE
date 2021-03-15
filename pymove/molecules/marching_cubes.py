
import numpy as np
from ase.data import vdw_radii,atomic_numbers,covalent_radii
from ase.data.colors import jmol_colors

from pymove import Structure
from pymove.io import read,write
from pymove.driver import BaseDriver_
from pymove.molecules.utils import com

import numpy as np
from scipy.spatial.distance import cdist
import scipy

from pymove.io import read,write
from pymove.molecules.align import align
from pymove.molecules.marching_cubes_lookup import *

from numba import jit
from numba.extending import overload
import time 



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


class MarchingCubes(BaseDriver_):
    
    def __init__(self, vdw=all_radii, update=True,
                 cache=0.25, spacing=0.25):
        self.vdw = vdw
        self.update = update
        self.struct = None
        self.spacing = spacing
        self.cache = cache
        self.offset_combination_dict = self.create_offset_dict_fast()
        
        ## Storage
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
        


    def calc_struct(self, struct):
        self.struct = struct
        volume = self.struct_to_volume(self.struct)
        total_volume,voxel_coords,cube_coords,coords,triangles = \
                self.marching_cubes(volume)
        
        self.struct.properties["Marching_Cubes_Volume"] = total_volume
        
        return total_volume

        
    def center_molecule(self, struct):
        """
        Simple centering operation.
        
        """
        mol_com = com(struct)
        geo = struct.get_geo_array()
        geo = geo - mol_com
        struct.from_geo_array(geo, struct.geometry["element"])
    
    
    def point_to_grid(self, points):
        """
        Returns the nearest point on the grid with respect to the argument.
        Also, returns the index of this point with respect to the grid coords. 
        
        Arguments
        ---------
        points: array
            2D array of points
        """
        if len(self.x_vals) == 0 or len(self.y_vals) == 0 or len(self.z_vals) == 0:
            raise Exception()
        
        points_on_grid = np.round(points / self.spacing)*self.spacing
        ### Compute index with respect to grid limits
        min_loc = np.array([self.x_vals[0],self.y_vals[0],self.z_vals[0]])
        temp_grid_coords = points_on_grid-min_loc
        
        grid_region_idx = np.round(temp_grid_coords / self.spacing)
        grid_region_idx = grid_region_idx.astype(int)
        
        return points_on_grid,grid_region_idx
    
    
    def sphere_to_grid(self, radius, center):
        """
        Returns how a new sphere would be added to the current grid. 
        
        """
        spacing = self.spacing
        min_loc = np.array([self.x_vals[0],self.y_vals[0],self.z_vals[0]])
        
        center_on_grid = np.round(center / self.spacing)*self.spacing
        rad_spacing = np.round(radius / self.spacing).astype(int)
        all_idx = self.offset_combination_dict[rad_spacing+1]
        temp_grid_coords = all_idx*spacing
        temp_norm = np.linalg.norm(temp_grid_coords,axis=-1)
        final_idx = np.where(temp_norm < radius)[0]
        temp_grid_coords = temp_grid_coords[final_idx]

        ### 20200429 Trying to correct grid filling
        temp_grid_coords = temp_grid_coords+center_on_grid-min_loc
        
        grid_region_idx = np.round(temp_grid_coords / spacing)
        grid_region_idx = grid_region_idx.astype(int)
        
        return grid_region_idx
        
    
    def get_grid(self, struct=None, spacing=0):
        """
        Prepares the grid points in a numerically stable way about the origin. 
        If the molecule is not centered at the origin, this will be corrected 
        automatically.
        
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
    
    
    def struct_to_volume(self, struct=None, spacing=0, center_com=True):
        if spacing == 0:
            spacing = self.spacing
            
        if struct == None:
            struct = self.struct
        
        if center_com:
            self.center_molecule(struct)
        self.place_atom_centers(struct)
        self.get_grid(struct)
        
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
        total_volume = self.full_voxels.shape[0]*self.spacing*self.spacing*self.spacing
        
#        print("BEFORE LOOP: {}".format(time.time() - start))
        
        proj_total_time = 0
        inner_loop_time = 0
        radius_loop_time = 0
        for idx,entry in enumerate(surface_voxel):
            
            ### Get Cartesian Coordinates index
            temp_ref_idx = surface_voxel_vert[idx]
            ### Get populated coordinates
            voxel_coords.append(grid_point_reference[
                    temp_ref_idx[entry.astype(bool)]])
        
            ### Get Cart Cube vertex and edges
            temp_vertices = grid_point_reference[temp_ref_idx]
            temp_edges = compute_edge_sites(temp_vertices)
            
            inner_loop_start = time.time()
            
            
            ### Performing projections onto sphere surfaces for each edge point
            for edge_idx,edge in enumerate(temp_edges):
                
                rad_loop_start = time.time()
                ### Project onto surface of each sphere present
                temp_projected_edge_list = []
                temp_projected_centers = []
                
                ### First choose relevant spheres
                edge_to_center = np.linalg.norm(edge - self.centers, axis=-1)
                edge_to_center_inside = edge_to_center - self.radii
                proj_sphere_idx = np.where(np.abs(edge_to_center_inside) <=
                                           (self.spacing*2))[0]
                
                for r_idx in proj_sphere_idx:
                    ## Also, need center of the atom for proper projection
                    temp_center = self.centers[r_idx]
                    temp_projected_centers.append(temp_center)
                    radius = self.radii[r_idx]
                    
                    
                    proj_edge_start = time.time() 
                    ## Get the projected edge for this sphere
#                    temp_proj_edge = self.proj_edge(edge, 
#                                                   edge_idx, 
#                                                   temp_vertices, 
#                                                   radius, 
#                                                   temp_center)
                    temp_proj_edge = numba_proj_edge(edge, 
                                                   edge_idx, 
                                                   temp_vertices, 
                                                   radius, 
                                                   temp_center)
                    proj_total_time += time.time() - proj_edge_start
                    
                    ## If there was no change, do not append
                    if np.linalg.norm(temp_proj_edge - edge) < 1e-6:
                        continue
                    
                    ## Append
                    temp_projected_edge_list.append(temp_proj_edge)
                
                ## Let's see if this problem can be solved in a different way
                if len(temp_projected_edge_list) == 0:
                    continue
                elif len(temp_projected_edge_list) == 1:
                    choice_idx = 0
                else:
                    cdist_distances = cdist(temp_projected_edge_list,
                                               temp_projected_centers)
                    ## Choose the one that maximizes distances
                    cdist_sum = np.sum(cdist_distances,axis=-1)
                    choice_idx = np.argmax(cdist_sum)
                
                ### Hard code for now because only interested in testing for one sphere
                temp_edges[edge_idx] = temp_projected_edge_list[choice_idx]
            
            inner_loop_time += time.time() - inner_loop_start
            
            ### Get the tri_idx for this surface voxel
            triangles_bool = tri_connectivity[tostring(entry)].astype(bool)
            array_to_mask = np.repeat(np.arange(0,12)[None,:], 
                                triangles_bool.shape[0], 
                                axis=0)
            tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
        
            ### Build triangles for grid point reference
            tri_idx = tri_idx + len(coords)*12
        
            ### Save results for plotting
            cube_coords.append(temp_vertices)
            coords.append(temp_edges)
            triangles.append(tri_idx)
            
            ## Compute volume with the projected edges
            total_volume += get_volume(entry, temp_vertices, temp_edges)
        
        ### For debugging purposes
        self.o_voxel_coords = voxel_coords.copy()
        self.o_cube_coords = cube_coords.copy()
        self.o_coords = coords.copy()
        self.o_triangles = triangles.copy()
        self.surface_voxel = surface_voxel
        self.surface_voxel_vert = surface_voxel_vert
        
        voxel_coords = np.vstack(voxel_coords)
        cube_coords = np.vstack(cube_coords)
        coords = np.vstack(coords)
        triangles = np.vstack(triangles)
        
#        print("AFTER LOOP: {}".format(time.time() - start))
#        print("PROJ TOTAL TIME: {}".format(proj_total_time))
#        print("INNER LOOP TIME: {}".format(inner_loop_time))
#        print("RADIUS LOOP TIME: {}".format(radius_loop_time))
        
        return total_volume,voxel_coords,cube_coords,coords,triangles

    
    def proj_edge(self, edge, edge_idx, vertices, radius, center):
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
        temp_pos_dir = np.linalg.norm((proj + proj_dir_center) - proj_dir_value)
        temp_neg_dir = np.linalg.norm((-proj + proj_dir_center) - proj_dir_value)
        
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
    
    
    def marching_cubes_basic(self, volume):
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
        
        #### Working on surface triangulation
        
        ## Get the voxels that correspond to the surface of the molecule
        surface_voxel = voxel_mask[voxel_surface_vertex_idx].astype(int)
        ## Get corresponding grid_point_reference idx for each of the surface voxel
        ## verticies
        surface_voxel_vert = voxel_idx[voxel_surface_vertex_idx]
        
        voxel_coords = []
        cube_coords = []
        coords = []
        triangles = []
        total_volume = self.full_voxels.shape[0]*self.spacing*self.spacing*self.spacing
        for idx,entry in enumerate(surface_voxel):
            
            ### Get Cartesian Coordinates index
            temp_ref_idx = surface_voxel_vert[idx]
            ### Get populated coordinates
            voxel_coords.append(grid_point_reference[
                    temp_ref_idx[entry.astype(bool)]])
        
            ### Get Cart Cube vertex and edges
            temp_vertices = grid_point_reference[temp_ref_idx]
            temp_edges = compute_edge_sites(temp_vertices)
            
            ### Get the tri_idx for this surface voxel
            triangles_bool = tri_connectivity[tostring(entry)].astype(bool)
            array_to_mask = np.repeat(np.arange(0,12)[None,:], 
                                triangles_bool.shape[0], 
                                axis=0)
            tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
        
            ### Build triangles for grid point reference
            tri_idx = tri_idx + len(coords)*12
        
            ### Save results for plotting
            cube_coords.append(temp_vertices)
            coords.append(temp_edges)
            triangles.append(tri_idx)
            
            adjusted_vol = tri_volume[tostring(entry)]
            total_volume += (adjusted_vol*self.spacing*self.spacing*self.spacing)
        
        ### For debugging purposes
        self.o_voxel_coords = voxel_coords.copy()
        self.o_cube_coords = cube_coords.copy()
        self.o_coords = coords.copy()
        self.surface_voxel = surface_voxel
        
        voxel_coords = np.vstack(voxel_coords)
        cube_coords = np.vstack(cube_coords)
        coords = np.vstack(coords)
        triangles = np.vstack(triangles)
        
        return total_volume,voxel_coords,cube_coords,coords,triangles


@jit(nopython=True)
def numba_handle_edges(temp_edges,
                       temp_vertices,
                       centers,
                       radii,
                       spacing):
        
    ### MUCH FASTER BUT NOT TESTED
    ### Performing projections onto sphere surfaces for each edge point
    for edge_idx,edge in enumerate(temp_edges):
        ### First choose relevant spheres
        temp = edge-centers
        edge_to_center = numba_norm(temp)
        edge_to_center_inside = edge_to_center - radii
        proj_sphere_idx = np.where(np.abs(edge_to_center_inside) <=
                                   (spacing*2))[0]
        
        ### Project onto surface of each sphere present
        temp_projected_edge_list = np.zeros((len(proj_sphere_idx),3))
        temp_projected_centers = np.zeros((len(proj_sphere_idx),3))
        
        for r_idx in proj_sphere_idx:
            ## Also, need center of the atom for proper projection
            temp_center = centers[r_idx]
#            temp_projected_centers.append(temp_center)
            radius = radii[r_idx]
            
            temp_proj_edge = numba_proj_edge(edge, 
                                           edge_idx, 
                                           temp_vertices, 
                                           radius, 
                                           temp_center)
            
            ## If there was no change, do not append
#            if np.linalg.norm(temp_proj_edge - edge) < 1e-6:
#                continue
            
            ## Append
#            temp_projected_edge_list.append(temp_proj_edge)
            
            temp_projected_centers[r_idx] = temp_center
            temp_projected_edge_list[r_idx] = temp_proj_edge
        
        ## Let's see if this problem can be solved in a different way
        if len(temp_projected_edge_list) == 0:
            continue
        elif len(temp_projected_edge_list) == 1:
            choice_idx = 0
        else:
#                cdist_distances = cdist(temp_projected_edge_list,
#                                           temp_projected_centers)
#            cdist_distances = np.linalg.norm(temp_projected_edge_list - 
#                                    temp_projected_centers[:,None], 
#                                    axis=-1)
            temp = temp_projected_edge_list - np.expand_dims(temp_projected_centers,1)
            cdist_distances = numba_norm_projected(temp)
            ## Choose the one that maximizes distances
            cdist_sum = np.sum(cdist_distances,axis=-1)
            choice_idx = np.argmax(cdist_sum)
        
        ### Hard code for now because only interested in testing for one sphere
        temp_edges[edge_idx] = temp_projected_edge_list[choice_idx]
    
    return temp_edges

@jit(nopython=True)
def numba_norm(matrix):
    result = np.zeros((matrix.shape[0]))
    for idx,entry in enumerate(matrix):
        result[idx] = np.sqrt(np.sum(np.square(entry)))
    return result

@jit(nopython=True)
def numba_norm_projected(matrix):
    result = np.zeros((matrix.shape[0],matrix.shape[1]))
    for idx1,entry1 in enumerate(matrix):
        for idx2,entry2 in enumerate(entry1):
            result[idx1,idx2] = np.sqrt(np.sum(np.square(entry2)))
    return result


@jit(nopython=True)
def numba_proj_edge(edge, edge_idx, vertices, radius, center):
#        x,y,z = edge
#        a,b,c = center
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
    
    
if __name__ == "__main__":
    import json
    from scipy.optimize import linear_sum_assignment
    import time
    
#    spacing = 0.01
#    
#    start = time.time()
#    m = MarchingCubes(cache=spacing)\
#    end = time.time()
#    
#    print("Class Construction: {}".format(end - start))
    