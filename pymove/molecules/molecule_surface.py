

import numpy as np

# cKDTREE is 1000x faster than the Python KDTree implementation
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from ase.data import atomic_numbers
from ase.data.colors import jmol_colors

from pymove import Structure
from pymove.driver import BaseDriver_
from pymove.molecules.marching_cubes_numba import MarchingCubesNumba,all_radii


class MoleculeSurface(BaseDriver_):
    """
    Class for computing the molecular surface and molecular volume given a 
    probe radius. If the probe radius is zero, this just returns the 
    result from the marching cubes algorithm.
    
    Arguments
    ---------
    spacing: float
        Spacing to use for generation of the voxel grid used to construct 
        surface
    probe_radius: float
        Probe radius to use when generating the molecular surface. 
    
    """
    def __init__(self, spacing=0.3, probe_radius=1.4):
        self.spacing=spacing
        self.probe_radius=probe_radius
        self.SA_radii = all_radii.copy()+probe_radius
        
        self.SA_m = MarchingCubesNumba(spacing=spacing,
                                       cache=spacing,
                                       vdw=self.SA_radii)
        self.m = MarchingCubesNumba(spacing=spacing,
                                    cache=spacing)
        
        self.struct = None
    
    
    def calc_struct(self, struct):
        """
        Algorithm is pretty straight forward and goes as follows:
            1) Get solvent accessible volume given the probe radius
            2) Find the points on the surface of the SA volume
            3) Build KDTree for surface points and query all points in the
               SA volume. 
            4) All points in the SA volume that are further from the surface 
               than the probe radius are points in the molecular surface 
               volume.
            5) Generate molecular surface volume using standard radii. 
            6) Add the points from step 4) to the molecular surface volume
            7) Performing marching cubes and calculate volume
            
        Also, from this, it's quite easy to evaluate the size of the voids 
        found in the molecular structure.  
               
        """
        self.struct = struct
        
        ## Get molecular surface coordates from the SA volume
        self.SA_volume = self.SA_m.struct_to_volume(self.struct)
        ## Now, get the molecule volume for the regular radii
        self.volume = self.m.struct_to_volume(self.struct)
        
        ## See if anything needs to be kept based on tol value
        tol = self.probe_radius-self.spacing*0.9
        if tol > 0:
            surface_coords = self.get_surface_coords(self.SA_volume)
            kdtree = cKDTree(surface_coords, leafsize=5)
            SA_coords = self.SA_m.grid_coords[self.SA_volume.ravel().astype(bool)]
            dist,idx = kdtree.query(SA_coords, k=1)
            
            keep_idx = np.where((dist-tol) > 0)[0]
            molecular_surface_coords = SA_coords[keep_idx]
        
            ## Add points
            coords,grid_idx = self.m.point_to_grid(molecular_surface_coords)
            self.volume[grid_idx[:,0],grid_idx[:,1],grid_idx[:,2]] = 1
            
        else:
            ## If tol is less than zero, then it's known that nothing needs
            ## to be added to the volumes
            pass
            
        total_volume,voxel_coords,cube_coords,coords,triangles = \
                 self.m.marching_cubes(self.volume)
        
        ## Storage for plotting purposes
        self.total_volume = total_volume
        self.voxel_coords = voxel_coords
        self.cube_coords  = cube_coords
        self.coords = coords
        self.triangles = triangles
        
        ## Store volume
        self.struct.properties["Molecular_Surface_Volume"] = total_volume
        
        return total_volume
    
    
    def get_plot_info(self, plotly=True, nearest=True):
        """
        Return information used for plotting the surface of the molecule.
        
        Arguments
        ---------
        plotly: bool
            If True, will return colors in the form accepted by Plotly.
            If False, returns in array of floats between 0-1 for RGB
        
        """
        if self.struct == None:
            raise Exception("Must calculate structure before calling "+
                    "for plot information")
            
        if nearest: 
            tri_atom_idx = self.assign_triangles_nearest(
                                self.struct, 
                                self.coords, 
                                self.triangles)
        else:
            tri_atom_idx = self.assign_triangles(
                                self.struct, 
                                self.coords, 
                                self.triangles, 
                                self.m.atom_idx)
        
        if plotly:
            colors = self.get_plotly_colors(self.struct, tri_atom_idx)
        else:
            ele = self.struct.geometry["element"]
            colors = [jmol_colors[atomic_numbers[ele[x]]].tolist() 
                      for x in tri_atom_idx]
        
        return self.coords,self.triangles,colors
        
    
    
    def get_surface_coords(self, SA_volume):
        """
        Obtain the surface coordinates given the solvent accessible volume. 
        
        """
        ## Build KD tree of the surface points
        X,Y,Z = np.meshgrid(self.SA_m.x_vals, 
                            self.SA_m.y_vals, 
                            self.SA_m.z_vals,
                            indexing="ij")
        grid_point_reference = np.c_[X.ravel(),
                                     Y.ravel(),
                                     Z.ravel()]
        
        x_num,y_num,z_num = SA_volume.shape
        
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
        
        voxel_mask = np.take(SA_volume, voxel_idx)
        voxel_sum = np.sum(voxel_mask, axis=-1)
        voxel_surface_vertex_idx = np.where(np.logical_and(voxel_sum != 0,
                                             voxel_sum != 8))[0]
        surface_vertex_idx = voxel_idx[voxel_surface_vertex_idx][
                                        voxel_mask[voxel_surface_vertex_idx].astype(bool)]
        
        ## Get surface cartesian coordinates
        surface_coords = np.unique(grid_point_reference[surface_vertex_idx],axis=0)
        return surface_coords
    
    
    def assign_triangles(self, struct, edge_coords, triangles, atom_idx):
        """
        
        Arguments
        ---------
        edge_coords: 
            Positions of all relevant edges
        triangles: np.array (n,3)
        atom_idx: (n,12)
            Assignment of each edge to the appropriate atom
        
        """
        tri_atom_idx = []
        geo = struct.get_geo_array()
        geo_radii = [all_radii[atomic_numbers[x]] for x in struct.geometry["element"]]
        for tri_idx in triangles:
            edge_atom_idx = np.take(atom_idx, tri_idx).astype(int)
            unique_idx,unique_counts = np.unique(edge_atom_idx, return_counts=True)
            if len(unique_idx) == 1:
                if unique_idx[0] == -1:
                    tri_coords = edge_coords[tri_idx]
                    dist = cdist(tri_coords,geo)
                    shortest_dist_atom_idx = np.argmin(dist) - int(
                            np.argmin(dist)/geo.shape[0])*geo.shape[0]
                    tri_atom_idx.append(shortest_dist_atom_idx)
                    continue
    
            non_negative = np.where(edge_atom_idx >= 0)[0]
            tri_coords = edge_coords[tri_idx]
            tri_coords = tri_coords[non_negative]
            dist = cdist(tri_coords,geo)
            ## Turn absolute magnitude into a magnitude dependent on the radius of the atom
            dist = dist / geo_radii
            shortest_dist_atom_idx = np.argmin(dist) - int(
                    np.argmin(dist)/geo.shape[0])*geo.shape[0]
            tri_atom_idx.append(shortest_dist_atom_idx)
        
        return np.array(tri_atom_idx)
    
    
    def assign_triangles_nearest(self, struct, edge_coords, triangles):
        """
        Assign triangles simply by computing nearest atom. 
        
        """
        tri_atom_idx = []
        
        ## Need to use atom adjust positions where positions are snapped to grid
        ## locations
        geo = self.m.centers
        
        geo_radii = [self.m.vdw[atomic_numbers[x]] 
                     for x in struct.geometry["element"]]
        
        for tri_idx in triangles:
            tri_coords = edge_coords[tri_idx]
            dist = cdist(tri_coords,geo)
            ## Turn absolute magnitude into a magnitude dependent on the 
            ## radius of the atom
            dist = dist / geo_radii
            min_dist = np.min(dist)
            
            if min_dist <= 1.15:
                shortest_dist_atom_idx = np.argmin(dist) - int(
                       np.argmin(dist)/geo.shape[0])*geo.shape[0]
            else:
                shortest_dist_atom_idx = -1
                
            tri_atom_idx.append(shortest_dist_atom_idx)

        return tri_atom_idx


    def get_plotly_colors(self, struct, atom_idx):
        ele = struct.geometry["element"]
        
        colors = []
        for entry in atom_idx:
            if entry >= 0:
                colors.append(jmol_colors[atomic_numbers[ele[entry]]].tolist())
            else:
                colors.append([0.886,0.7765,1.0])
        
        plotly_colors = []
        for color in colors:
            plotly_colors.append("rgb({},{},{})".format(255*color[0],
                                                        255*color[1],
                                                        255*color[2]))
        return plotly_colors
    
    

class SlabSurface(MoleculeSurface):
    """
    Class for generating only the upper surface of a periodic surface slab. 
    This can be used for characterizing roughness or for performing a best
    fit between two surface slabs. 
    
    Algorithm is as follows:
        1) Generate a supercell of the surface slab in the plane perpendicular 
           to the vacuum direction in order to capture periodic nature of the 
           unit cell. 
        2) Generate the voxelized volume representation of the supercell.
        3) Identified the positions that should be added to the voxelized 
           system as part of the molecular surface in the usual way. 
        4) Perform a modified marching cubes algorithm for surfaces. This 
           algorithm performs marching cubes for each column in the z direction
           and stops when the first entirely populated cube is found. In then
           backtracks 1 position because this will be the surface location. 
        5) Regenerates the volume of the supercell but only including those
           populated surface positions identified in 3).
        6) Keeps only the surface positions inside the unit cell by converting 
            the positions to a fractional coordinate such that the surface 
            points that are less than [1,1,0] are inside the unit cell.
        7) Performs sphere projected marching cubes algorithm on the unit cell's
            surface to complete the construction of the surface contour for a 
            slab structure. 
    
    Arguments
    ---------
    spacing: float
        Spacing to use for generation of the voxel grid used to construct 
        surface
    probe_radius: float
        Probe radius to use when generating the molecular surface. 
    surface_supercell: iterable
        Iterable containing two integers for the supercell used in the final
        generation of the slab surface.
    supercell_radius: float
        Radius in angstrom to use beyond the surface_supercell for the 
        proper construction of the surface. This will construct the supercell
        such that the plane extends beyond greater than or equal to the 
        supercell radius using integer lattice translations. 
    vacuum_direction: int
        Define the vacuum direction for the surface slab. 
    
    """
    def __init__(self, 
                 spacing=0.3, 
                 probe_radius=1.4, 
                 surface_supercell=[1,1],
                 supercell_radius=1,
                 vacuum_direction="z"):
        self.spacing=spacing
        self.probe_radius=probe_radius
        self.surface_supercell=surface_supercell
        self.supercell_radius=supercell_radius
        self.SA_radii = all_radii.copy()+probe_radius
        
        self.SA_m = MarchingCubesNumba(spacing=spacing,
                                       cache=spacing,
                                       vdw=self.SA_radii)
        self.m = MarchingCubesNumba(spacing=spacing,
                                    cache=spacing)
        
        self.struct = None
        
        
        if vacuum_direction != "z":
            raise Exception("Vacuum direction must be the z direction for now."
            +"In principle, I think it's as easy as turning vacuum "
            +"such that the vacuum is in z and the rotating back after.")
        
        
    def calc_struct(self, struct):
        """
        Follows algorithm as described above.  
        
        """
        self.struct = struct
        
        ## 1) Build surface supercell
        self.supercell = self.build_surface_supercell(self.struct,
                                                      self.surface_supercell,
                                                      self.supercell_radius)
        
        ## Remove atoms outside the relevant surface_supercell value
        self.supercell = self.get_atoms_inside_unitcell(self.supercell)
        
        ## 2) Get SA volume and normal volume
        self.SA_volume = self.SA_m.struct_to_volume(self.supercell)
        self.volume = self.m.struct_to_volume(self.supercell)
        
        ## 3) Usual idenfication of positions to add for molecular surface
        ## Can improve the efficiency of the algorithm by removing identifying
        ## relevant vacuum surface and then removing all volume points that 
        ## are below the relevant distance from the z-value of the lowest 
        ## surface point. 
        tol = self.probe_radius-self.spacing*0.9
        if tol > 0:
            ## Set surface coords to be used 
            surface_coords = self.get_surface_coords(self.SA_volume)
            kdtree = cKDTree(surface_coords, leafsize=5)
            
            ## Still need to query for all points in the SA_volume.
            SA_coords = self.SA_m.grid_coords[self.SA_volume.ravel().astype(bool)]
            
            dist,idx = kdtree.query(SA_coords, k=1)
            
            keep_idx = np.where((dist-tol) > 0)[0]
            molecular_surface_coords = SA_coords[keep_idx]
        
            ## Add points
            coords,grid_idx = self.m.point_to_grid(molecular_surface_coords)
            self.volume[grid_idx[:,0],grid_idx[:,1],grid_idx[:,2]] = 1
        else:
            ## If tol is less than zero, then it's known that nothing needs
            ## to be added to the volumes
            pass
        
        
        ## With modified volume using the SA volume, pass through the vacuum surface
        ## and unit cell identification algorithm again
        self.vacuum_surface_volume = self.get_vacuum_surface(self.volume)
        self.supercell_surf_pos,self.unitcell_surf_pos = \
            self.get_surf_pos(self.vacuum_surface_volume, self.m)
        
        ## 7) Perform marching cubes algorithm for just this region
        self.final_surface_volume = np.zeros(self.volume.shape)
        _,idx = self.m.point_to_grid(self.unitcell_surf_pos)
        self.final_surface_volume[idx[:,0],idx[:,1],idx[:,2]] = 1
        
        total_volume,voxel_coords,cube_coords,coords,triangles = \
            self.m.marching_cubes(self.final_surface_volume)
        
        self.coords = coords
        self.triangles = triangles      
        
        roughness = self.get_roughness(mode="mean")
        
        self.struct.properties["Roughness"] = roughness
        return roughness
        
    
    def get_surf_pos(self, volume, ms):
        """
        1. Converts the volume to cartesian coordinates. 
           Returns these as the first argument. 
        2. Keeps only positions that are within the self.surface_supercell setting. 
           Returns these as the second argument. 
        
        Arguments
        ---------
        volume: np.array
            3D array that describes the voxelized surface coordinates. 
        ms: pymove.molecules.marching_cubes
            Marching cubes object in order to construct the appropriate grid
            point reference values. 
            
        
        Returns
        -------
        np.array,np.array
            1. SA_supercell_surf_pos which is 2D array of coordinates 
               for the entire supercell surface which is used in the construction of the 
               unit cell surface. 
            2. SA_unitcell_surf_pos is not strictly the surface of the single unit 
               cell. This is the surface for the user defined setting of 
               self.surface_supercell. This is called the unit cell because this is 
               the region that the user is actually interested in. 
        
        """
        ## Get Cartesian coordinates for the surface
        X,Y,Z = np.meshgrid(ms.x_vals, 
                            ms.y_vals, 
                            ms.z_vals,
                            indexing="ij")
        grid_point_reference = np.c_[X.ravel(),
                                     Y.ravel(),
                                     Z.ravel()]
        ## Can be used for plotting
        SA_supercell_surf_pos = np.unique(
            grid_point_reference[volume.ravel().astype(bool)],
            axis=0)
        
        ## 5) Transform cartesian into fractional coordinates of the original structure
        lat = np.vstack(self.struct.get_lattice_vectors())
        lat_inv = np.linalg.inv(lat.T)
        SA_frac_pos = np.dot(lat_inv,SA_supercell_surf_pos.T).T

        ## Keep only those withint the self.surface_supercell region
        unit_cell_keep_idx = np.where(
            np.logical_and(
            np.logical_and(
                SA_frac_pos[:,0]>=(-self.surface_supercell[0]/2),
                SA_frac_pos[:,0]<=(self.surface_supercell[0]/2)
            ),
            np.logical_and(
                SA_frac_pos[:,1]>=(-self.surface_supercell[0]/2),
                SA_frac_pos[:,1]<=(self.surface_supercell[0]/2)
            ))
        )[0]

        SA_unitcell_surf_pos = SA_supercell_surf_pos[unit_cell_keep_idx]
        
        return SA_supercell_surf_pos, SA_unitcell_surf_pos
        
    
    def build_surface_supercell(self, struct=None, supercell=[1,1], radius=1):
        """
        Builds the surface supercell that will be used for the generation of the
        surface slab contour. 
    
        Arguments
        ---------
        surface_supercell: iterable
            Iterable containing two integers for the supercell used in the final
            generation of the slab surface.
        supercell_radius: float
            Radius in angstrom to use beyond the surface_supercell for the 
            proper construction of the surface. This will construct the supercell
            such that the plane extends beyond greater than or equal to the 
            supercell radius using integer lattice translations. 
    
    
        """
        if struct == None:
            struct = self.struct
    
        ## Get additional range based on the radius value
        lat = np.vstack(struct.get_lattice_vectors())[[0,1],:]
        lat_norm = np.linalg.norm(lat, axis=-1)
        ## Supercell contribution from radius setting
        rad_supercell = np.ceil(radius/lat_norm).astype(int)
    
        ## Construct integer ranges 
        full_supercell = np.array(supercell) + rad_supercell
        
        ## Plus two to center the supercell around the unit cell
        range1 = np.arange(-full_supercell[0],
                           full_supercell[0]+2,1)
        range2 = np.arange(-full_supercell[1],
                           full_supercell[1]+2,1)
        
        trans1,trans2 = np.meshgrid(range1,range2)
        all_trans = np.c_[trans1.ravel(),trans2.ravel()]
        
        geo = struct.get_geo_array()
        ele = struct.geometry["element"]
        geo_list = [geo]
        ele_list = [ele]
        for trans in all_trans:
            if list(trans) == [0,0]:
                continue
            
            ## Verified to be correct operation
            trans_vec = np.dot(trans, lat)
            temp_geo = geo.copy()
            temp_geo += trans_vec[None,:]
            geo_list.append(temp_geo)
            ele_list.append(ele)
    
        geo_list = np.vstack(geo_list)
        ele_list = np.hstack(ele_list)
    
        supercell = Structure.from_geo(geo_list, ele_list)
        supercell.struct_id = "{}_Slab_Supercell_{}-{}".format(struct.struct_id,
                                                               full_supercell[0],
                                                               full_supercell[1])
        
        return supercell
    
    
    def get_atoms_inside_unitcell(self, supercell):
        """
        Keep only the points that are inside the unit cell. 
        
        Arguments
        ---------
        tol: float
            Tolerance to be applied with respect to fractional coordinates. 
            
        """
        geo = supercell.get_geo_array()
        ele = supercell.geometry["element"]
        
        ## Transform cartesian into fractional coordinates of the original structure
        lat = np.vstack(self.struct.get_lattice_vectors())
        lat_inv = np.linalg.inv(lat.T)
        frac_pos = np.dot(lat_inv,geo.T).T
        
        ## Choose correct tolerance for the system
        tol=[]
        if len(tol) == 0:
            ## Find largest radius in the system using the SA radii
            radii = [self.SA_m.vdw[atomic_numbers[x]] for x in ele]
            max_radius = np.max(radii)
            lat_norm = np.linalg.norm(lat[[0,1],:], axis=-1)
            
            ## Tolerance is equal to the maximum radius value of the SA radii
            ## divided by the each lat_norm because the tolerance needs 
            ## to be in fractional coordinates
            tol = [max_radius / lat_norm[0], 
                   max_radius / lat_norm[1]]

        ## Keep only those within tthe self.surface_supercell region
        unit_cell_keep_idx = np.where(
            np.logical_and(
            np.logical_and(
                frac_pos[:,0]>=(-self.surface_supercell[0]/2-tol[0]),
                frac_pos[:,0]<=(self.surface_supercell[0]/2+tol[0])
            ),
            np.logical_and(
                frac_pos[:,1]>=(-self.surface_supercell[0]/2-tol[0]),
                frac_pos[:,1]<=(self.surface_supercell[0]/2+tol[0])
            ))
        )[0]
            
        geo = geo[unit_cell_keep_idx]
        ele = ele[unit_cell_keep_idx]
        
        supercell.from_geo_array(geo, ele)
        
        return supercell


    def get_vacuum_surface(self, SA_volume):
        """
        Obtain the surface coordinates between the slab and the vacuum. 
        This finds the first location in the vacuum direction that is not a 
        completely filled cube for each value in the two perpendicular 
        directions. 
        
        Returns
        -------
        np.array
            Modified volume with only surface voxel populated.
        
        """
        x_num,y_num,z_num = SA_volume.shape
        
        ## First, need to define the Z-direction to search over for each column
        ## Construct cubes in the z-direction
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
        
        
        ## Now, for each one of these Z columns of cubes, need to search over the xy 
        y_proj = np.arange(0,y_num-1)[:,None]*(z_num)
        front_plane_top_left_idx = front_plane_top_left_idx + y_proj
        front_plane_bot_left_idx = front_plane_bot_left_idx+ y_proj
        back_plane_top_left_idx = back_plane_top_left_idx+ y_proj
        back_plane_bot_left_idx = back_plane_bot_left_idx+ y_proj
        front_plane_top_right_idx = front_plane_top_right_idx+ y_proj
        front_plane_bot_right_idx = front_plane_bot_right_idx+ y_proj
        back_plane_top_right_idx = back_plane_top_right_idx+ y_proj
        back_plane_bot_right_idx = back_plane_bot_right_idx+ y_proj
        
        x_proj = np.arange(0,x_num-1)[:,None,None]*(y_num*z_num)
        front_plane_top_left_idx = front_plane_top_left_idx + x_proj
        front_plane_bot_left_idx = front_plane_bot_left_idx + x_proj
        back_plane_top_left_idx = back_plane_top_left_idx + x_proj
        back_plane_bot_left_idx = back_plane_bot_left_idx + x_proj
        front_plane_top_right_idx = front_plane_top_right_idx + x_proj
        front_plane_bot_right_idx = front_plane_bot_right_idx + x_proj
        back_plane_top_right_idx = back_plane_top_right_idx + x_proj
        back_plane_bot_right_idx = back_plane_bot_right_idx + x_proj
        
        ## Take the values for the cube vertices from the SA_volume
        front_plane_top_left_idx_mask = np.take(SA_volume, front_plane_top_left_idx)
        front_plane_bot_left_idx_mask = np.take(SA_volume, front_plane_bot_left_idx)
        back_plane_top_left_idx_mask = np.take(SA_volume, back_plane_top_left_idx)
        back_plane_bot_left_idx_mask = np.take(SA_volume, back_plane_bot_left_idx)
        front_plane_top_right_idx_mask = np.take(SA_volume, front_plane_top_right_idx)
        front_plane_bot_right_idx_mask = np.take(SA_volume, front_plane_bot_right_idx)
        back_plane_top_right_idx_mask = np.take(SA_volume, back_plane_top_right_idx)
        back_plane_bot_right_idx_mask = np.take(SA_volume, back_plane_bot_right_idx)
        
        ### Combine results from cube to find filled cubes
        combined = \
            np.logical_and(back_plane_bot_right_idx_mask,
            np.logical_and(back_plane_top_right_idx_mask,
            np.logical_and(front_plane_bot_right_idx_mask,
            np.logical_and(front_plane_top_right_idx_mask,
            np.logical_and(back_plane_bot_left_idx_mask,
            np.logical_and(back_plane_top_left_idx_mask,
            np.logical_and(front_plane_top_left_idx_mask,
                           front_plane_bot_left_idx_mask)))))))
        
        ### Iterate over combined results in the z direction
        keep_loc = []
        for idx_x in range(combined.shape[0]):
            for idx_y in range(combined.shape[1]):
                column_of_cubes = combined[idx_x,idx_y,:]
                filled_cube = np.where(column_of_cubes == 1)[0]
                if len(filled_cube) == 0:
                    continue
                
                max_z_pos = np.max(filled_cube)
                ## Move 1 up for surface position
                max_z_pos += 1
                
                keep_loc.append([idx_x,idx_y,max_z_pos])
        keep_loc = np.vstack(keep_loc)

        front_plane_top_left_idx_keep = front_plane_top_left_idx[keep_loc[:,0], 
                                                                 keep_loc[:,1], 
                                                                 keep_loc[:,2]]
        front_plane_bot_left_idx_keep = front_plane_bot_left_idx[keep_loc[:,0], 
                                                                 keep_loc[:,1], 
                                                                 keep_loc[:,2]]
        back_plane_top_left_idx_keep = back_plane_top_left_idx[keep_loc[:,0], 
                                                                 keep_loc[:,1], 
                                                                 keep_loc[:,2]]
        back_plane_bot_left_idx_keep = back_plane_bot_left_idx[keep_loc[:,0], 
                                                                 keep_loc[:,1], 
                                                                 keep_loc[:,2]]
        front_plane_top_right_idx_keep = front_plane_top_right_idx[keep_loc[:,0], 
                                                                 keep_loc[:,1], 
                                                                 keep_loc[:,2]]
        front_plane_bot_right_idx_keep = front_plane_bot_right_idx[keep_loc[:,0], 
                                                                 keep_loc[:,1], 
                                                                 keep_loc[:,2]]
        back_plane_top_right_idx_keep = back_plane_top_right_idx[keep_loc[:,0], 
                                                                 keep_loc[:,1], 
                                                                 keep_loc[:,2]]
        back_plane_bot_right_idx_keep = back_plane_bot_right_idx[keep_loc[:,0], 
                                                                 keep_loc[:,1], 
                                                                 keep_loc[:,2]]
        
        voxel_idx = np.c_[front_plane_top_left_idx_keep.ravel(),
                      front_plane_bot_left_idx_keep.ravel(),
                      back_plane_bot_left_idx_keep.ravel(),
                      back_plane_top_left_idx_keep.ravel(),
                      front_plane_top_right_idx_keep.ravel(),
                      front_plane_bot_right_idx_keep.ravel(),
                      back_plane_bot_right_idx_keep.ravel(),
                      back_plane_top_right_idx_keep.ravel(),
                      ]
        
        voxel_mask = np.take(SA_volume, voxel_idx)
        voxel_sum = np.sum(voxel_mask, axis=-1)
        voxel_surface_vertex_idx = np.where(np.logical_and(voxel_sum != 0,
                                             voxel_sum != 8))[0]
        surface_vertex_idx = voxel_idx[voxel_surface_vertex_idx][
                            voxel_mask[voxel_surface_vertex_idx].astype(bool)]
        
        final_volume = np.zeros(SA_volume.shape).ravel()
        final_volume[surface_vertex_idx] = 1
        final_volume = final_volume.reshape(SA_volume.shape)
        
        return final_volume
    
    
    def get_plot_info(self, plotly=True):
        """
        Return information used for plotting the surface of the molecule.
        
        Arguments
        ---------
        plotly: bool
            If True, will return colors in the form accepted by Plotly.
            If False, returns in array of floats between 0-1 for RGB
        
        """
        if self.struct == None:
            raise Exception("Must calculate structure before calling "+
                    "for plot information")
        
        tri_atom_idx = self.assign_triangles(
                            self.supercell, 
                            self.coords, 
                            self.triangles)
        
        if plotly:
            colors = self.get_plotly_colors(self.supercell, tri_atom_idx)
            
        else:
            ele = self.struct.geometry["element"]
            colors = [jmol_colors[atomic_numbers[ele[x]]].tolist() 
                      for x in tri_atom_idx]
        
        return self.coords,self.triangles,colors
    
    
    def assign_triangles(self, struct, edge_coords, triangles):
        """
        Assign triangles simply by computing nearest atom. 
        
        """
        tri_atom_idx = []
        
        ## Need to use atom adjust positions where positions are snapped to grid
        ## locations
        geo = self.m.centers
        geo_radii = [self.m.vdw[atomic_numbers[x]] 
                     for x in struct.geometry["element"]]
        
        for tri_idx in triangles:
            tri_coords = edge_coords[tri_idx]
            dist = cdist(tri_coords,geo)
            ## Turn absolute magnitude into a magnitude dependent on the radius of the atom
            dist = dist / geo_radii
            min_dist = np.min(dist)
            
            if min_dist <= 1.15:
                shortest_dist_atom_idx = np.argmin(dist) - int(
                       np.argmin(dist)/geo.shape[0])*geo.shape[0]
            else:
                shortest_dist_atom_idx = -1
                
            tri_atom_idx.append(shortest_dist_atom_idx)

        return tri_atom_idx
    
    
    def get_plotly_colors(self, struct, atom_idx):
        ele = struct.geometry["element"]
        
        colors = []
        for entry in atom_idx:
            if entry >= 0:
                colors.append(jmol_colors[atomic_numbers[ele[entry]]].tolist())
            else:
                colors.append([0.886,0.7765,1.0])
        
        plotly_colors = []
        for color in colors:
            plotly_colors.append("rgb({},{},{})".format(255*color[0],
                                                        255*color[1],
                                                        255*color[2]))
        return plotly_colors
    
    
    def get_roughness(self, mode="mean"):
        """
        Obtains roughness value for the relevent surface contour. This will use
        the highest edge position and compute the distance between all other 
        edge positions and this maximum z value as a measurement of the 
        roughness. 
        
        """
        if mode == "mean":
            max_z = np.max(self.coords[:,-1])
            return np.mean(max_z - self.coords[:,-1])
        else:
            raise Exception(
                    "Only mean roughness mode is implemented at this time.")
    
    

class Absorbate(SlabSurface):
    """
    Finding the optimal position for an absorbate on a surface. Algorithm 
    works as follows:
        
        1) Get 
        2) 
    
    
    """




    
if __name__ == "__main__":
    from pymove.io import read,write
    
    s = read("/Users/ibier/Research/Interfaces/Surface_Contour/20200531_Development/SCF/TETCEN.001.1.0.json")
    
    




