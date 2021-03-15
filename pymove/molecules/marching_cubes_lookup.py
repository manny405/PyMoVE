
import numpy as np 
import scipy
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt


def basic_cube():
    """
    Cube based on ordering in program
    """
    return np.array([
       [-7.156285  , -3.80337925, -1.95817204],
       [-7.156285  , -3.80337925, -1.70817204],
       [-7.156285  , -3.55337925, -1.70817204],
       [-7.156285  , -3.55337925, -1.95817204],
       [-6.906285  , -3.80337925, -1.95817204],
       [-6.906285  , -3.80337925, -1.70817204],
       [-6.906285  , -3.55337925, -1.70817204],
       [-6.906285  , -3.55337925, -1.95817204]])


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


def unit_cube():
    return np.array([
            [0,0,0],
            [0,0,1],
            [0,1,1],
            [0,1,0],
            [1,0,0],
            [1,0,1],
            [1,1,1],
            [1,1,0]
            ])


def all_operations_vertex():
    
    def rot_opposite_faces_x(idx):
        return idx[[4,0,3,7,5,1,2,6]]

    def rot_opposite_faces_y(idx):
        return idx[[3,0,1,2,7,4,5,6]]
    
    def rot_opposite_faces_z(idx):
        return idx[[4,5,1,0,7,6,2,3]]
    
    def rot_cart_frame(idx):
        return idx[[0,4,5,1,3,7,6,2]]
    
    def rot_opposite_edges(idx):
        return idx[[4,7,6,5,0,3,2,1]]
    
    start_idx = np.arange(0,8)
    idx_list = [start_idx]
    for i in range(3):
        idx_list.append(rot_opposite_faces_x(idx_list[-1]))
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(3):
            rot_idx_list.append(rot_opposite_faces_y(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(3):
            rot_idx_list.append(rot_opposite_faces_z(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(2):
            rot_idx_list.append(rot_cart_frame(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(2):
            rot_idx_list.append(rot_opposite_edges(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    all_idx = np.vstack(idx_list)
#    all_idx = np.unique(all_idx,axis=0)
    return all_idx

def all_operations_edge(idx_list):
    
    def rot_opposite_faces_x(idx):
        return idx[[4,9,5,1,8,10,2,0,7,11,6,3]]

    def rot_opposite_faces_y(idx):
        return idx[[1,2,3,0,5,6,7,4,9,10,11,8]]
    
    def rot_opposite_faces_z(idx):
        return idx[[8,4,0,7,9,1,3,11,10,5,2,6]]
    
    def rot_cart_frame(idx):
        return idx[[4,0,7,8,1,3,11,9,5,2,6,10]]
    
    def rot_opposite_edges(idx):
        return idx[[9,8,11,10,4,7,6,5,1,0,3,2]]
    
    start_idx = np.arange(0,12)
    idx_list = [start_idx]
    for i in range(3):
        idx_list.append(rot_opposite_faces_x(idx_list[-1]))
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(3):
            rot_idx_list.append(rot_opposite_faces_y(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(3):
            rot_idx_list.append(rot_opposite_faces_z(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(2):
            rot_idx_list.append(rot_cart_frame(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(2):
            rot_idx_list.append(rot_opposite_edges(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    all_idx = np.vstack(idx_list)
#    all_idx = np.unique(all_idx,axis=0)
    
    return all_idx


def apply_vertex_symmetry(vertex_idx):
    #### Let's perform all rotations on lookup idx first
    
    symmetry_idx_list = all_operations_vertex()
    
    all_vertex_idx = []
    
    for idx_list in symmetry_idx_list:
        all_vertex_idx.append(vertex_idx[idx_list])
    
    return all_vertex_idx

def apply_edge_symmetry(edge_idx):
        
        
    ### Construct corresponding symmetry relevant ordings for vertex/edge 
    ### for triangulation
    edge_symmetry_idx_list = all_operations_edge(edge_idx)
    edge_symmetry_idx_list = np.array(edge_symmetry_idx_list)
    
    all_edge_idx = []
    for row in edge_symmetry_idx_list:  
        all_edge_idx.append(edge_idx[:,row])
    
    return all_edge_idx



########### Let's build the vertex lookup table
all_comb = np.meshgrid([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1])
all_comb = np.c_[
        all_comb[0].ravel(),
        all_comb[1].ravel(),
        all_comb[2].ravel(),
        all_comb[3].ravel(),
        all_comb[4].ravel(),
        all_comb[5].ravel(),
        all_comb[6].ravel(),
        all_comb[7].ravel()]
vertex_lookup = np.zeros((2,2,2,2,2,2,2,2,12))
    

def tostring(array):
    """
    1D array to string
    """
    return ",".join([str(x) for x in array])

def fromstring(array_str):
    return np.fromstring(array_str, dtype=int, sep=",")

## Program fourteen primitives
## https://www.researchgate.net/publication/3410984_Brodlie_K_Improving_the_robustness_and_accuracy_of_the_marching_cubes_algorithm_for_isosurfacing_IEEE_Trans_Viz_and_Comput_Graph_91_16-29/figures?lo=1

#### For holding information to operate on using symmetry operations and store 
#### tri connectivity
vertex_mask_idx = np.zeros((15,8)).astype(int)
tri_mask = np.zeros((16,12))

#### Build connectivity dict for these simple cases
#### Each entry is 2D array with one entry per connectivity and number of entries
#### equal to the number of triangles
tri_connectivity = {}

#### Same as tri_connectivity but populated with volume information for volume
#### adjustments to be made for each type. 
#### Valued entered is a ratio out of 1 with respect to the volume of the 
#### voxel that the entry adds. 
tri_volume = {}
tri_volume_modifier = {}

### I will define the edges and the verticies that make up the shape that is
### needed to calculate the volume using a ConvexHull method. Data type 
### is such that for each entry, there will be list of shapes that need to be 
### evaluated. Each shape is defined by a tuple with the first being a mask for 
### vertices and the second being a mask for edges. 
volume_shape_mask = {}



#### 1. First entry all zeros
entry = vertex_mask_idx[0]
tri_connectivity[tostring(entry)] = np.zeros((1,12))
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
## Set volume
tri_volume[tostring(entry)] = 0
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1

#### 2. Simple Triangle
vertex_mask_idx[1,[0]] = 1
tri_mask[1,[0,1,4]] = 1
entry = vertex_mask_idx[1]
tri_connectivity[tostring(entry)] = np.zeros((1,12))
tri_connectivity[tostring(entry)][0][[0,1,4]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 3. Simple Plane down
vertex_mask_idx[2,[0,4]] = 1
tri_mask[2,[0,1,8,9]] = 1
entry = vertex_mask_idx[2]
tri_connectivity[tostring(entry)] = np.zeros((2,12))
tri_connectivity[tostring(entry)][0][[0,1,9]] = 1
tri_connectivity[tostring(entry)][1][[0,8,9]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.125
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 4. Across face double triangle
vertex_mask_idx[3,[0,5]] = 1
## First Tri
tri_mask[3,[0,1,4]] = 1
## Second Tri
tri_mask[3,[7,8,11]] = 1
entry = vertex_mask_idx[3]
tri_connectivity[tostring(entry)] = np.zeros((2,12))
tri_connectivity[tostring(entry)][0][[0,1,4]] = 1
tri_connectivity[tostring(entry)][1][[7,8,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 2*0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 5. Across body double triangle
vertex_mask_idx[4,[0,6]] = 1
## First Tri
tri_mask[4,[0,1,4]] = 1
## Second Tri
tri_mask[4,[6,10,11]] = 1
entry = vertex_mask_idx[4]
tri_connectivity[tostring(entry)] = np.zeros((2,12))
tri_connectivity[tostring(entry)][0][[0,1,4]] = 1
tri_connectivity[tostring(entry)][1][[6,10,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 2*0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 6. Three Bottom Corners
vertex_mask_idx[5,[3,4,7]] = 1
tri_mask[5,[1,4,8,10,2]] = 1
entry = vertex_mask_idx[5]
tri_connectivity[tostring(entry)] = np.zeros((3,12))
tri_connectivity[tostring(entry)][0][[1,4,8]] = 1
tri_connectivity[tostring(entry)][1][[1,8,2]] = 1
tri_connectivity[tostring(entry)][2][[2,8,10]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.35416667
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 7. One plane down one tri
vertex_mask_idx[6,[0,4,6]] = 1
## Plane down
tri_mask[6,[0,8,1,9]] = 1
## Upper Tri 6
tri_mask[6,[6,10,11]] = 1
entry = vertex_mask_idx[6]
tri_connectivity[tostring(entry)] = np.zeros((3,12))
tri_connectivity[tostring(entry)][0][[0,1,9]] = 1
tri_connectivity[tostring(entry)][1][[0,8,9]] = 1
tri_connectivity[tostring(entry)][2][[6,10,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.125+0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 8. Triple Tri
vertex_mask_idx[7,[1,4,6]] = 1
## Tri 1
tri_mask[7,[0,3,7]] = 1
## Tri 4
tri_mask[7,[4,8,9]] = 1
## Tri 6
tri_mask[7,[6,10,11]] = 1
entry = vertex_mask_idx[7]
tri_connectivity[tostring(entry)] = np.zeros((3,12))
tri_connectivity[tostring(entry)][0][[0,3,7]] = 1
tri_connectivity[tostring(entry)][1][[4,8,9]] = 1
tri_connectivity[tostring(entry)][2][[6,10,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 3*0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 9. Middle Plane
vertex_mask_idx[8,[0,3,4,7]] = 1
## Mid Plane
tri_mask[8,[0,2,8,10]] = 1
entry = vertex_mask_idx[8]
tri_connectivity[tostring(entry)] = np.zeros((2,12))
tri_connectivity[tostring(entry)][0][[0,8,10]] = 1
tri_connectivity[tostring(entry)][1][[0,2,10]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.5
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 10. Hexagon
vertex_mask_idx[9,[0,2,3,7]] = 1
## Hexagon
tri_mask[9,[0,3,4,6,9,10]] = 1
entry = vertex_mask_idx[9]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[0,3,6]] = 1
tri_connectivity[tostring(entry)][1][[0,6,10]] = 1
tri_connectivity[tostring(entry)][2][[0,9,10]] = 1
tri_connectivity[tostring(entry)][3][[0,4,9]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.375
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 11. Double Plane
vertex_mask_idx[10,[0,1,6,7]] = 1
## Plane 1
tri_mask[10,[1,3,4,7]] = 1
## Plane 2
tri_mask[10,[5,6,9,11]] = 1
entry = vertex_mask_idx[10]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[1,3,7]] = 1
tri_connectivity[tostring(entry)][1][[1,4,7]] = 1
tri_connectivity[tostring(entry)][2][[5,6,11]] = 1
tri_connectivity[tostring(entry)][3][[5,9,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.75
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 12. 
vertex_mask_idx[11,[0,3,6,7]] = 1
## Plane
tri_mask[11,[0,2,4,6,9,11]] = 1
entry = vertex_mask_idx[11]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[4,9,11]] = 1
tri_connectivity[tostring(entry)][1][[2,6,11]] = 1
tri_connectivity[tostring(entry)][2][[0,2,4]] = 1
tri_connectivity[tostring(entry)][3][[2,4,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.375
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 13. 6+tri
vertex_mask_idx[12,[1,3,4,7]] = 1
## 6 Plane
tri_mask[12,[1,4,8,10,2]] = 1
## Tri 1
tri_mask[12,[0,3,7]] = 1
entry = vertex_mask_idx[12]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[1,4,8]] = 1
tri_connectivity[tostring(entry)][1][[1,8,2]] = 1
tri_connectivity[tostring(entry)][2][[2,8,10]] = 1
tri_connectivity[tostring(entry)][3][[0,3,7]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.5
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1



#### 14. Quad Tri
vertex_mask_idx[13,[0,2,5,7]] = 1
## Tri 0
tri_mask[13,[0,1,4]] = 1
## Tri 2
tri_mask[13,[2,3,6]] = 1
## Tri 5
tri_mask[13,[7,8,11]] = 1
## Tri 7
tri_mask[13,[5,9,10]] = 1
entry = vertex_mask_idx[13]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[0,1,4]] = 1
tri_connectivity[tostring(entry)][1][[2,3,6]] = 1
tri_connectivity[tostring(entry)][2][[7,8,11]] = 1
tri_connectivity[tostring(entry)][3][[5,9,10]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 4*0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### 15. 
vertex_mask_idx[14,[2,3,4,7]] = 1
entry = vertex_mask_idx[14]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[1,3,4]] = 1
tri_connectivity[tostring(entry)][1][[4,3,10]] = 1
tri_connectivity[tostring(entry)][2][[3,6,10]] = 1
tri_connectivity[tostring(entry)][3][[4,8,10]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.375
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]
tri_volume_modifier[tostring(entry)] = 0
tri_volume_modifier[tostring(not_entry)] = 1


#### Performing rotations to populate the entire tri_connectivity
iterations = [(keys,values) for keys,values in tri_connectivity.items()]
for key,value in iterations:
    key_array = fromstring(key)
    all_vertex = apply_vertex_symmetry(key_array)
    all_edge = apply_edge_symmetry(value)
    
    for temp_idx,vertex in enumerate(all_vertex):
        tri_connectivity[tostring(vertex)] = all_edge[temp_idx]

iterations = [(keys,values) for keys,values in tri_volume.items()]
for key,value in iterations:
    key_array = fromstring(key)
    all_vertex = apply_vertex_symmetry(key_array)
    
    for temp_idx,vertex in enumerate(all_vertex):
        tri_volume[tostring(vertex)] = value

iterations = [(keys,values) for keys,values in tri_volume_modifier.items()]
for key,value in iterations:
    key_array = fromstring(key)
    all_vertex = apply_vertex_symmetry(key_array)
    
    for temp_idx,vertex in enumerate(all_vertex):
        tri_volume_modifier[tostring(vertex)] = value
        

#### Plotting all primitives
def plot_primitives(figname="marching_cubes_primitive_no_numbers.pdf"):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    cart_points = basic_cube()
    fig = plt.figure(figsize=(24,24))
    
    for entry_idx,vertex_row in enumerate(vertex_mask_idx):
        ax = fig.add_subplot(4,4,entry_idx+1, projection='3d')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.scatter(cart_points[:,0][0:8],
                   cart_points[:,1][0:8],
                   cart_points[:,2][0:8],
                   facecolor=(0,0,0,0),
                   edgecolor="k",
                   s=100)
        
#        ## Add numbering
#        for idx,point in enumerate(cart_points[0:8]):
#            ax.text(point[0],
#                   point[1], 
#                   point[2],
#                   "{}".format(idx),
#                   fontsize=16)
            
        cube_vertex = cart_points[:8]
        edge_vertex = compute_edge_sites(cube_vertex)
        
        #### Visualize edge points
        ax.scatter(edge_vertex[:,0],
                   edge_vertex[:,1],
                   edge_vertex[:,2],
                   edgecolor="k",
                   facecolor="tab:red",
                   s=100)
        
#        ## Number edge cites
#        for idx,point in enumerate(edge_vertex):
#            ax.text(point[0],
#                   point[1], 
#                   point[2],
#                   "{}".format(idx),
#                   fontsize=16)
        
        ## Plot relevant vertices
        vertex_row_bool = vertex_row.astype(bool)
        temp_vertex = cart_points[vertex_row_bool,:]
        if len(temp_vertex) > 0:
            ax.scatter(
                    temp_vertex[:,0],
                    temp_vertex[:,1],
                    temp_vertex[:,2],
                    c="tab:green",
                    edgecolor="k",
                    s=100)
        ## Tri idx
        entry = tostring(vertex_row)
        triangles_bool = tri_connectivity[entry].astype(bool)
        array_to_mask = np.repeat(np.arange(0,12)[None,:], 
                                triangles_bool.shape[0], 
                                axis=0)
        tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
        
        if len(tri_idx) != 0:
            ax.plot_trisurf(
                    edge_vertex[:,0],
                    edge_vertex[:,1],
                    edge_vertex[:,2],
                    triangles=tri_idx)
    
    fig.savefig(figname, 
                dpi=400)


##### Plotting all in tri_connectivity
def plot_all_cubes(figname="all_marching_cubes.pdf"):
    cart_points = basic_cube()
    fig = plt.figure(figsize=(48,192))
    
    entry_idx = 0
    for key,value in tri_connectivity.items():
        ax = fig.add_subplot(32,8,entry_idx+1, projection='3d')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.scatter(cart_points[:,0][0:8],
                   cart_points[:,1][0:8],
                   cart_points[:,2][0:8])
        
        ## Add numbering
        for idx,point in enumerate(cart_points[0:8]):
            ax.text(point[0],
                   point[1], 
                   point[2],
                   "{}".format(idx),
                   fontsize=16)
            
        cube_vertex = cart_points[:8]
        edge_vertex = compute_edge_sites(cube_vertex)
        
        #### Visualize edge points
        ax.scatter(edge_vertex[:,0],
                   edge_vertex[:,1],
                   edge_vertex[:,2],
                   edgecolor="k",
                   facecolor="tab:red")
        
        ## Number edge cites
        for idx,point in enumerate(edge_vertex):
            ax.text(point[0],
                   point[1], 
                   point[2],
                   "{}".format(idx),
                   fontsize=16)
            
        ## Plot Triangle
        triangles_bool = value.astype(bool)
        array_to_mask = np.repeat(np.arange(0,12)[None,:], 
                                triangles_bool.shape[0], 
                                axis=0)
        tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
        
        if len(tri_idx) != 0:
            ax.plot_trisurf(
                    edge_vertex[:,0],
                    edge_vertex[:,1],
                    edge_vertex[:,2],
                    triangles=tri_idx)
        
        entry_idx += 1
    
    fig.savefig("all_marching_cubes.pdf")



##### Deriviing volumes for each primitive
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#
cart_points = unit_cube()


#flat_rows = {}
#flat_rows[tostring(np.array([1,1,0,0,0,0,0,0]))] = 1
#all_vertex = apply_vertex_symmetry(np.array([1,1,0,0,0,0,0,0]))
#for entry in all_vertex:
#    flat_rows[tostring(entry)] = 1
#flat_rows[tostring(np.array([1,1,1,1,0,0,0,0]))] = 1
#all_vertex = apply_vertex_symmetry(np.array([1,1,1,1,0,0,0,0]))
#for entry in all_vertex:
#    flat_rows[tostring(entry)] = 1
#    
#
#def get_volume(vertex_row, edges):
#    """
#    Arguments
#    ---------
#    vertex_row: array 
#        Array of shape (8,) equal to a binary mask of all of the populated voxels
#    edges: array
#        Arry of shape (12,0) with edges, either normal or projected
#    
#    """
#    edge_vertex=edges
#    
#    ## Check that the surface will not be flat with respec to the 
#    ## Z direction
#    if tostring(vertex_row) in flat_rows:
#        print("FLAT")
#        try: 
#            triangles_bool = tri_connectivity[tostring(vertex_row)].astype(bool)
#            array_to_mask = np.repeat(np.arange(0,12)[None,:], 
#                                   triangles_bool.shape[0], 
#                                   axis=0)
#            tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
#
#            all_tri = edge_vertex[tri_idx]
#            vol = 0
#            for tri_entry in all_tri[:-1]:
#                xyz = tri_entry
#                d = scipy.spatial.Delaunay(xyz[:,:2])
#        except:
#            rot_matrix = np.array([[1,0,0],[0,0,1],[0,1,0]])
#            edge_vertex = np.dot(rot_matrix,edge_vertex.T).T
#    
#    case_11 = False
#    if np.linalg.norm(vertex_row - np.array([1, 1, 0, 0, 0, 0, 1, 1])) < 1e-3:
#        case_11 = True
#    ## Logical NOT
#    elif np.linalg.norm(vertex_row - np.array([0, 0, 1, 1, 1, 1, 0, 0])) < 1e-3:
#        case_11 = True
#        
#    case_13 = False
#    if np.linalg.norm(vertex_row - np.array([0, 1, 0, 1, 1, 0, 0, 1])) == 0:
#        case_13 = True
#    elif np.linalg.norm(vertex_row - np.array([1, 0, 1, 0, 0, 1, 1, 0])) == 0:
#        case_13 = True
#    
#    ## Handle Case 11 with two planes in Z direction
#    if case_11:
#        entry = tostring(vertex_row)
#        triangles_bool = tri_connectivity[entry].astype(bool)
#        array_to_mask = np.repeat(np.arange(0,12)[None,:], 
#                               triangles_bool.shape[0], 
#                               axis=0)
#        tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
#
#        rot_matrix = np.array([[1,0,0],[0,0,1],[0,1,0]])
#        
#        edge_vertex = np.dot(rot_matrix,edge_vertex.T).T
#        all_tri = edge_vertex[tri_idx]
#
#        vol = 0
#        for tri_entry in all_tri[0:2]:
#            xyz = tri_entry
#            d = scipy.spatial.Delaunay(xyz[:,:2])
#            tri = xyz[d.vertices]
#            a = tri[:,0,:2] - tri[:,1,:2]
#            b = tri[:,0,:2] - tri[:,2,:2]
#            proj_area = np.cross(a, b).sum(axis=-1)
#            zavg = tri[:,:,2].sum(axis=1)
#            vol += np.abs(zavg * np.abs(proj_area) / 6.0)
#            
#        for entry in all_tri[2:]:
#            xyz = tri_entry
#            d = scipy.spatial.Delaunay(xyz[:,:2])
#            tri = xyz[d.vertices]
#            a = tri[:,0,:2] - tri[:,1,:2]
#            b = tri[:,0,:2] - tri[:,2,:2]
#            proj_area = np.cross(a, b).sum(axis=-1)
#            zavg = tri[:,:,2].sum(axis=1)
#            vol += np.abs(zavg * np.abs(proj_area) / 6.0)
#        
#        return vol
#    
#    ### Handle case 13 with missing corner
#    if case_13:
#        entry = tostring(vertex_row)
#        triangles_bool = tri_connectivity[entry].astype(bool)
#        array_to_mask = np.repeat(np.arange(0,12)[None,:], 
#                               triangles_bool.shape[0], 
#                               axis=0)
#        tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
#
#        all_tri = edge_vertex[tri_idx]
#        vol = 0
#        for tri_entry in all_tri[:-1]:
#            xyz = tri_entry
#            d = scipy.spatial.Delaunay(xyz[:,:2])
#            tri = xyz[d.vertices]
#            a = tri[:,0,:2] - tri[:,1,:2]
#            b = tri[:,0,:2] - tri[:,2,:2]
#            proj_area = np.cross(a, b).sum(axis=-1)
#            zavg = tri[:,:,2].sum(axis=1)
#            vol += np.abs(zavg * np.abs(proj_area) / 6.0)
#        
#        tri_entry = all_tri[-1]
#        xyz = tri_entry
#        d = scipy.spatial.Delaunay(xyz[:,:2])
#        tri = xyz[d.vertices]
#        a = tri[:,0,:2] - tri[:,1,:2]
#        b = tri[:,0,:2] - tri[:,2,:2]
#        proj_area = np.cross(a, b).sum(axis=-1)
#        zavg = tri[:,:,2].sum(axis=1)
#        vol -= np.abs(zavg * np.abs(proj_area) / 6.0)
#        
#        return vol
#
#    ## Tri idx
#    entry = tostring(vertex_row)
#    triangles_bool = tri_connectivity[entry].astype(bool)
#    array_to_mask = np.repeat(np.arange(0,12)[None,:], 
#                           triangles_bool.shape[0], 
#                           axis=0)
#    tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
#
#    all_tri = edge_vertex[tri_idx]
#    vol = 0
#    for tri_entry in all_tri:
#        xyz = tri_entry
#        try:
#            d = scipy.spatial.Delaunay(xyz[:,:2])
#        except:
#            continue
#        tri = xyz[d.vertices]
#        a = tri[:,0,:2] - tri[:,1,:2]
#        b = tri[:,0,:2] - tri[:,2,:2]
#        proj_area = np.cross(a, b).sum(axis=-1)
#        zavg = tri[:,:,2].sum(axis=1)
#        vol += np.abs(zavg * np.abs(proj_area) / 6.0)
#
#    return vol


### Vertex Mask idx are all of the primitive entries. 
### This leads to 163 entires
all_unique = []
primitive_dict = {}
for row in vertex_mask_idx:
    row_str = tostring(row)
    primitive_dict[row_str] = {row_str: 1}
    all_temp = np.vstack(apply_vertex_symmetry(row))
    
    ### Algorithm for removing duplicate rows
    R,C = np.triu_indices(all_temp.shape[0],1)
    mask = (np.abs(all_temp[R] - all_temp[C]) < 1e-3).all(axis=(1))
    I,G = R[mask], C[mask]
    remove_idx = np.unique(G)
    original_idx = np.arange(0,all_temp.shape[0])
    final_idx = np.setdiff1d(original_idx,remove_idx)
    all_temp = all_temp[final_idx]
    
    all_unique.append(all_temp)
    
    for entry in all_temp:
        primitive_dict[row_str][tostring(entry)] = 1
        
### Now do same operation but for the not of the primitive
not_primitive_dict = {}
for row in vertex_mask_idx:
    row = np.logical_not(row).astype(int)
    row_str = tostring(row)
    not_primitive_dict[row_str] = {row_str: 1}
    all_temp = np.vstack(apply_vertex_symmetry(row))
    
    ### Algorithm for removing duplicate rows
    R,C = np.triu_indices(all_temp.shape[0],1)
    mask = (np.abs(all_temp[R] - all_temp[C]) < 1e-3).all(axis=(1))
    I,G = R[mask], C[mask]
    remove_idx = np.unique(G)
    original_idx = np.arange(0,all_temp.shape[0])
    final_idx = np.setdiff1d(original_idx,remove_idx)
    all_temp = all_temp[final_idx]
    
    all_unique.append(all_temp)
    
    for entry in all_temp:
        not_primitive_dict[row_str][tostring(entry)] = 1

### Defines the correct edges that form a triangle given the vertex of the cube 
triangles = {}
triangles[0] = [0,1,4]
triangles[1] = [0,3,7]
triangles[2] = [2,3,6]
triangles[3] = [1,2,5]
triangles[4] = [4,8,9]
triangles[5] = [7,8,11]
triangles[6] = [6,10,11]
triangles[7] = [5,9,10]

## Nearest neighbors along edges of cube
nearest_neighbors = {}
nearest_neighbors[0] = [1,3,4]
nearest_neighbors[1] = [0,2,5]
nearest_neighbors[2] = [1,3,6]
nearest_neighbors[3] = [0,2,7]
nearest_neighbors[4] = [0,5,7]
nearest_neighbors[5] = [1,4,6]
nearest_neighbors[6] = [2,5,7]
nearest_neighbors[7] = [3,4,6]

### For primitive with plane and triangle to determine where the plane is 
### and where the triangle is
plane_tri_dict = {}
for entry in primitive_dict["1,0,0,0,1,0,1,0"].keys():
    temp_array = fromstring(entry)
    pos_idx = np.where(temp_array == 1)[0]
    for idx in pos_idx:
        plane = [idx]
        nn = nearest_neighbors[idx]
        for value in nn:
            if value in pos_idx:
                plane.append(value)
                break
        if len(plane) == 2:
            tri_idx = np.setdiff1d(pos_idx, plane)
            break
    
    plane_tri_dict[entry] = plane
    



def get_volume(vertex_row, vert, edges):
    """
    Algorithm is as follows:
        1. Identify what the primitive shape should be
        2. Rotate into original? No, can't do that easily
        3. Calculate the volume correctly based on this primitive shape
    
    Arguments
    ---------
    vertex_row: array 
        Array of shape (8,) equal to a binary mask of all of the populated voxels
    vert: array
        Array of shape (8,3) for cartesian positions of the cube. 
    edges: array
        Arry of shape (12,3) with edges, either normal or projected
    
    """
    row_str = tostring(vertex_row)
    
    
    found = False
    primitive = []
    not_value = False
    for key,value in primitive_dict.items():
        if row_str in value:
            found = True
            primitive = key
            break
    if not found:
        for key,value in not_primitive_dict.items():
            if row_str in value:
                primitive = key
                found = True
                not_value = True
#                 raise Exception("Just need to use logical_not as primitive and then just do 1-final volume."+
#                                 " {}".format(row_str))

                ## Now go back over to find other equivalent
                vertex_row = np.logical_not(vertex_row).astype(int)
                row_str = tostring(vertex_row)
                found = False
                for key,value in primitive_dict.items():
                    if row_str in value:
                        found = True
                        primitive = key
                        break
                break
                
    if found == False:
        raise Exception("{}".format(row_str))
        
#     print(row_str,primitive)
    triangles_bool = tri_connectivity[row_str].astype(bool)
    ## Mask to get active vertices
    active_vert = vert[vertex_row.astype(bool)]
    
    
    if primitive == '0,0,0,0,0,0,0,0':
        return 0
    ## One triangle
    elif primitive == '1,0,0,0,0,0,0,0':
        active_edges = edges[triangles_bool[0]]
        shape_vert = np.vstack([active_edges,active_vert])
        try:
            vol = ConvexHull(shape_vert).volume
        except:
            vol = 0
    ## Plane
    elif primitive == '1,0,0,0,1,0,0,0':
        plane_edges_mask = np.logical_or(triangles_bool[0],triangles_bool[1])
        plane_edges =  edges[plane_edges_mask]
        shape_vert = np.vstack([active_vert,plane_edges])
        try:
            vol = ConvexHull(shape_vert).volume
        except:
            vol = 0
    ## Double Triangle
    elif primitive == '1,0,0,0,0,1,0,0':
        tri_idx = np.where(vertex_row == 1)[0]
        vol = 0
        for idx in tri_idx:
            ## Get edge idx for known triangle orientations
            tri_edges_idx = triangles[idx]
            tri_edges = edges[tri_edges_idx]
            
            ## Get vert for this triangle
            temp_active_vert = vert[idx]
            
            temp_shape_vert = np.vstack([tri_edges,temp_active_vert])
            try:
                temp_vol = ConvexHull(temp_shape_vert).volume
                vol += temp_vol
            except:
                pass
    ## Double Triangle body diagonal
    elif primitive == '1,0,0,0,0,0,1,0':
        tri_idx = np.where(vertex_row == 1)[0]
        vol = 0
        for idx in tri_idx:
            ## Get edge idx for known triangle orientations
            tri_edges_idx = triangles[idx]
            tri_edges = edges[tri_edges_idx]
            
            ## Get vert for this triangle
            temp_active_vert = vert[idx]
            
            temp_shape_vert = np.vstack([tri_edges,temp_active_vert])
            try:
                temp_vol = ConvexHull(temp_shape_vert).volume
                vol += temp_vol
            except:
                pass
    ## Strange shape, but it can safely be evaluated
    ## Three bottom corners
    elif primitive == '0,0,0,1,1,0,0,1':
        ## Just get all active edges
        plane_edges_mask = np.sum(triangles_bool,axis=0).astype(bool)
        plane_edges =  edges[plane_edges_mask]
        shape_vert = np.vstack([active_vert,plane_edges])
        try:
            vol = ConvexHull(shape_vert).volume
        except:
            vol = 0
    ## One plane one triangle
    elif primitive == '1,0,0,0,1,0,1,0':
#        raise Exception("HARD TO EVALUATE")
        pos_idx = np.where(vertex_row == 1)[0]
        plane_idx = plane_tri_dict[row_str]
        tri_idx = np.setdiff1d(pos_idx, plane)[0]
        
        ## Get triangle shape
        tri_edge_idx = triangles[tri_idx]
        tri_edges = edges[tri_edge_idx]
        tri_vert = vert[tri_idx]
        tet_vert = np.vstack([tri_vert,tri_edges])
        tri_vol = 0
        try: 
            tri_vol = ConvexHull(tet_vert).volume
        except:
            pass
        
        ## Get plane shape
        plane_vert = vert[plane_idx]
        ## Get edges manually
        plane_only_row = np.zeros((8,))
        plane_only_row[plane_idx] = 1
        plane_edges_mask = tri_connectivity[tostring(plane_only_row.astype(int))
                                            ].astype(bool)
        plane_edges_mask = np.logical_or(triangles_bool[0],triangles_bool[1])
        plane_edges = edges[plane_edges_mask]
        plane_vert = np.vstack([plane_vert,plane_edges])
        plane_vol = 0
        try:
            plane_vol = ConvexHull(plane_vert).volume
        except:
            pass
        
        return tri_vol + plane_vol
          
    ## Three Triangles
    elif primitive == '0,1,0,0,1,0,1,0':
        ## Just iterate over triangles
        tri_idx = np.where(vertex_row == 1)[0]
        vol = 0
        for idx in tri_idx:
            ## Get edge idx for known triangle orientations
            tri_edges_idx = triangles[idx]
            tri_edges = edges[tri_edges_idx]
            
            ## Get vert for this triangle
            temp_active_vert = vert[idx]
            
            temp_shape_vert = np.vstack([tri_edges,temp_active_vert])
            try:
                temp_vol = ConvexHull(temp_shape_vert).volume
                vol += temp_vol
            except:
                pass
    ## Simple plane, can just cat together
    elif primitive == '1,0,0,1,1,0,0,1':
        ## Just get all active edges
        plane_edges_mask = np.sum(triangles_bool,axis=0).astype(bool)
        plane_edges =  edges[plane_edges_mask]
        shape_vert = np.vstack([active_vert,plane_edges])
        try:
            vol = ConvexHull(shape_vert).volume
        except:
            vol = 0
    ## Hexagon, just cat together
    elif primitive == '1,0,1,1,0,0,0,1':
        ## Just get all active edges
        plane_edges_mask = np.sum(triangles_bool,axis=0).astype(bool)
        plane_edges =  edges[plane_edges_mask]
        shape_vert = np.vstack([active_vert,plane_edges])
        try:
            vol = ConvexHull(shape_vert).volume
        except:
            vol = 0
    ## Double plane. I think everything can just be cat together
    elif primitive == '1,1,0,0,0,0,1,1':
        ## Just get all active edges
        plane_edges_mask = np.sum(triangles_bool,axis=0).astype(bool)
        plane_edges =  edges[plane_edges_mask]
        shape_vert = np.vstack([active_vert,plane_edges])
        ## Using 1 minus because it will be the more common case for the 
        ## moleucles
        try:
            vol = 1-ConvexHull(shape_vert).volume
        except:
            vol = 0
    ## Weird, but can just stick everything together
    elif primitive == '1,0,0,1,0,0,1,1':
        ## Just get all active edges
        plane_edges_mask = np.sum(triangles_bool,axis=0).astype(bool)
        plane_edges =  edges[plane_edges_mask]
        shape_vert = np.vstack([active_vert,plane_edges])
        try:
            vol = ConvexHull(shape_vert).volume
        except:
            vol = 0
    ## Weird, but can just stick everything together
    elif primitive == '0,1,0,1,1,0,0,1':
        ## Just get all active edges
        plane_edges_mask = np.sum(triangles_bool,axis=0).astype(bool)
        plane_edges =  edges[plane_edges_mask]
        shape_vert = np.vstack([active_vert,plane_edges])
        try:
            vol = ConvexHull(shape_vert).volume
        except:
            vol = 0
    ## Four triangles
    elif primitive == '1,0,1,0,0,1,0,1':
        ## Just iterate over triangles
        tri_idx = np.where(vertex_row == 1)[0]
        vol = 0
        for idx in tri_idx:
            ## Get edge idx for known triangle orientations
            tri_edges_idx = triangles[idx]
            tri_edges = edges[tri_edges_idx]
            
            ## Get vert for this triangle
            temp_active_vert = vert[idx]
            
            temp_shape_vert = np.vstack([tri_edges,temp_active_vert])
            try:
                temp_vol = ConvexHull(temp_shape_vert).volume
                vol += temp_vol
            except:
                pass
    elif primitive == '0,0,1,1,1,0,0,1':
        ## Just get all active edges
        plane_edges_mask = np.sum(triangles_bool,axis=0).astype(bool)
        plane_edges =  edges[plane_edges_mask]
        shape_vert = np.vstack([active_vert,plane_edges])
        try:
            vol = ConvexHull(shape_vert).volume
        except:
            vol = 0
    else:
        raise Exception("PRIMITIVE NOT FOUND for {}".format(primitive))
    
    if not_value:
        ## First compute volume for entire cube
        spacing = np.linalg.norm(vert[0] - vert[1])
        cube_vol = spacing*spacing*spacing
        return cube_vol - vol
    else:
        return vol



if __name__ == "__main__":
    pass