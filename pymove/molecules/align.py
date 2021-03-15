
import numpy as np

from ase.data import atomic_numbers,atomic_masses_iupac2016
from pymatgen.symmetry.analyzer import PointGroupAnalyzer as PGA

from pymove import Structure   
from pymove.molecules.utils import com
from pymove.molecules.symmetry import get_symmetry 


def align(struct, sym_kw={"max_rot": 4, "tol": 0.3}, recursion_count=0):
    """
    Aligns the molecule such that the COM is at the origin and the axes defined 
    by the moment of inertia tensor are oriented with the origin. In addition, 
    the operation is safe with respect to the symmetry of the molecule and the
    symmetry of the moment of inertia tensor with respect to the geometry of
    the molecule. 
    
    """
    ### First move COM of molecule to the origin
    trans = com(struct)
    geo = struct.get_geo_array()
    ele = struct.geometry["element"]
    geo = geo - trans
    struct.from_geo_array(geo, ele)
    
    principal_axes = get_principle_axes(struct)
    principle_axes,symmetric = _correct_principle_axes_symmetry(
                                    struct, principal_axes, sym_kw)
    rot = principle_axes.T
    
    ### If principal axes don't pass any symmetric checks, that it can be 
    ### known that the rotation must be applied to the geometry to properly 
    ### align it with the cartesian coordinate system. 
    if symmetric == False:
        geo =  np.dot(rot, geo.T).T
        struct.from_geo_array(geo, ele)
    
        ### Check if the newly aligned structure has actually aligned the 
        ### molecule with the origin. 
        ### Multiple alignments may need to be made if the molecule is highly 
        ### symmetric to converge the alignment process. 
        ### Recursion is an easy way to do this. 
        
        ### First check if more alignment is necessary
        new_principal_axes = get_principle_axes(struct)
        new_principle_axes,new_symmetric = _correct_principle_axes_symmetry(
                                    struct, new_principal_axes, sym_kw)
        
        ### Aligned structure did not lead to symmetric alignment with the
        ### origin
        if new_symmetric == False:
            #### Break recursion if tried too many times
            if recursion_count == 5:
                raise Exception("Alignment Failed. Is {} highly symmetric?"
                                .format(struct.struct_id))
            else:
                # print("Recurring: {}".format(recursion_count+1))
                align(struct, sym_kw, recursion_count+1)
    
    return struct
        
        
    
def _correct_principle_axes_symmetry(struct, principal_axes, sym_kw):
    geo = struct.get_geo_array()
    ele = struct.geometry["element"]
    
    ### Get symmetry of the molecule
    sym_ops = get_symmetry(struct, **sym_kw)
    
    ### Correct operation to apply to align axes with the origin is the inverse
    ### of the principles axes, which is just equal to its transpose. 
    rot = principal_axes.T
    
    ### Before applying rotation, check that this is not just a symmetry 
    ### operation of the molecule or a symmetry operation of the inertia
    ### tensor construction
    symmetric = False
    ### First check it's not identity already
    identity_check = np.linalg.norm(principal_axes - 
                                    np.array([[1.,0,0], [0,1,0], [0,0,1]]))
    if identity_check < 1e-4:
        symmetric = True
    
    ### Check principle axes against all molecular symmetry
    if symmetric == False:
        for entry in sym_ops:
            diff = np.linalg.norm(principal_axes - entry)
            if diff < 1e-4:
                principal_axes = np.array([[1.,0,0], [0,1,0], [0,0,1]])
                symmetric = True
                break
    
    if symmetric == False:
        #### Check for symmetry of principle axis under the operation of the 
        #### principle axes rotation. 
        rot_geo =  np.dot(rot, geo.T).T
        temp_struct = Structure.from_geo(rot_geo, ele)
        rot_principal_axes = get_principle_axes(temp_struct)
        
        diff = np.linalg.norm(principal_axes - rot_principal_axes)
        if diff < 1e-4:
            principal_axes = np.array([[1.,0,0], [0,1,0], [0,0,1]])
            symmetric = True
            # print("@@@@@@@@@@@ Priniciple Axis Symmetry @@@@@@@@@@@")
    
    #### Check for symmetry of principle axis with respect to the 
    #### molecular inversion symmetry
    if symmetric == False:
        identity_check = np.linalg.norm(np.abs(principal_axes) - 
                                        np.array([[1.,0,0], [0,1,0], [0,0,1]]))
        if identity_check < 1e-4:
            #### Check for inversion matrix in symmetry ops. Inversion symmetry
            #### can cause the principal axes operation to really be the 
            #### identity operation, but this is also not a symmetry operation 
            #### of the molecule. This is because the principal axes are 
            #### symmetric under more operations than the molecule itself. 
            inv = np.array([[-1,0,0], [0,-1,0], [0,0,-1]])
            for entry in sym_ops:
                diff = np.linalg.norm(inv - entry)
                if diff < 1e-8:
                    principal_axes = np.array([[1.,0,0], [0,1,0], [0,0,1]])
                    symmetric = True
                    # print("@@@@@@@@@@@ INVERSION @@@@@@@@@@@")
    
    return principal_axes,symmetric


def get_principle_axes(struct):
    """
    Obtain principle axes of inertia for the given structure. 
    
    """
    inertial_tensor = get_inertia_tensor(struct)
    ### SVD Decomposition to get axes of principle axes
    u,s,vh = np.linalg.svd(inertial_tensor)
    principal_axes = u
    return principal_axes


def get_inertia_tensor(struct):
    """
    Calculates inertia tensor of the given structure. 
    
    """
    geo = struct.get_geo_array()
    ele = struct.geometry["element"]
    mass =  np.array([atomic_masses_iupac2016[atomic_numbers[x]] for x in ele])
    
    inertial_tensor = np.zeros((3,3))
    ## Handle diangonal calculations first
    inertial_tensor[0,0] = np.sum(mass*(np.square(geo[:,1]) + np.square(geo[:,2])))
    inertial_tensor[1,1] = np.sum(mass*(np.square(geo[:,0]) + np.square(geo[:,2])))
    inertial_tensor[2,2] = np.sum(mass*(np.square(geo[:,0]) + np.square(geo[:,1])))
    
    ## Handle off diagonal terms
    inertial_tensor[0,1] = -np.sum(mass*geo[:,0]*geo[:,1])
    inertial_tensor[1,0] = inertial_tensor[0,1]
    
    inertial_tensor[0,2] = -np.sum(mass*geo[:,0]*geo[:,2])
    inertial_tensor[2,0] = inertial_tensor[0,2]
    
    inertial_tensor[1,2] = -np.sum(mass*geo[:,1]*geo[:,2])
    inertial_tensor[2,1] = inertial_tensor[1,2]
    
    total_inertia = np.dot(mass,np.sum(np.square(geo), axis=-1))
    inertial_tensor = inertial_tensor / total_inertia
    
    return inertial_tensor
    


def moit_pymatgen(struct):
    """
    Calculates the moment of inertia tensor for the system using Pymatgen. 
    However, this method is very slow due to Pymatgen performing many other 
    operations through the point group analyzer. It may still be used for
    validation purposes. Otherwise, use the high-performance moit function. 

    """
    mol = struct.get_pymatgen_structure()
    pga = PGA(mol)
    ax1,ax2,ax3 = pga.principal_axes
    return np.vstack([ax1,ax2,ax3])


def moit(struct):
    """
    Obtain the principle axes of the molecule using the moment of inertial 
    tensor.
    
    """
    inertial_tensor = np.zeros((3,3))
    
    geo = struct.get_geo_array()
    ele = struct.geometry["element"]
    mass =  np.array([atomic_masses_iupac2016[atomic_numbers[x]] for x in ele])
    
    ### First center molecule
    total = np.sum(mass)
    com = np.sum(geo*mass[:,None], axis=0)
    com = com / total
    geo = geo - com
    
    ## Handle diangonal calculations first
    inertial_tensor[0,0] = np.sum(mass*(np.square(geo[:,1]) + np.square(geo[:,2])))
    inertial_tensor[1,1] = np.sum(mass*(np.square(geo[:,0]) + np.square(geo[:,2])))
    inertial_tensor[2,2] = np.sum(mass*(np.square(geo[:,0]) + np.square(geo[:,1])))
    
    ## Handle off diagonal terms
    inertial_tensor[0,1] = -np.sum(mass*geo[:,0]*geo[:,1])
    inertial_tensor[1,0] = inertial_tensor[0,1]
    
    inertial_tensor[0,2] = -np.sum(mass*geo[:,0]*geo[:,2])
    inertial_tensor[2,0] = inertial_tensor[0,2]
    
    inertial_tensor[1,2] = -np.sum(mass*geo[:,1]*geo[:,2])
    inertial_tensor[2,1] = inertial_tensor[1,2]
    
    total_inertia = np.dot(mass,np.sum(np.square(geo), axis=-1))
    
    inertial_tensor = inertial_tensor / total_inertia
    eigvals, eigvecs = np.linalg.eig(inertial_tensor)
    
    ax1,ax2,ax3 = eigvecs.T
    principal_axes = np.vstack([ax1,ax2,ax3])
    
    return principal_axes


def orientation(struct):
    """
    Returns a rotation matrix for the moment of inertial tensor 
    for the given molecule Structure. 

    """
    axes = moit(struct)
    return np.linalg.inv(axes.T)
    

def show_axes(struct, ele="He"):
    """
    Visualize the COM of the molecule and the axes defined
    by the moment of inertial tensor of the molecule by adding
    an atom of type ele to the structure.

    """
    com_pos = com(struct) 
    axes = moit(struct)
    struct.append(com_pos[0],com_pos[1],com_pos[2],ele)
    for row in axes:
        row += com_pos
        struct.append(row[0],row[1],row[2],ele)


    
