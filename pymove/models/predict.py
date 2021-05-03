
import numpy as np 

from pymove.driver import BaseDriver_
from pymove.models.trained import features as trained_features
from pymove.models.trained import coef as trained_coef
from pymove.molecules.molecule_surface import MoleculeSurface
from pymove.molecules.topological_fragments import TopologicalFragments


class PredictVolume(BaseDriver_):
    """
    Predict the solid-form volume (or equivalently the density) of a given 
    molecule given the features and coefficients of the trained model.
    
    """
    def __init__(self, 
                 features=None, 
                 coef=None,
                 tf=None,
                 ms=None,
                 spacing=0.3,
                 probe_radius=3.0):
        
        if features == None:
            features = trained_features
        if coef == None:
            coef = trained_coef
        
        self.features = np.array(features)
        self.coef = np.array(coef)
        
        if tf == None:
            tf = TopologicalFragments()
        if ms == None:
            ms = MoleculeSurface(spacing=spacing, probe_radius=probe_radius)
        
        self.tf = tf
        self.ms = ms
    
    
    def calc_struct(self, struct):
        self.struct = struct
        
        feature_vector = np.zeros(self.features.shape[0])
        vol_idx = np.where(self.features == "Molecular_Surface_Volume")[0]
        
        packing_accessible_volume = self.ms.calc_struct(struct)
        feature_vector[vol_idx] = packing_accessible_volume
        
        temp_frag,temp_counts = self.tf.calc(self.struct)
        f_idx,c_idx = np.nonzero(self.features[:,None] == temp_frag)
        c = np.array(temp_counts)
        f = np.array(temp_frag)
        feature_vector[f_idx] = c[c_idx]
        
        final_solid_volume = np.sum(np.dot(feature_vector,self.coef))
        struct.properties["predicted_volume"] = final_solid_volume
        
        return final_solid_volume
        
        
        
        
        
        
        
        
            
        
        
