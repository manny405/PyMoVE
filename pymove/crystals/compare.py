


"""
File for structure checks:
    - Duplicates
    - Physical Structure
    - Molecule in structure checks

"""

import os,json
import numpy as np

from pymove import Structure,StructDict
from pymove.io import read,write

from pymatgen.analysis.structure_matcher import (StructureMatcher,
                                                ElementComparator,
                                                SpeciesComparator,
                                                FrameworkComparator)

from pymove.libmpi.base import _JobParallel


class pymatgen_compare():
    def __init__(self, pymatgen_kw=
                 {
                        "ltol": 0.2,                                    
                        "stol": 0.3,
                        "angle_tol": 5,
                        "primitive_cell": True,
                        "scale": True,
                        "attempt_supercell": False
                 }):
        self.kw = pymatgen_kw
    
    
    def __call__(self, struct1, struct2):
        sm =  StructureMatcher(
                    **self.kw,                                 
                    comparator=SpeciesComparator())                           
                                                                               
        pstruct1 = struct1.get_pymatgen_structure()                                      
        pstruct2 = struct2.get_pymatgen_structure() 
        
        return sm.fit(pstruct1, pstruct2)
    
    
    
class DuplicateCheck():
    """
    Checks if there are duplicate structures in an entire structure dictionary. 
    Duplicaate check may be ran in a serial or parallel way using MPI. This is
    automatically detected and handled by the program. 
    
    Arguments
    ---------
    struct_dict: StructDict
        Dictionary containing all structures that should be checked for 
        duplicates.
    mode: str
        Mode of operation. Can be one of pair or complete. Pair will do the 
        minimal number of comparisons (n chose 2). Complete will always compare
        each structure to every other structure in the struct_dict.
    compare_fn: callable
        Callable that performs the comparison. This function can be arbitrary,
        but it must take in two structures as an argument and return True if
        the structures are duplicates, or False if they are not. See the 
        pymatgen_compare class for an example of what this might look like.
        
    """
    def __init__(self, 
                 struct_dict={}, 
                 mode="pair", 
                 compare_fn=pymatgen_compare()):
        
        self.struct_dict = struct_dict
        self.compare_fn = compare_fn
        self._check_mode(mode)
        self.duplicates_dict = {}
        
        if len(struct_dict) != 0:
            self._comparisons()
            
        ## Instantial JobParallel if DuplicateCheck called with MPI
        self.jp = _JobParallel()
        
    
    def _check_mode(self, mode):
        self.modes = ["pair", "complete"]
        if mode not in self.modes:
            raise Exception("DuplicateCheck mode {} is not available. "
                            .format(mode) +
                            "Please use one of {}.".format(self.modess))
        else:
            self.mode = mode
    
    
    def _comparisons(self):
        """
        Get the dictionary of comparisons that have to be made for pair mode
        esxecution.
        
        """
#        keys = np.array([x for x in self.struct_dict.keys()])
#        square = np.char.add(np.char.add(keys, "_"), keys[:,None])
        
        ### Just doing this to get the correct shape and index values for a
        ## pairwise comparison
        temp = np.arange(0,len(self.struct_dict),1)
        square = temp + temp[:,None]
        idx = np.triu_indices(n=square.shape[0],
                              k=1,
                              m=square.shape[1])
        
        #### Build the dictionary of indicies that each structure must be 
        ## compared to for pairwise comparison.
        keys = [x for x in self.struct_dict.keys()]
        comparison_dict = {}
        for key in keys:
            comparison_dict[key] = []
            
        for idx_pair in zip(idx[0],idx[1]):
            key = keys[idx_pair[0]]
            comparison_dict[key].append(idx_pair[1])
        
        self.comparisons = comparison_dict
        
        for key in self.struct_dict.keys():
            self.duplicates_dict[key] = []
        
    
    def calc(self, struct_obj):
        """
        General calc wrapper
        
        Arguments
        ---------
        struct_obj: str,dict,Structure
            Can be a string, a dictionary, or a Structure. 
                str: Assume this is a directory path. 
                dict: Structure dictionary
                Structure: Currently not supported
                
        """
        if type(struct_obj) == str:
            if not os.path.isdir(struct_obj):
                raise Exception("{} is not a directory".format(struct_obj))
            if self.jp.size == 1:
                self.struct_dict = read(struct_obj)
                self._comparisons()
                self.calc_dict(self.struct_dict)
            else:
                ### Just read onto rank 0 for efficiency. Jobs will be 
                ### communicated later. 
                if self.jp.rank == 0:
                    self.struct_dict = read(struct_obj)
                    self._comparisons()
                    ## Don't need any arguments for parallel implementation
                    self.calc_dict(None)
                else:
                    ## Don't need any arguments for parallel implementation
                    self.calc_dict(None)
        if type(struct_obj) == StructDict or \
           type(struct_obj) == dict:
            self.calc_dict(struct_obj)
        elif type(struct_obj) == Structure:
            self.calc_struct(struct_obj)
            
    
    def calc_dict(self, struct_dict):
        
        if self.jp.size == 1: 
            ## Serial Mode
            if len(self.struct_dict) == 0:
                self.struct_dict = struct_dict
                self._comparisons()
            for struct_id,struct in struct_dict.items():
                self.calc_struct(struct)
        else:
            ## Parallel Mode Calculation
            self.get_job_list()
            self.jp.calc()
            
            
    def calc_struct(self, struct):
        eval("self.calc_struct_{}(struct)".format(self.mode))
        
        
    def calc_struct_pair(self, struct):
        """
        Pair mode implementation. 
        
        """
        keys = [x for x in self.struct_dict.keys()]
        if struct.struct_id not in keys:
            raise Exception("Structure ID {} was not found "
                    .format(struct.struct_id)+
                    "in the DuplicateCheck.struct_dict.")
        
        struct_dup_pool = [struct.struct_id]
        for idx in self.comparisons[struct.struct_id]:
            struct2 = self.struct_dict[keys[idx]]
            
            if self.compare(struct, struct2):
                struct_dup_pool.append(struct2.struct_id)
                print(struct.struct_id, struct2.struct_id)
            
        
        ## Now update the duplicates dict of all found duplicates with the 
        # same values
        for struct_id in struct_dup_pool:
            self.duplicates_dict[struct_id] += struct_dup_pool
            # Only use unique values
            self.duplicates_dict[struct_id] = \
                np.unique(self.duplicates_dict[struct_id]).tolist()
    
    
    def calc_struct_complete(self, struct):
        """
        Compare structure to all other structures in the structure dictionary.
        
        """
        raise Exception("Complete mode is not implemented yet.")
        
    
    def compare(self, struct1, struct2):      
        """
        Compare structures using pymatgen's StructureMatcher
        
        """                                                           
        return self.compare_fn(struct1, struct2)
    
    
    def write(self, file_name="duplicates.json"):
        """
        Output a format for the duplicates. Can outputs all the duplciates
        for each structure and a section for only the unique list of 
        duplicates found in the duplicates_dict.
        
        """
        ## Parallel Mode
        if self.jp.size > 1: 
            self.parallel_write(file_name)
            return
        
        ### First find unique duplicates in the duplicates_dict
        id_used = []
        unique = []
        for name,duplicates in self.duplicates_dict.items():
            if len(duplicates) == 1:
                continue
            elif name in id_used:
                continue
            else:
                unique.append(duplicates)
                [id_used.append(x) for x in duplicates]
        
        output_dict = {}
        output_dict["struct"] = self.duplicates_dict
        output_dict["dups"] = unique
        
        with open(file_name,"w") as f:
            f.write(json.dumps(output_dict, indent=4))
            
    
    def get_job_list(self):
        """
        Obtains job list for JobParallel calculation. 
        
        """
        
        ## Use rank 0 to communicate jobs
        if self.jp.rank == 0:
            keys = [x for x in self.struct_dict.keys()]
            job_list = []
            for struct1_id,value in self.comparisons.items():
                for idx in value:
                    struct2_id = keys[idx]
                    job_list.append([self.compare_fn, 
                        {"struct1": self.struct_dict[struct1_id],
                         "struct2": self.struct_dict[struct2_id]}])
            self.jp.job_list = job_list
        
        ## Send jobs from rank 0
        self.jp.send_jobs()
        
        return 
    
    
    def parallel_write(self, file_name="duplicates.json"):
        if self.jp.rank != 0:
            return
        
        all_results = self.jp.all_results
        
        if all_results == None:
            raise Exception("Cannot call write without calculating.")
        
        for entry in all_results:
            settings = entry[0]
            result = entry[1]
            
            if result == True:
                arguments = settings[1]
                struct1_id = arguments["struct1"].struct_id
                struct2_id = arguments["struct2"].struct_id
                
                ## Update both in duplicates dict
                self.duplicates_dict[struct1_id].append(struct2_id)
                self.duplicates_dict[struct2_id].append(struct1_id)
            else:
                pass
        
        for name,duplicates in self.duplicates_dict.items():
            self.duplicates_dict[name] = [name] + self.duplicates_dict[name]
        
        ### Then same algorithm as serial write
        id_used = []
        unique = []
        for name,duplicates in self.duplicates_dict.items():
            if len(duplicates) == 1:
                continue
            elif name in id_used:
                continue
            else:
                unique.append(duplicates)
                [id_used.append(x) for x in duplicates]
    
        output_dict = {}
        output_dict["struct"] = self.duplicates_dict
        output_dict["dups"] = unique
        
        with open(file_name,"w") as f:
            f.write(json.dumps(output_dict, indent=4))  
            

if __name__ == "__main__":     
    from pymove.io import read,write
    test_dir = "/Users/ibier/Research/Results/Hab_Project/genarris-runs/GIYHUR/20191103_Full_Relaxation/GIYHUR_Relaxed_spg"
    
    s = read(test_dir)
    keys = [x for x in s.keys()]
    keys = keys[0:5]
    test_s = {}
    for entry in keys:
        test_s[entry] = s[entry]
        
    test_s = read("/Users/ibier/Research/Results/Hab_Project/GAtor-runs/BZOXZT/20200726_Multi_GAtor_Report_Test/Dup_Check_Test/test")
        
    dc = DuplicateCheck(test_s)
    
    ### Parallel Testing
    dc.jp.job_list = dc.get_job_list()
    
#    dc.jp.calc()
    
    