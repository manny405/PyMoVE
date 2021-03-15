# -*- coding: utf-8 -*-


from pymove import StructDict
from pymove.io import read,write

from mpi4py import MPI


class BaseDriver_():
    """
    Base pymove Driver class that defines the API for all pymove Drivers. 
    Any Driver should inherit this classes API.
    
    Arguments
    ---------
    comm: MPI.COMM
        MPI communicator. In general, this is optional and need not be 
        provided to any Driver. Although, some Drivers are MPI aware because
        their workflows are not necessarily naively parallel. 
    
    """
    def __init__(self, comm=None, **settings_kw):
        ## initialize settings
        self.comm = comm
        self.init_mpi()
        pass
    

    def init_mpi(self):
        if self.comm == None:
            self.comm = MPI.COMM_WORLD
        
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
    

    def calc(self, struct_obj):
        """
        Wrapper function to enable operation for both a single Structure 
        object or an entire Structure Dictionary.
        
        Arguments
        ---------
        struct_obj: pymove.Structure or Dictionary
            Arbitrary structure object. Either dictionyar or structure.
            
        """
        if type(struct_obj) == dict or type(struct_obj) == StructDict:
            return self.calc_dict(struct_obj)
        else:
            return self.calc_struct(struct_obj)
            
    
    def calc_dict(self, struct_dict):
        """
        Calculates entire Structure dictionary.
        
        """
        for struct_id,struct in struct_dict.items():
            self.calc_struct(struct_obj)
            
    
    def calc_struct(self, struct):
        """
        Perform driver calculation on the input Structure. 
        
        """
        driver_info = ["Drivers may modify Structure objects. " +
                       "Because Structures are user defined objects, " +
                       "they are referenced by memory in Python. " + 
                       "This means any modification made to the Structure "+
                       "will be seen outside the function as well."]
        
        struct.properties["DriverInfo"] = driver_info
        
        write_info = ["Although, we may want to save the structure as a part "+
                      "of the class so it can be written as the output by the "+
                      "Driver.write method."]
        struct.properties["WriteInfo"] = write_info
        
        self.struct = struct
            
    
    def write(self, output_dir, file_format="json", overwrite=False):
        """
        Writes the Driver's output to the to the output_dir argument. The only 
        specification of the Driver.write function for the pymove API are the 
        arguments specified above. The output of a Driver is not necessarily a 
        Structure, although this is the most common situation so the write 
        arguments are tailored for this purpose. In principle, the output of an 
        pymove Driver is not specified by the API and could be anything.  
        
        Usually, a Driver will only need to output a Structure file, with the 
        struct_id as the name of the output file, to the output directory. 
        This is what's done here.
        
        """
        ## Define dictionary for write API. 
        temp_dict = {self.struct.struct_id: self.struct}
        write(output_dir, 
              temp_dict, 
              file_format=file_format, 
              overwrite=overwrite)
        
    
    def restart(self, output_dir):
        """
        Identify the progress of the Driver, find the proper way to restart, 
        and begin calculation. 
        
        """
        raise Exception("Not Implemented")
    
    
    def check(self, output_dir):
        """
        Check if the calculation has been performed correctly. 
        
        """
        raise Exception("Not Implemented")
        
    
    def report(self, output_dir, report=None, ):
        """
        Contribute or create a report for the Driver calculation. 
        
        """
        raise Exception("Not Implemented")
        
