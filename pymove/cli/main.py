
import sys,os
from pymove.io import read,write
from pymove.models.predict import PredictVolume


def main():
    file_path = sys.argv[-1]
    if not os.path.exists(file_path):
        raise Exception("Input file {} was not found".format(file_path))
    
    struct = read(file_path)
    pv = PredictVolume()
    pred_sf_volume = pv.calc_struct(struct)
    
    print("Predicted Solid-Form Volume of {}: {}"
          .format(struct.struct_id, pred_sf_volume))
    


