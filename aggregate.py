import xarray as xr
import os
import numpy as np
import sys

def make_average(file_list, directory):
    
    data= xr.open_dataset(os.path.join(directory, file_list[0]))
    for file_name in file_list:
        data += xr.open_dataset(os.path.join(directory, file_name))

    data.to_netcdf(os.path.join(directory, file_list[0][:-10] + '_aggregated.nc'))
    
if __name__ == '__main__':
    
    directory = sys.argv[1]
    file_list = os.listdir(directory)
    filtered = [file_name for file_name in file_list if '.nc' in file_name]
    make_average(filtered, directory)
