'''
script that rusn FSPS again if fsps files are available and repopulates the final csv files and summary plots.

OSError: /Users/radioactive/Desktop/PROJS/paper2/models/pelvis_disk/step_zrei9_mpeak_9e6/fsps_files/fsps_Suspicious_Disk_10283.dat.dat not found.

'''

import numpy as np  
from tqdm import tqdm
import concurrent.futures
import time
import os
import logging
import glob
import pandas as pd
try:import fsps 
except: print("pyFSPS has not been installed yet.")

from run_grumpy import read_config_file, argument_parser
from run_grumpy import extract_model_params, add_mags
from plotting import run_plotting
from fsps_model import clean_fsps, run_parallel, run_single_fsps, run_fsps_python
from run_grumpy import print_stage

    
if __name__ == '__main__':

    #run the plotting again as well

    args = argument_parser().parse_args()
    # read parameters and information from the run config file 
    iniconf = read_config_file(config_file=args.ini)

    cosmo_params, model_params = extract_model_params(iniconf=iniconf)

    mah_data_dir = iniconf['data paths']['mah_data_dir']
    #find the different suites we have to run model over 
    all_suites = np.array(glob.glob(mah_data_dir+"/*/"))

    all_suite_names = []
    for si in all_suites:
        suite_name = si.replace(mah_data_dir+"/","").replace("/","")
        all_suite_names.append(suite_name)
    
    run_fsps_python(iniconf=iniconf,cosmo=cosmo_params)

    add_mags(suite_names=all_suite_names, iniconf=iniconf)
    




# def run_fsps_python(iniconf=None,cosmo_params=None):
#     '''
#     running FSPS on all tabulated SFH and metallicities

#     Parameters:
#     -----------
#         iniconf: dictionary, information in ini file
#         cosmo_params: dictionary, cosmlogical parameters
#     Returns:
#     -----------
#         None
#     '''
    
#     fsps_input_path = iniconf['output paths']['fsps_input_dir']
#     fsps_output_path = iniconf['output paths']['fsps_output_dir']
#     # temp_txt_path = iniconf['code paths']['temp_dir'] + "/temp.txt"
#     #when rerunning fsps, we will not have the temp file available
#     #so instead we will glob glob the directory of fsps files to get all the names

#     print(fsps_input_path)

#     #this is time at z = 0
#     iH0 = 9.77814e2 / cosmo_params['H0']

#     sfh_names_list=[]
#     template_name = fsps_input_path + "/fsps_*.dat"
#     sfh_names_list = glob.glob(template_name)

#     #intialize the stellar population model. 
#     #we set the Stellar Population model for tabular functionality. Thus, zcontinuous = 3 and sfh = 3 
#     #We set IMF to Chabrier type.
#     print_stage("Initializing Stellar Population.",ch="-")
#     sps = fsps.StellarPopulation(zcontinuous=3, sfh=3, imf_type=1)
#     print_stage("Finished initializing Stellar Population.",ch="-")

    
#     #use suite names to find the fsps input files only for those suites that are needed

#     # sfh_names_list = list(np.loadtxt(temp_txt_path, dtype = "str"))
#     # sfh_names_list = [os.path.basename(x) for x in glob.glob(fsps_input_path + '/*.dat')]


#     iter_input = []
    
#     for k in sfh_names_list:
#         dicti = {"sps":sps, "sfh_name":k.replace(".dat",'').replace(fsps_input_path+"/",""),"fsps_input_path":fsps_input_path,
#                  "fsps_output_path":fsps_output_path,"time":iH0,"final_ending":None } 
#         iter_input.append(dicti)

#     #run the first calculation. This usually takes 5 min. The remaining ones happen fast. 
#     run_single_fsps(iter_input[0])

#     #running fsps in parallel
#     run_parallel(run_single_fsps, iter_input[1:])

#     return