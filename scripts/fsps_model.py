import numpy as np  
from tqdm import tqdm
import pandas as pd
import concurrent.futures
import time
import os
import glob
import logging
from age_integrate import age
from scipy.interpolate import UnivariateSpline
from pathlib import Path
try:
    import fsps 
except:
    print("pyFSPS has not been installed yet.")

#all_bands = ['u','b','v','cousins_r','cousins_i', 'sdss_u','sdss_g','sdss_r','sdss_i','sdss_z']

def clean_fsps(iniconf=None):
    '''
    Function to clean the fsps input and output file directories. 
    '''
    temp_dir = iniconf['output paths']['fsps_input_dir'][:-5]
    os.chdir(temp_dir)
    os.system('> temp.txt')

    return 


def run_parallel(func, iter_input):
    '''
    running a function in parallel.

    Parameters:
    -----------
        func: function, the function to be parallelized
        iter_input: list, the list of function inputs to loop over

    Returns:
    -----------
        results: list, entire function output

    '''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, iter_input),
                            total = len(iter_input))) 
    return results

def print_stage(line2print, ch='-'):
    nl = len(line2print)
    print(ch*nl)
    print(line2print)
    print(ch*nl)
    print(' ')

def run_single_fsps(input_stuff=None):
    '''
    running FSPS on a single tabulated SFH.

    Parameters:
    -----------
        input_stuff: dictionary, information like SPS object, data paths and time of extraction.

    Returns:
    -----------
        None   
    '''
    sps = input_stuff["sps"]
    sfh_name = input_stuff["sfh_name"]
    fsps_input_path = input_stuff["fsps_input_path"]
    fsps_output_path = input_stuff["fsps_output_path"]
    time = input_stuff["time"]
    final_ending = input_stuff["final_ending"]
    fsps_filters = input_stuff["fsps_filters"]
    
    table_path = fsps_input_path + "/" + sfh_name + ".dat"
    #check if this path exists. If this path does not exist, print all the relevant quantities.
    my_path = Path(table_path)
    if my_path.is_file():
        pass
    else:
        print(fsps_input_path)
        print(sfh_name)
        raise ValueError("fsps_model.py: The table path %s does not exist!"%table_path)

    data = np.loadtxt(table_path)
    age = data[:,0]
    sfr = data[:,1]
    Z = data[:,2]
    if np.max(sfr) == 0:
        sfr[np.argmax(sfr)] = 1e-32
        #the fsps python requires atleast one value with sfr > 1e-33
    sps.set_tabular_sfh(age, sfr, Z)

    mags = sps.get_mags(tage=time, bands=fsps_filters)
        
    if final_ending is None:
        output_file = fsps_output_path + "/" + sfh_name.replace(".dat","").replace("fsps_","") + "_output.txt"
    else:
        output_file = fsps_output_path + "/" + sfh_name.replace(".dat","").replace("fsps_","") + "_output_" + final_ending + ".txt"

    np.savetxt(output_file,mags)

    return 

def run_fsps_python(iniconf=None, cosmo=None, use_time=None):
    '''
    running FSPS on all tabulated SFH and metallicities

    Parameters:
    -----------
        iniconf: dictionary, information in ini file
        cosmo_params: dictionary, cosmlogical parameters
    Returns:
    -----------
        None
    '''
    
    fsps_input_path = iniconf['output paths']['fsps_input_dir']
    fsps_output_path = iniconf['output paths']['fsps_output_dir']
    temp_txt_path = iniconf['output paths']['fsps_input_dir'][:-5] + "/temp.txt"

    #intialize the stellar population model. 
    #we set the Stellar Population model for tabular functionality. Thus, zcontinuous = 3 and sfh = 3 
    #We set IMF to Chabrier type.
    imf_kind = iniconf['fsps params']['imf_type']


    if imf_kind == "Salpeter55":
        imf_num = 0
    if imf_kind == "Chabrier03":
        imf_num = 1
    if imf_kind == "Kroupa01":
        imf_num = 2
    if imf_kind == "Dokkum08":
        imf_num = 3
    if imf_kind == "Dave08":
        imf_num = 4


    print_stage("Initializing Stellar Population using '%s' IMF."%imf_kind,ch="-")

    sps = fsps.StellarPopulation(zcontinuous=3, sfh=3, imf_type=imf_num)

    print_stage("Finished initializing Stellar Population.",ch="-")

    sfh_names_temp = list(np.loadtxt(temp_txt_path, dtype = "str"))

    sfh_names_list=[]
    template_name = fsps_input_path + "/fsps_*.dat"
    sfh_names_list = glob.glob(template_name)

    if len(sfh_names_list) != len(sfh_names_temp):
        raise ValueError("fsps_model.py: the two lists sfh_names_list and sfh_names_temp should have the same length.")

    iter_input = []

    if use_time is None:
        t_fsps = 9.77814e2 / cosmo.H0 # 1/H0 in Gyrs
    else:
        t_fsps = use_time
        
    fsps_filters =  iniconf['fsps params']['fsps_filters']
    fsps_filters = list(fsps_filters.split(","))
   
    for sfhi in tqdm(sfh_names_list):
        #as we read the file names from glob.glob, we have to remove all the additional text aroudn file name

        dicti = {"sps": sps, "sfh_name": sfhi.replace(".dat",'').replace(fsps_input_path+"/",""),
                 "fsps_input_path": fsps_input_path, "fsps_output_path": fsps_output_path,
                 "time": t_fsps, "final_ending": None, "fsps_filters": fsps_filters} 

        iter_input.append(dicti)

    #run the first calculation. This usually takes 5 min. The remaining ones happen fast.
    print('will produce magnitudes in the following filters:', fsps_filters)
    run_single_fsps(iter_input[0])

    #running fsps in parallel
    run_parallel(run_single_fsps, iter_input[1:])

    return

