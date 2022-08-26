'''
In this script, we construct summary catalogs (just like what the main pipeline outputs) but for any given previous epoch.
This is used for the MW forward modelling project as we needed to find LMCs that were on their first infall and at the right distance (~ 50kpc)

This reads the same ini file of the main pipeline 

The basic steps are:
1. Using the textract value, read all the corresponding rows in the tract data txt files for that halo (which have been made when running the main pipeline)

2. Find the value of time (in Gyrs) corresponding to that textract value and rerun fsps for that single value. This should be fairly quick. 

3. Combine all the FSPS outputs (at that textract value) and step 1 data and construct a new summary csv file for that epoch


-----------------------------------------------------------------------------------------------
Some notes to myself about understanding disruption modelling and related stuff in this context.
While looking at a previous epoch, I am considering the state of the host + satellites at that specific time. I am doing the same analysis as I was doing for z = 0. 
However, when modelling disruption, I will be using info that was calculated at z = 0. It is possible that one object was inside the host at a previous epoch but outside at z = 0.
This will have a disruption probabiltiy of 0. This is OK.
'''

##I SHOULD CONFIRM THE DIFFERENCE IN NUMBER OF HALOS WHEN LOOKING AT SAY Z=0 AND Z=0.25(like the largest look back redshift I will be looking at)
##If there is substantial difference, I will have re-run the merger tree part
##check that merger tree extraction is all good.
##Find the LMC indicies so I can extract their track data. 


import os
import argparse
from configparser import ConfigParser, ExtendedInterpolation
import pandas as pd
import numpy as np
import fsps 
import glob
import fsps
from tqdm import tqdm
from fsps_model import run_fsps_python, run_single_fsps, run_parallel
from run_grumpy import extract_model_params,read_config_file, print_stage

#these bands are for FSPS
all_bands = ['u','b','v','cousins_r','cousins_i', 'sdss_u','sdss_g','sdss_r','sdss_i','sdss_z']


def argument_parser():
    '''
    Function that parses the arguments passed while running a script
    '''
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # path to the config file with parameters and information about the run
    result.add_argument('-ini', dest='ini', type=str) 
    return result


def extract_time_rows(iniconf=None, suite_name = None, textract = None):
    '''
    Function that weaves through the track txt file and extracts rows corresponding to that time t. 

    suite_name is a string that appears in the name of the relevant host on which we wish to do this look-back analysis.
    e.g. suite name is caterpillar_4_LX14

    textract is the index at which we extract all the information (a previous epoch)

    This is for a single host.
    '''
    #get all the file names of track data 

    track_dir = iniconf['output paths']['final_output_dir']

    track_name_template = track_dir + "/track_data/" + suite_name + "*_track.csv"

    all_track_files = glob.glob(track_name_template)

    if len(all_track_files) == 0:
        raise ValueError("No track files found! Check that entered suite_name of track_name_template variable in this function is correct.")
    #we will be cross matching them with the original catalogs for info on disk disruption
    #the name of the track file can be used to cross match with the original catalog

    summary_epoch = {"file_name":[], "xpos": [], "ypos" : [], "zpos": [], "Mvir": [], "M200c": [], "Mh": [], "Ms": [], "hdist": [], "rsize":[], "rvir":[],"rs": [], "upID":[], "ID":[]}
    #Mvir, rvir and rs is needed to compute mass within 100kpc using NFW profile

    #these are columns from the track data dataframe that we want to read
    rel_columns = ["xpos","ypos","zpos","M200c","Mh","Ms","hdist","rsize","Mvir", "rvir","rs","upID","ID"]

    #let us read the original cmain summary catalog file
    summary_z0_path = iniconf['output paths']['final_output_dir'] + "/" + suite_name + "_" + iniconf['output paths']['run_name'] + "_run.csv"
    summary_z0 = pd.read_csv(summary_z0_path)

    #all the file names from summary_z0
    mah_names = np.array(summary_z0["file_name"])

    if len(mah_names) != len(all_track_files):
        raise ValueError('The number of objects in z=0 summary file to not match with number of track files of that host.')

    all_t_epochs = []
    all_zred_epochs = []
    #now we loop over each of these columns and extract the relevant row
    for mhi in tqdm(mah_names):
        #for the file name fi, we need to find the corresponding track file
        #this method make it possible for us to match the two catalogs correctly

        track_file_i = track_dir + "/track_data/" + mhi + "_track.csv"
        dfi = pd.read_csv(track_file_i)

        #we extract the actual time at which we the epoch extraction is happening
        all_times = np.array(dfi["t"])
        all_zreds = np.array(dfi["z_redshift"])

        all_t_epochs.append(all_times[textract])
        all_zred_epochs.append(all_zreds[textract])

        #Some particles might not exist this long.
        #if they do not exist...we consider everything to be -99.

        summary_epoch["file_name"].append(mhi)

        for reli in rel_columns:
            if reli == "hdist":
                #we have to do this separate because hdist info does not exist in the track data folder
                rel_x = np.array(dfi["xpos"])[textract]
                rel_y = np.array(dfi["ypos"])[textract]
                rel_z = np.array(dfi["zpos"])[textract] 
                hdist_i = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)*1e3
                summary_epoch["hdist"].append(hdist_i)
            else:
                rel_info = np.array(dfi[reli])
                summary_epoch[reli].append(rel_info[textract])

    if np.min(all_t_epochs) != np.max(all_t_epochs):
        raise ValueError("the textract is resulting in different times of extraction.")

    #check if the lenghts match up

    if len(np.array(summary_z0['hdist'])) != len(summary_epoch["hdist"]):
        raise ValueError('The number of objects in z=0 summary file to not match with previous epoch computation.')


    #make summary_epoch into df 
    summary_epoch_df = pd.DataFrame(summary_epoch)

    summary_epoch_df["surv_probs"] = np.array(summary_z0["surv_probs"])
    summary_epoch_df["d_peri"] = np.array(summary_z0["d_peri"])
    summary_epoch_df["a_peri"] = np.array(summary_z0["a_peri"])
    summary_epoch_df["Mpeak"] = np.array(summary_z0["Mpeak"])

    

    final_extraction_time = all_t_epochs[0] #any element in this list (in Gyr)
    final_extraction_zred = all_zred_epochs[0] #any element in this list
    final_extract_scale = 1/(1+final_extraction_zred)

    print_stage("The extraction time of " + suite_name + " is %.3f Gyrs or scale factor = %.3f"%(final_extraction_time,final_extract_scale))

    #save this new df as a csv file
    summary_epoch_df_path = iniconf['output paths']['final_output_dir'] + "/" + suite_name + "_" + iniconf['output paths']['run_name'] + "_run_pe_t_%.2f.csv"%final_extraction_time
    summary_epoch_df.to_csv(summary_epoch_df_path,index=False)

    print_stage("The previous-epoch summary csv file is being written to %s"%summary_epoch_df_path)

    return final_extraction_time, summary_epoch_df_path



def add_mags(suite_names=None,summary_epoch_df_paths=None,iniconf=None):
    '''
    Function that reads FSPS magnitude output and writes them to the summary data file for each suite. 

    Parameters:
    -----------
    suite_names: list, simulation suite names
    iniconf: dict, dictionary of parameters in the config file
    '''

    for i,si in enumerate(suite_names):
        if si not in summary_epoch_df_paths[i]:
            raise ValueError("potential issue of not mismatching host names in add_mags function.")

        final_csv_path = summary_epoch_df_paths[i]
        df = pd.read_csv(final_csv_path)
        mU,mB,mV,mR,mI,mu,mg,mr,mi,mz = [], [], [], [], [], [], [], [], [], []
        file_names = np.array(df["file_name"])
        #filter for file_names that contain the suite name
        file_names_f =[]
        for fi in file_names:
            if si+"_" in fi:
                file_names_f.append(fi)
            
        if len(file_names_f) != len(df["Mvir"]):
            raise ValueError("The number of filtered fsps files found does not match the number of objects in the existing csv catalog.")

        for fi in file_names_f: 
            fsps_output_path = iniconf['output paths']['fsps_output_dir']
            fname = fsps_output_path + "/" + fi + "_output_pet.txt" 
            #note the extra pet above.
            mags = np.loadtxt(fname)
            mU.append(mags[0])
            mB.append(mags[1])
            mV.append(mags[2])
            mR.append(mags[3])
            mI.append(mags[4])
            mu.append(mags[5])
            mg.append(mags[6])
            mr.append(mags[7])
            mi.append(mags[8])
            mz.append(mags[9])

        #append the below columns to the dataframe
        df['mU'],df["mB"],df["mV"],df["mR"],df["mI"],df["mu"],df["mg"],df["mr"],df["mi"],df["mz"]  = mU,mB,mV,mR, mI, mu, mg, mr, mi, mz

        #write the updated dataframe to same path 
        df.to_csv(final_csv_path, index=False)

    print_stage("Final summary files updated with FSPS magnitudes.", ch='-')

    return 

 

if __name__ == '__main__':

    # read in command line arguments
    args = argument_parser().parse_args()
    # read parameters and information from the run config file 
    iniconf = read_config_file(config_file=args.ini)
    cosmo_params, model_params = extract_model_params(iniconf=iniconf)


    #extract the suite names for extracting different epoch summary file

    _names = iniconf['prev epoch']['suites'].split(',')
    suite_names = []

    for name in _names: 
        name = name.strip()
        suite_names.append(name)

    fsps_input_path = iniconf['output paths']['fsps_input_dir']
    fsps_output_path = iniconf['output paths']['fsps_output_dir']

    all_textracts = []
    textracts = iniconf['prev epoch']['textract'].split(',')
    for tei in textracts: 
        tei = tei.strip()
        all_textracts.append(int(tei))

    if len(all_textracts) != len(suite_names):
        raise ValueError('Number of textracts provided does not agree with number of suite names provided')


    #construct summary csv for previous epoch using extract_time_rows function
    all_si_times = []
    all_si_paths = []

    for k,si in enumerate(suite_names):
        si_time,si_epoch_path  = extract_time_rows(iniconf=iniconf, suite_name=si, textract = all_textracts[k])
        all_si_paths.append(si_epoch_path)
        all_si_times.append(si_time)

    #intialize the stellar population model. s
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


    #the above will have all the fsps names, we will only need those relevant to our suite names
    iter_input = []

    for j,si in enumerate(suite_names):

        template_name = fsps_input_path + "/fsps_" + si + "_*.dat"
        all_files = glob.glob(template_name)
        si_time = all_si_times[j]
        #loop over all these files 
        for i,sfhi_path in tqdm(enumerate(all_files)):
            #we strip of the path of sfhi_path
            sfhi = sfhi_path.replace(fsps_input_path+"/","").replace(".dat","")
            if i < 3:
                print(sfhi)

            dicti = {"sps":sps, "sfh_name":sfhi,"fsps_input_path":fsps_input_path,
                    "fsps_output_path":fsps_output_path,"time":si_time, "final_ending" : "pet" } 

            iter_input.append(dicti)
   


    #run the first calculation. This usually takes 5 min. The remaining ones happen fast. 
    run_single_fsps(iter_input[0])

    #running fsps in parallel
    run_parallel(run_single_fsps, iter_input[1:])

    #add fsps outputs to csv files
    add_mags(iniconf=iniconf,summary_epoch_df_paths=all_si_paths,suite_names=suite_names)
    print_stage("All the summary files for has been updated with FSPS magnitudes.")

