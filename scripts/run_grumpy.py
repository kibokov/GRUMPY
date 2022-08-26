#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A pipeline to run galaxy formation model calculations and analyses using halo
tracks extracted from numerical simulations
"""

import os
import glob
import argparse
from configparser import ConfigParser, ExtendedInterpolation
from mah_fitting import prepare_mh_tracks
from galaxy_model import integrate_model, get_beta
from run_disruption import run_disruption_model, filter_disruption
from plotting import run_plotting
from galaxy_model import post_process
import pandas as pd
from collections import namedtuple
import numpy as np
import logging
from tqdm import tqdm
import concurrent.futures
from fsps_model import run_fsps_python, clean_fsps

def argument_parser():
    '''
    Function that parses the arguments passed while running a script
    '''
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # path to the config file with parameters and information about the run
    result.add_argument('-ini', dest='ini', type=str) 
    return result

def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

def print_stage(line2print, ch='-',end_space=True):
    '''
    Function that prints lines for organizational purposes in the code outputs.

    Parameters:
    -----------
    line2print: str, the message to be printed
    ch : str, the boundary dividing character of the message
    '''
    nl = len(line2print)
    print(ch*nl)
    print(line2print)
    print(ch*nl)
    if end_space == True:
        print(' ')


def check_path_existence(all_paths=None):
    '''
    Creates directories if they do not exist

    Parameters:
    --------------
    all_paths: list, directory list to loop over
    '''
    for pi in all_paths:
        if not os.path.exists(pi):
            print_stage('The path {:s} did not exist. It has now been created.'.format(pi),ch="-")
            os.makedirs(pi)
    return



def read_config_file(config_file=None):
    '''
    Function that reads the ini file

    Parameters:
    -------------
    config_file: str, path to the config file

    Returns:
    ------------
    iniconf: dict, dictionary of parameters in the config file

    '''
    # check whether the config file exists and the final ending is indeed .csv
    if config_file is None:
        raise ValueError('input configuration file is not provided. Use run_series.py -ini config_file_path to specify the config file')
    if not os.path.isfile(config_file):
        raise ValueError('input configuration file {:s} does not exist!'.format(config_file))
    iniconf = ConfigParser(interpolation=ExtendedInterpolation())
    iniconf.read(config_file)
    return iniconf 


 
def extract_model_params(iniconf=None,print_params = True):
    '''
    Function that parses the config file dictionary to read the model parameters

    Parameters:
    -------------
    iniconf: dict, dictionary of parameters in the config file

    Returns:
    -------------
    cosmo_params: dict, dictionary for cosmological parameters
    model_params: dict, dictionary for model parameters
    '''

    bool_strings = ["False","True"]


    all_bool_strings = [iniconf['run params']['run_fsps'],iniconf['run params']['run_suite_parallel'],
                        iniconf['run params']['uniform_grid'],
                        iniconf['run params']['disable_tqdm'],
                        iniconf['cosmology']['flat']   ]

    all_flags = ["run params:run_fsps","run params:run_suite_parallel",
                "run params:disable_tqdm","cosmology:flat"]

    for i,bsi in enumerate(all_bool_strings):
        if bsi not in bool_strings:
            raise ValueError('%s is not a valid value for "%s" parameter.'%(bsi,all_flags[i]))

    #check if all data paths exist or not
    all_data_paths = [ iniconf['main paths']['mah_data_dir'],iniconf['output paths']['pickle_mah_dir'] ]
    all_flag_paths = ["data paths:mah_data_dir","data paths:pickle_mah_dir"]

    for i,dpi in enumerate(all_data_paths):
        if not os.path.exists(dpi):
            raise ValueError('The path "%s" does not exist. Make sure correct path for %s has been entered'%(dpi,all_flag_paths[i])) 

    #the allowed h2 models. 
    h2_model_vals = ["gd14","gd14_new","gd14_improved","kmt12"]

    if iniconf['model params']['h2_model'] not in h2_model_vals:
        raise ValueError('The H2 model "{:s}" is not a valid modelname. Allowed names : [gd14, gd14_new, gd14_improved, kmt12] !'.format(iniconf['model params']['h2_model']))
    
    #constraint allowed value of zrei:
    #i do not think zrei > 10.8 is allowed..need to check this.
    if float(iniconf['model params']['zrei']) > 10.8:
        raise ValueError('Value of zrei should be <10.8')

    imf_types = ['Salpeter55', 'Chabrier03', 'Kroupa01', 'Dokkum08', 'Dave08']

    if iniconf['fsps params']['imf_type'] not in imf_types:
        raise ValueError("%s is an incorrect IMF type. Allowed options are : ['Salpeter55', 'Chabrier03', 'Kroupa01', 'Dokkum08', 'Dave08']"%(iniconf['fsps params']['imf_type']))    

    if iniconf['model params']['rei_model'] not in ['step','okamoto','noreion']:
        raise ValueError("%s is an incorrect reionization model. Allowed options are : ['step','okamoto','noreion'] "%(iniconf['model params']['rei_model']))


    Om0 = float(iniconf['cosmology']['Om0'])

    if iniconf['cosmology']['flat'] == "True":
        OmL = 1.0 - Om0
    else:
        OmL = float(iniconf['cosmology']['OmL'])
        
    #some cosmology calculations
    iH0 = 9.77814e2 / float(iniconf['cosmology']['H0']) # 1/H0 in Gyrs
    H0iGyr = 1. / iH0
    h = 1.e-2 * float(iniconf['cosmology']['H0'])
    h23i = 1./h**(2./3) / 1.687
    fbuni = float(iniconf['cosmology']['Ob0']) / Om0

    cosmo_string = "H0 H0iGyr h23i Ob0 Om0 OmL fbuni"
    cosmo = namedtuple("cosmo",cosmo_string)

    cosmo = cosmo(H0 = float(iniconf['cosmology']['H0']),
                  H0iGyr = H0iGyr,
                  h23i = h23i,
                  Ob0 = float(iniconf['cosmology']['Ob0']), 
                  Om0 = Om0,
                  OmL = OmL,
                  fbuni = fbuni)
    
    param_string = "tausf eta_power eta_norm eta_mass eta_c Rloss1 Zsun Z_IGM SigHIth epsin_pr zwind zrei rei_model gamma beta h2_model rpert_sig yZ mstar_ini"

    model_params = namedtuple("model_params",param_string)

    #compute beta parameter in reionization model
    beta = get_beta(float(iniconf['model params']['zrei']), float(iniconf['model params']['gamma']))

    #assign parameter values to this named tuple
    model_params = model_params(tausf = float(iniconf['model params']['tausf']),
                                eta_power = float(iniconf['model params']['eta_power']),
                                eta_norm = float(iniconf['model params']['eta_norm']),
                                eta_mass = float(iniconf['model params']['eta_mass']),
                                eta_c = float(iniconf['model params']['eta_c']),
                                Rloss1 = float(iniconf['model params']['Rloss1']),
                                Zsun = float(iniconf['model params']['Zsun']), 
                                Z_IGM = float(iniconf['model params']['Z_IGM'])*float(iniconf['model params']['Zsun']), 
                                SigHIth = float(iniconf['model params']['SigHIth']), 
                                epsin_pr = float(iniconf['model params']['epsin_pr']),
                                zwind = float(iniconf['model params']['zwind']),
                                zrei = float(iniconf['model params']['zrei']), 
                                rei_model = iniconf['model params']['rei_model'],
                                gamma = float(iniconf['model params']['gamma']),
                                beta = beta,
                                h2_model = iniconf['model params']['h2_model'],
                                rpert_sig = float(iniconf['model params']['rpert_sig']),
                                yZ = float(iniconf['model params']['yZ']),
                                mstar_ini = float(iniconf['model params']['mstar_ini']))

    ###print the parameter value summary!
    if print_params == True:
        print(' ')
        print_stage("Model Parameter Summary",ch = "-",end_space=False)
        for i,mpi in enumerate(model_params._fields):
            if mpi == "eta_mass":
                print("%s = %.2e"%(mpi,model_params[i]))
            elif mpi == "Z_IGM":
                print("Z_IGM = %.2e Zsun"%(model_params[i]/model_params.Zsun))
            elif mpi == "beta":
                print("beta = %.3f"%(model_params[i]))
            else:
                print("%s = %s"%(mpi,model_params[i]))
        print(' ')
        print_stage("Cosmology Parameter Summary",ch="-",end_space=False)
        for i,ci in enumerate(cosmo._fields):
            print("%s = %.3f"%(ci,cosmo[i]))
        print(' ')
    else:
        pass

    return cosmo, model_params



def store_results(summary_stats=None,iniconf=None,disruption_data=None,suite_name=None):
    '''

    this function will be used only when store summary is True

    Function that writes a summary csv file using the model results 

    Parameters:
    ------------
    run_type = str, type of simulation run (eg. elvis_iso, elvis_hires, pelvis_dmo etc.)
    elvis_name = list, simulation suite names
    summary_stats = dict, dictionary of model results
    disruption_dict = dict, dictionary of disruption modelling parameters
    iniconf = dict, dictionary of parameters in the config file

    Returns:
    -----------
    None

    '''

    final_csv_path = iniconf['output paths']['final_output_dir'] + "/" + suite_name + "_" + iniconf['run params']['run_name'] + "_run.csv"

    if disruption_data is not None:
        #add the disruption data to the summary stats
        summary_stats["surv_probs"] =  disruption_data["surv_probs"]
        summary_stats["m_acc"] = disruption_data["m_acc"]
        summary_stats["v_acc"] = disruption_data["v_acc"]
        summary_stats["a_acc"] = disruption_data["a_acc"]
        summary_stats["a_peri"] = disruption_data["a_peri"]
        summary_stats["d_peri"] = disruption_data["d_peri"]

    df = pd.DataFrame(summary_stats)

    df.to_csv(final_csv_path, index=False)

    print_stage("The final summary file is stored here: " + final_csv_path, ch='-')

    return 

def add_mags(suite_names=None,iniconf=None):
    '''
    Function that reads FSPS magnitude output and writes them to the summary data file for each suite. 

    Parameters:
    -----------
    suite_names: list, simulation suite names
    iniconf: dict, dictionary of parameters in the config file
    '''

    for si in suite_names:

        final_csv_path = iniconf['output paths']['final_output_dir'] + "/" + si + "_" + iniconf['run params']['run_name'] + "_run.csv"
        df = pd.read_csv(final_csv_path)
        mU,mB,mV,mR,mI,mu,mg,mr,mi,mz = [], [], [], [], [], [], [], [], [], []
        file_names = np.array(df["file_name"])
        #filter for file_names that contain the suite name
        file_names_f =[]
        ##NEED TO FIX THIS WHEN I WILL BE RUNNING DIFFERENT SUITES AT THE SAME TIME 
        for fi in file_names:
            # if si+"_" in fi:
            file_names_f.append(fi)
            
        if len(file_names_f) != len(df["Mvir"]):
            raise ValueError("The number of filtered fsps files found does not match the number of objects in the existing csv catalog.")

        for fi in file_names_f: 
            fsps_output_path = iniconf['output paths']['fsps_output_dir']
            fname = fsps_output_path + "/" + fi + "_output.txt" 
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


def save_code(code_save_dir,ini_file):
    '''
    Function to clean the fsps input and output file directories. 
    '''

    scripts_c = code_save_dir+'/scripts'
    ini_c = code_save_dir+'/model.ini'

    os.system('cp -r scripts %s'%scripts_c)
    os.system('cp  %s  %s'%(ini_file,ini_c))
    
    return 


def run_grumpy(input_dict):
    '''
    Function that runs the entire model on a single simulation suite

    Parameters:
    -----------
    model_inputs: dict, mah data path, cosmo_params, iniconf

    Returns:
    ----------
    None

    '''
    suite_name = input_dict["suite_name"]
    cosmo = convert(input_dict["cosmo"])
    iniconf = input_dict["iniconf"]

    #prepare all the track data
    data, good_index, tmdlmdlt, zspl, good_files = prepare_mh_tracks(cosmo=cosmo,iniconf=iniconf,suite_name = suite_name)

    # #run the integration now
    Mout, Mpeak, zpeak, rperturb = integrate_model(tmdlmdlt=tmdlmdlt, 
                                                    good_index=good_index, 
                                                    data=data, zspl=zspl, 
                                                    params=model_params,
                                                    cosmo=cosmo,
                                                    iniconf=iniconf)

    #run the post processing now
    summary_stats,sub_good_ind = post_process(good_index=good_index,
                                                good_files=good_files,
                                                tmdlmdlt=tmdlmdlt,
                                                data=data,
                                                params=model_params,
                                                cosmo=cosmo,
                                                zspl=zspl, 
                                                Mout=Mout,
                                                Mpeak=Mpeak,
                                                zpeak=zpeak,
                                                rperturb=rperturb,
                                                iniconf=iniconf)

    #run disk disruption if true
    if iniconf['run params']['run_disk_disruption'] == "True":
        disruption_dict = run_disruption_model(pickle_name=suite_name,data=data,zspl=zspl,iniconf=iniconf)
        #run the filter on the disruption output
        disruption_data = filter_disruption(disruption_dict=disruption_dict,sub_good_ind=sub_good_ind)
    else:
        disruption_data = None

    #store the final results 
    store_results(summary_stats=summary_stats,iniconf=iniconf,disruption_data=disruption_data,suite_name=suite_name)

    return


if __name__ == '__main__':
    #ignore runtime warnings
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    # read in command line arguments
    args = argument_parser().parse_args()
    # read parameters and information from the run config file 
    iniconf = read_config_file(config_file=args.ini)

    #check if all the paths exist. If not, create them
    #this final_output_dir contains all the final csv files
    #the track data folder will have sub-folders corresponding to each suite. eg. caterpillar_1, caterpillar_2 etc.
    pickle_mah_dir = iniconf['output paths']['pickle_mah_dir']    
    pickle_disruption_dir = iniconf['output paths']['pickle_disruption_dir']
    base_output_dir = iniconf['main paths']['output_dir']
    final_output_dir = iniconf['output paths']['final_output_dir']
    track_data_dir = iniconf['output paths']['final_output_dir'] + "/track_data"

    code_save_dir = final_output_dir + "/code_save"

    check_path_existence(all_paths = [base_output_dir, final_output_dir,track_data_dir,code_save_dir, pickle_mah_dir, pickle_disruption_dir])

    cosmo, model_params = extract_model_params(iniconf=iniconf)

    #copy the .ini file and scripts into the code subdir. 
    #This is for keeping track of the code that was used to run that specific model.
    save_code(code_save_dir,args.ini)

    run_fsps = iniconf['run params']['run_fsps']

    if run_fsps == "True":
        fsps_input_dir = iniconf['output paths']['fsps_input_dir']
        fsps_output_dir = iniconf['output paths']['fsps_output_dir']
        check_path_existence(all_paths=[fsps_input_dir,fsps_output_dir])

    #clean the fsps directories
    if run_fsps == "True":
        clean_fsps(iniconf=iniconf)

    mah_data_dir = iniconf['main paths']['mah_data_dir']

    #give notifications
    print_stage("The model results will be stored at: " + iniconf['output paths']['final_output_dir'],ch='-')

    #find the different suites we have to run model over
    #we can also provide a few suite names
    which_suites = iniconf['run params']['which_suites']

    if which_suites == "None":
        all_suites = np.array(glob.glob(mah_data_dir+"/*/"))
    else:
        #we have to read the list of folders that are provided
        temp_suites = which_suites.split(',')
        all_suites = []
        for tempi in temp_suites:
            #adding the front path to its name to make it consistent with the glob output above
            all_suites.append(mah_data_dir + "/" + tempi + "/")

    #NOTE THAT THE SUITE NAME DOES NOT HAVE TO BE CONTAINED IN THE TRACK FILE NAME

    #organize this. This is solely in caterpillar. 
    # all_suite_numbers = []
    # for sni in all_suites:
    #     all_suite_numbers.append( int(sni.replace(mah_data_dir+"/caterpillar_","")[:-6]) )
    # #we need to do -6 here because there is the "/" included at the end as well
    # suite_sort_inds = np.argsort(all_suite_numbers)

    # print_stage(all_suite_numbers)

    # all_suites = all_suites[suite_sort_inds]

    all_suites_iter = []
    all_suite_names = []

    for si in all_suites:
        suite_name = si.replace(mah_data_dir+"/","").replace("/","")
        all_suite_names.append(suite_name)
        iter_i = {"suite_name":suite_name,"cosmo":cosmo._asdict(),"iniconf": iniconf}

        all_suites_iter.append(iter_i)

    print_stage(all_suite_names)
     
    run_suite_parallel = iniconf['run params']['run_suite_parallel']
    ncores = int(iniconf['run params']['ncores'])
    #run the grumpy model on MAHs
    if run_suite_parallel == "True":
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
            executor.map(run_grumpy,all_suites_iter)
    else:
        #serial version
        for suite_i in all_suites_iter:
            run_grumpy(input_dict=suite_i)

    #run FSPS now
    if run_fsps == "True":
        #FSPS will be run at the same time on all the track data

        print_stage("FSPS is starting up now.", ch='-')
        
        run_fsps_python(iniconf=iniconf,cosmo=cosmo)

        print_stage("FSPS is finished.",ch="-")

        #add the mag tracks to the track data or summary file
        add_mags(suite_names=all_suite_names,iniconf=iniconf)


