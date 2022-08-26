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


def add_mags(suite_names=None,iniconf=None):
    '''
    Function that reads FSPS magnitude output and writes them to the summary data file for each suite. 

    Parameters:
    -----------
    suite_names: list, simulation suite names
    iniconf: dict, dictionary of parameters in the config file
    '''

    for si in suite_names:

        final_csv_path = iniconf['output paths']['final_output_dir'] + "/" + si + "_" + iniconf['output paths']['run_name'] + "_run.csv"

        try:
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
        except:
            print_stage("This path does not exist: %s"%final_csv_path)

    print_stage("Final summary files updated with FSPS magnitudes.", ch='-')

    return 


if __name__ == '__main__':
    #ignore runtime warnings
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    # read in command line arguments
    args = argument_parser().parse_args()
    # read parameters and information from the run config file 
    iniconf = read_config_file(config_file=args.ini)

    mah_data_dir = iniconf['data paths']['mah_data_dir']
    #find the different suites we have to run model over 
    all_suites = np.array(glob.glob(mah_data_dir+"/*/"))
    all_suite_names = []

    for si in all_suites:
        suite_name = si.replace(mah_data_dir+"/","").replace("/","")
        all_suite_names.append(suite_name)

    #add the mag tracks to the track data or summary file
    add_mags(suite_names=all_suite_names,iniconf=iniconf)