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
from galaxy_model import MH2
import pandas as pd
import pickle
from run_grumpy import extract_model_params
import numpy as np
from tqdm import tqdm
import concurrent.futures
from collections import namedtuple
from Chempy.solar_abundance import solar_abundances
from gen_chem_pickle import create_chem_pickle,create_chem_pickle_TRIAL
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from chem_params import extract_chem_params



def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def argument_parser():
    '''
    Function that parses the arguments passed while running a script
    '''
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # path to the config file with parameters and information about the run
    result.add_argument('-ini', dest='ini', type=str) 
    return result

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


# def compute_net_yield(yi_net,t_low, t_high,ssps_tbirths, ssps_z, ssps_ms):
#     '''
#     yi_net is the function that we want to use to compute the net yield.
    
#     Used for analytic yields
    
#     '''
#     #need to compute the yield between two time steps
#     #we also need to multiply this yields by the stellar mass!!!
#     #we can feed the entire thing to the interpolation directly!
#     t_age_uppers = t_high  - ssps_tbirths
#     t_age_lowers = t_low - ssps_tbirths

#     if np.min(t_age_uppers) < 0 or np.min(t_age_lowers) < 0:
#         print("ISSUE!! THERE HAS BEEN A NEGATIVE or ZERO TIME!!")
        
#     t_age_lowers[t_age_lowers == 0] == 1e-3

#     #this is the time arrays we will feed to interp object 
#     t_age_up_log = np.log10(t_age_uppers)
#     t_age_low_log = np.log10(t_age_lowers)

#     ssps_z_inis = np.log10(ssps_z)
#     #the ssps_z_inis should be of same length as t_age_uppers

#     y_upper = yi_net(ssps_z_inis,t_age_up_log,grid = False)
#     y_lower = yi_net(ssps_z_inis,t_age_low_log,grid = False)
    
#     y_delta = (y_upper - y_lower)*ssps_ms #we multiply all the yields by the mass of the SSP 

#     return np.sum(y_delta)

def save_code(code_save_dir,ini_file):
    '''
    Function to clean the fsps input and output file directories. 
    '''
    ini_c = code_save_dir+'/model.ini'

    os.system('cp  %s  %s'%(ini_file,ini_c))
    
    return 

def compute_Mn_S21_gross_yield(Mn_IA_net,Mn_CC_AGB_net,Mn_diff,t_low,t_high,ssps_tbirths,ssps_z,feh_value, ssps_ms, ini_frac,org_frac):
    '''

    In this function, we use the modified Mn yield from Sanders 21

    yi_net is the function for net yield
    yi_diff is the function for gross yield - net yield
    org_frac is the elemental fractional assumed in computation of this grid ... 
    it should be a single value

    feh_value is the feh values for all the SSPs under consideration
    We will need to conside the Mn yield of each SSP separately

    The relation is 
    y_gross_fnew = y_net + (fnew/forg) * (y_gross_forg - y_net)

    the (y_gross_forg - y_net) is decided by the fractions we used in our SSP model initialization
    this is something we should be storing so we can scale the net yield
    
    Used for analytic yields
    ini_frac is the initial fraction of hte 
    '''
    #need to compute the yield between two time steps
    #we also need to multiply this yields by the stellar mass!!!
    #we can feed the entire thing to the interpolation directly!
    t_age_uppers = t_high  - ssps_tbirths
    t_age_lowers = t_low - ssps_tbirths

    if np.min(t_age_uppers) < 0 or np.min(t_age_lowers) < 0:
        print("ISSUE!! THERE HAS BEEN A NEGATIVE or ZERO TIME!!")
        
    t_age_lowers[t_age_lowers == 0] == 1e-3

    #this is the time arrays we will feed to interp object 
    t_age_up_log = np.log10(t_age_uppers)
    t_age_low_log = np.log10(t_age_lowers)

    ssps_z_inis = np.log10(ssps_z)
    #feh value are already in log scale
    feh_inis = feh_value
    #the ssps_z_inis should be of same length as t_age_uppers

    y_Mn_Ia_upper = Mn_IA_net(feh_inis,t_age_up_log,grid = False)
    y_Mn_Ia_lower = Mn_IA_net(feh_inis,t_age_low_log,grid = False)

    y_Mn_CC_AGB_upper = Mn_CC_AGB_net(ssps_z_inis,t_age_up_log,grid = False)
    y_Mn_CC_AGB_lower = Mn_CC_AGB_net(ssps_z_inis,t_age_low_log,grid = False)

    # y_upper = yi_net(ssps_z_inis,t_age_up_log,grid = False)
    # y_lower = yi_net(ssps_z_inis,t_age_low_log,grid = False)
    
    #gross - net yield
    delta_gross_net_upper = Mn_diff(ssps_z_inis,t_age_up_log,grid = False)
    delta_gross_net_lower = Mn_diff(ssps_z_inis,t_age_low_log,grid = False)
    
    y_Mn_CC_AGB_gross_upper = y_Mn_CC_AGB_upper + (ini_frac/org_frac) * delta_gross_net_upper
    y_Mn_CC_AGB_gross_lower = y_Mn_CC_AGB_lower + (ini_frac/org_frac) * delta_gross_net_lower
    
    y_delta = (y_Mn_CC_AGB_gross_upper - y_Mn_CC_AGB_gross_lower)*ssps_ms + (y_Mn_Ia_upper - y_Mn_Ia_lower)*ssps_ms #we multiply all the yields by the mass of the SSP 

    return np.sum(y_delta)


def compute_gross_yield(yi_net,yi_diff,t_low, t_high,ssps_tbirths, ssps_z, ssps_ms, ini_frac,org_frac):
    '''
    yi_net is the function for net yield
    yi_diff is the function for gross yield - net yield
    org_frac is the elemental fractional assumed in computation of this grid ... 
    it should be a single value
    
    
    The relation is 
    y_gross_fnew = y_net + (fnew/forg) * (y_gross_forg - y_net)

    the (y_gross_forg - y_net) is decided by the fractions we used in our SSP model initialization
    this is something we should be storing so we can scale the net yield
    
    Used for analytic yields
    ini_frac is the initial fraction of hte 
    '''
    #need to compute the yield between two time steps
    #we also need to multiply this yields by the stellar mass!!!
    #we can feed the entire thing to the interpolation directly!
    t_age_uppers = t_high  - ssps_tbirths
    t_age_lowers = t_low - ssps_tbirths

    if np.min(t_age_uppers) < 0 or np.min(t_age_lowers) < 0:
        print("ISSUE!! THERE HAS BEEN A NEGATIVE or ZERO TIME!!")
        
    t_age_lowers[t_age_lowers == 0] == 1e-3

    #this is the time arrays we will feed to interp object 
    t_age_up_log = np.log10(t_age_uppers)
    t_age_low_log = np.log10(t_age_lowers)

    ssps_z_inis = np.log10(ssps_z)
    #the ssps_z_inis should be of same length as t_age_uppers

    y_upper = yi_net(ssps_z_inis,t_age_up_log,grid = False)
    y_lower = yi_net(ssps_z_inis,t_age_low_log,grid = False)
    
    #gross - net yield
    delta_gross_net_upper = yi_diff(ssps_z_inis,t_age_up_log,grid = False)
    delta_gross_net_lower = yi_diff(ssps_z_inis,t_age_low_log,grid = False)
    
    y_gross_upper = y_upper + (ini_frac/org_frac) * delta_gross_net_upper
    y_gross_lower = y_lower + (ini_frac/org_frac) * delta_gross_net_lower
    
    y_delta = (y_gross_upper - y_gross_lower)*ssps_ms #we multiply all the yields by the mass of the SSP 

    return np.sum(y_delta)


def compute_yZ_gross(yZ_gross,t_low, t_high,ssps_tbirths, ssps_z, ssps_ms):
    '''
    Returns the total metal yield ...
    I do not think the individual metal fractions (ie ini_frac) matters here 
    
    So it is possible to have two SSPs with same ssp_z, however, different metal fractions. 
    The different metal fractions does not affect the net yield as expected
    however it will affect the gross yield. 
    
    This will result in some inconsistency, however, it should be fine. There are larger uncertainties here. 
    
    '''
    #need to compute the yield between two time steps
    #we also need to multiply this yields by the stellar mass!!!
    #we can feed the entire thing to the interpolation directly!
    t_age_uppers = t_high  - ssps_tbirths
    t_age_lowers = t_low - ssps_tbirths

    if np.min(t_age_uppers) < 0 or np.min(t_age_lowers) < 0:
        print("ISSUE!! THERE HAS BEEN A NEGATIVE or ZERO TIME!!")
        
    t_age_lowers[t_age_lowers == 0] == 1e-3


    #this is the time arrays we will feed to interp object 
    t_age_up_log = np.log10(t_age_uppers)
    t_age_low_log = np.log10(t_age_lowers)

    ssps_z_inis = np.log10(ssps_z)
    #the ssps_z_inis should be of same length as t_age_uppers

    y_upper = yZ_gross(ssps_z_inis,t_age_up_log,grid = False)
    y_lower = yZ_gross(ssps_z_inis,t_age_low_log,grid = False)
        
    y_delta = (y_upper - y_lower)*ssps_ms #we multiply all the yields by the mass of the SSP 

    return np.sum(y_delta)

def compute_all_ms_surv(ms_surv_interp,t_low,ssps_tbirths, ssps_z, ssps_ms):
    '''
    This function takes in the list of ssp info and computes the surviving stellar mass so far
    
    the interpolation object takes (logZ and logtime) and returns the quantity assuming 1Msun SSP, so we have to renormalize it
    Used for analytic yields
    '''
    t_age_lowers = t_low - ssps_tbirths
    if np.min(t_age_lowers) < 0:
        print("ISSUE!! THERE HAS BEEN A NEGATIVE TIME!! in ms surv")
    #this is the time arrays we will feed to interp object 
    t_age_low_log = np.log10(t_age_lowers)
    ssps_z_inis = np.log10(ssps_z)    
    
    t_age_lowers[t_age_lowers == 0] == 1e-3

        
    ms_survs = ms_surv_interp(ssps_z_inis,t_age_low_log,grid = False)

    ms_survs_tot = np.sum(ms_survs*ssps_ms)
    #the ms_survs*ssps_ms is the surviving mass of the SSP
    #we are returning so we can calculate the amount of element in 
    
    return ms_survs_tot,ms_survs*ssps_ms 
    

def compute_all_feedback_ms(ms_feedback_interp,t_low, t_high, ssps_tbirths,ssps_z, ssps_ms ):
    '''
    This function computes the total mass ejected by the SSPs between two time steps 
    Used for analytic yields
    '''
    t_age_lowers = t_low - ssps_tbirths
    t_age_uppers = t_high - ssps_tbirths
    
    if np.min(t_age_lowers) < 0 or np.min(t_age_uppers) < 0:
        print("ISSUE!! THERE HAS BEEN A NEGATIVE TIME!! in ms feedback")
        
        
    t_age_lowers[t_age_lowers == 0] == 1e-3
    #this is the time arrays we will feed to interp object 
    t_age_low_log = np.log10(t_age_lowers)
    t_age_up_log = np.log10(t_age_uppers)
    
    ssps_z_inis = np.log10(ssps_z) 
    
    ms_feedback_low = ms_feedback_interp(ssps_z_inis,t_age_low_log,grid = False )
    ms_feedback_up = ms_feedback_interp(ssps_z_inis,t_age_up_log,grid = False )

    ms_ej_feed = np.sum((ms_feedback_up - ms_feedback_low)*ssps_ms)

    return ms_ej_feed


def store_chem_results(final_dict=None,elements_to_track=None,track_path=None,iniconf=None,final_store_path=None):
    '''
    function that will save the chemical model results to existing file structures or make new ones
    
    final_store_path is the folder name in which all the chem_evo and ssps_prop files are stored
    '''

    sub_dict = {"t":final_dict["t"],"Ms":final_dict["Ms"], "MZs":final_dict["MZs"],"Mg":final_dict["Mg"],
                "MZg":final_dict["MZg"]}

    MXs = final_dict['MXs']
    MXg = final_dict['MXg']


    ssps_props = {}
    ssps_ini_fracs = final_dict["ssps_ini_zfracs"]
    ssps_ms = final_dict["ssps_ms"]
    ssps_tbirth = final_dict["ssps_tbirth"]
    ssps_z = final_dict["ssps_z"]
    ssps_ms_surv = final_dict["ssps_surv_ms"]


    for i,ei in enumerate(elements_to_track):
        sub_dict["Ms_" + ei] = MXs[:,i]
        sub_dict["Mg_" + ei] = MXg[:,i]
        ssps_props[ei] = ssps_ini_fracs[:,i]

    ssps_props["formation_mass"] = ssps_ms
    ssps_props["formation_time"] = ssps_tbirth
    ssps_props["formation_zmel"] = ssps_z
    ssps_props["surviving_mass"] = ssps_ms_surv

    #we save this dict as a dataframe in tracks folder
    df = pd.DataFrame(sub_dict)

    input_track_path = iniconf['chem setup']['input_track_path']
    track_path_only = track_path.replace(input_track_path + "/track_data/","")
    
    chem_track_path = final_store_path + "/" + track_path_only.replace("_track.csv","") + "_chem_evo.csv"

    df.to_csv(chem_track_path,index = False)

    #we will also store the different ssps_ini_zfracs
    df_ssps = pd.DataFrame(ssps_props)
    ssps_track_path = final_store_path + "/" + track_path_only.replace("_track.csv","") + "_ssps_prop.csv"
    df_ssps.to_csv(ssps_track_path,index = False)

    return 

    
def run_chempy_track(input_stuff):

    
    '''
    This function runs the chempy SSP model on a GRUMPY track to 
    
    Parameters:
    df = dataframe object, this is the dataframe that contains the track information
    Z_IGM = this is the value of the assumed IGM metallicity floor, we draw this randomly from a distribution..
            this is unlike in our fiducial model we where we assume a fixed -3. (Z_IGM IS NOT IN SOLAR UNITS)
    all_interps_dict = this is a dictionary that contains all the relevant 2d interpolations e.g. net yields, ms surv etc. 
    ini_metal_rfrac = the initial relative metal abundance fractions (no H and He in this)
    model_params and cosmo_params are the 

    One fun point ->
    We need to follow the total metal content Z for SSP etc. 
    however, as a result, we do not need to track all the elemental abundances
    We already have a separate 2d interpolation for Z yield :)
    
    Note that I will only be modelling the initial abundance of the SSP
    Assume that the abundances do not change after evolution

    (df,rpd_val=None,nsteps = 500,cosmo=None,evolve_wind = False,
                     evolve_star = False,verbose = False,elements_to_track = None,
                     all_interps_dict = None, ini_metal_rfrac = None,metals_to_track=None,
                     model_params=None,cosmo_params = None,chem_params = None):
    
    '''


    track_path = input_stuff["track_path"]
    nsteps = input_stuff["nsteps"]
    evolve_wind = input_stuff["evolve_wind"]
    evolve_star = input_stuff["evolve_star"]
    elements_to_track = input_stuff["elements_to_track"]
    all_interps_dict = input_stuff["all_interps_dict"]
    ini_metal_rfrac = input_stuff["ini_metal_rfrac"]
    metals_to_track = input_stuff["metals_to_track"]
    model_params = input_stuff["model_params"]
    cosmo_params = input_stuff["cosmo_params"]
    chem_params = input_stuff["chem_params"]
    verbose = input_stuff["verbose"]
    rpd_val = input_stuff["rpd_val"]
    iniconf = input_stuff["iniconf"]
    final_store_path = input_stuff["final_path"]

    chem_params = convert(chem_params)
    model_params = convert(model_params)
    cosmo_params = convert(cosmo_params)

    df = pd.read_csv(track_path)

    Zsun = model_params.Zsun

    if chem_params.log_zigm_sig == 0:
        Z_IGM = Zsun * 10**(chem_params.log_zigm_mean)
    else:
        Z_IGM = (10**np.random.normal(chem_params.log_zigm_mean, chem_params.log_zigm_sig))*Zsun
    
    mgin_cumu = np.array(df["Mgin"])
    mgout_cumu = np.array(df["Mgout"])

    mg_evo = np.array(df["Mg"])
    ms_evo = np.array(df["Ms"])
    mh_evo = np.array(df["Mh"])
    tt = np.array(df["t"])

    
    #we convert them to linear interpolation objects
    mgin_spl = interpolate.interp1d(tt, mgin_cumu)
    mh_spl = interpolate.interp1d(tt,mh_evo)
    if evolve_star == False:
        ms_spl = interpolate.interp1d(tt,ms_evo)
    if evolve_wind == False:
        mgout_spl = interpolate.interp1d(tt, mgout_cumu)
    
    tstart = tt[0]
    ms_start = 10 #this is the initial stellar mass within the halo
    mg_start = mg_evo[0] #this is the initial gas mass in the halo. We take this from our pre-evolved track
    #Z_IGM is drawn from a distribution, and is fed into this function
    mg_z_start = mg_start * Z_IGM
    
    mg_X = np.zeros_like(elements_to_track,dtype = float) + -99
    
    #we initially assume that the relative metal fractions/abundances are the same as solar
    #if I want to focus on [Fe/H] < -3 stars then this assumption is less valid.
    #the same assumption is made for IGM gas as well
    
    ini_element_ratios = iniconf['chem model']['ini_element_ratios']

    if ini_element_ratios == "solar":

        for ei in elements_to_track:
            if ei == "H":
                mg_X[ elements_to_track == "H" ] = 0.75*(mg_start - mg_z_start)
            elif ei == "He":
                mg_X[ elements_to_track == "He" ] = 0.25*(mg_start - mg_z_start)
            else:
                #this for all the rest of the metals 
                # print(ini_metal_rfrac[metals_to_track == ei])
                mg_X[elements_to_track == ei] = mg_z_start * ini_metal_rfrac[metals_to_track == ei]

    else:
        #we use user inputted element ratios

        ini_metal_ratios = np.array(ini_element_ratios.split(",")).astype(float)
        for j,ei in enumerate(elements_to_track): #we skip over the H and He as this is just for metals
            if ei == "H":
                mg_X[ elements_to_track == "H" ] = 0.75*(mg_start - mg_z_start)
            elif ei == "He":
                mg_X[ elements_to_track == "He" ] = 0.25*(mg_start - mg_z_start)
            else:
                # print(ini_metal_ratios[j - 2])
                mg_X[elements_to_track == ei] = mg_z_start * ini_metal_ratios[j - 2]

    if np.min(mg_X) < 0:
        print(mg_X)
        print(elements_to_track)
        print("SOME VALUE WAS NOT UPDATED!")
        
    #the time array we will evaluating the SSPs and integrating to compute the time evolution...
    time_steps = np.linspace(tt[0],tt[-1],nsteps)
    #converting this to redshifts. This is needed to be fed to MH2 function if being used
    if evolve_star == True:
        #load the pickle object
        zspl_path = iniconf['data paths']['pickle_mah_dir'] + "/zspl.pickle"
        with open(zspl_path, 'rb') as f:
            zspl_obj = pickle.load(f)
        zspl = zspl_obj["zspl"]
        #need to convert time steps to redshift 
        zred_steps = zspl(time_steps)
    
    #these are the lists where we will store the properties of the SSP
    ssps_tbirths = [] #the formation time of SSP
    ssps_z = [] #the SSP formation metallicity Z (all metal content)
    ssps_ms = [] #the mass of this SSP
    ssps_ini_zfracs = [] #the SSP initialization metal FRACTIONS (including H,He) for all the metals being tracked. 
    
    #we append the initial SSP quantities here.
    #these are the values of the first SSP
    ssps_tbirths.append(time_steps[0])
    ssps_z.append(mg_z_start/mg_start) #assume same metal properties as gas
    ssps_ms.append(ms_start)
    ssps_ini_zfracs.append( mg_X/mg_start )

    ssps_tbirths = np.array(ssps_tbirths)
    ssps_z = np.array(ssps_z)
    ssps_ms = np.array(ssps_ms)
    ssps_ini_zfracs = np.array(ssps_ini_zfracs)

    #these are the individual elemental metallicities in the IGM matter
    Z_X_IGM = mg_X / mg_start
            
    #the initial elemental abundance locked inside stars
    ms_x_start = ms_start * Z_X_IGM
    ms_z_start = ms_start * Z_IGM
    #these are the evolution tracks that we will be population
    mg_tracks = np.array([mg_start]) 
    ms_tracks = np.array([ms_start])
    msx_tracks = np.array([ms_x_start])
    msz_tracks = np.array([ms_z_start])
    mgz_tracks = np.array([mg_z_start]) #this is the initial total metal mass
    mgx_tracks = np.array([mg_X])
    
    t_tracks = np.array([tstart])
    
    #we do the euler forward integration now ... 
    
    #some relevant stuff needed in SSP yield computation
    org_elements = np.array(all_interps_dict["elements"])
    org_fracs = np.array(all_interps_dict["solar_fractions"]) #the list of elemental fractions used in 2d interpolation
    
    Rloss1 = model_params.Rloss1
    eta_norm = model_params.eta_norm
    eta_power = model_params.eta_power
    eta_c = model_params.eta_c
    eta_mass = model_params.eta_mass

    #the above are lists/arrays of same size, 1-to-1 correspondance between 
    for i in range(len(time_steps) -1 ):
        ti = time_steps[i]
        dt = time_steps[i+1] - time_steps[i] #this is the time step to be taken 
        
        #load the current values 
        Mg = mg_tracks[-1]
        Ms = ms_tracks[-1]
        MgX = mgx_tracks[-1]
        MgZ = mgz_tracks[-1]
    
        if evolve_star == True:
            z = zred_steps[i]
            rpd = rpd_val
            SigHIth = model_params.SigHIth
            tausf = model_params.tausf
            h2_model = model_params.h2_model

            h23i = cosmo_params.h23i
            Om0 = cosmo_params.Om0
            OmL = cosmo_params.OmL

            
            if i == 0:
                sfr = 0 #as their is no star formation at the very start we set it to zero (just like GRUMPY)

            M_H2, _ = MH2(Mh=mh_spl(ti), Mg=Mg, MZg=MgZ, sfr=sfr, z=z, h23i=h23i, Om0=Om0, OmL=OmL, 
                          Zsun= Zsun, Z_IGM = Z_IGM/Zsun, SigHIth=SigHIth, rpert = rpd,h2_model = h2_model )

            #compute the star formation rate and hence the new star mass formed in Delta_t step
            sfr = M_H2 / tausf
            delta_ms = sfr * dt #this is the new stellar mass formed
            
        if evolve_star == False:
            
            #we only need to know SFR from the Ms track
            sfr = (ms_spl(time_steps[i+1]) - ms_spl(time_steps[i]))/dt
            delta_ms = sfr * dt / Rloss1 #this is the new stellar mass forme
            #we correct for the assumed stellar return mass fraction assumed in org grumpy run
    
        if evolve_wind == True:
            eta_wind = 0.
            if Ms > 1:
                eta_wind = np.maximum( 0, eta_norm*(Ms/eta_mass)**(-1*eta_power) - eta_c )
                eta_wind = np.minimum(eta_wind, 2000.)
                wind_loss = dt*sfr*eta_wind
                
        if evolve_wind == False:
            #the wind evolution is based on the star formation rate and so we do not need to recompute it
            wind_loss = mgout_spl(time_steps[i+1]) - mgout_spl(time_steps[i])
            
        #let us compute the new gas that has been accreting into halo during this delta_t
        mgin_new = mgin_spl(time_steps[i+1]) - mgin_spl(time_steps[i])
        #the corresponding enrichment of elements X through IGM is
        
        mginX_new = mgin_new * Z_X_IGM
        mginZ_new = mgin_new * Z_IGM

        #total metal mass injected into ISM
        #CHECK BELOW STEP
        Z_yield = compute_yZ_gross(all_interps_dict["yZ_gross"],time_steps[i], time_steps[i+1], ssps_tbirths,ssps_z, ssps_ms)
        ms_survs, ms_survs_i = compute_all_ms_surv(all_interps_dict["ms_surv"],time_steps[i],ssps_tbirths, ssps_z, ssps_ms)
        ssp_feedbacks = compute_all_feedback_ms(all_interps_dict["ms_feedback"],time_steps[i], time_steps[i+1], ssps_tbirths,ssps_z, ssps_ms )

        #X yields are the amount of element X injected into ISM. We thus want the gross yields...
        X_yields = []
                
        #looping over each element...
        for f,xf in enumerate(elements_to_track):
            org_fracs_f = org_fracs[org_elements == xf]
            ind_f = int(np.where(elements_to_track == xf)[0])
            ssps_ini_zfracs_f = ssps_ini_zfracs[:,ind_f]
            #we extract the relevant column that corresponds to the element under inspection here...
#             compute_gross_yield(yi_net,yi_diff,t_low, t_high,ssps_tbirths, ssps_z, ssps_ms, ini_frac,org_frac)
            xf_yields = compute_gross_yield(all_interps_dict[xf+"_net"],all_interps_dict[xf+"_diff"],time_steps[i], time_steps[i+1],ssps_tbirths, ssps_z, ssps_ms, ssps_ini_zfracs_f,org_fracs_f)
            X_yields.append(xf_yields)
            
        X_yields = np.array(X_yields)
 
        #WE UPDATE VALUES FOR NEXT TIME STEP NOW
        Mg_next = Mg + mgin_new + ssp_feedbacks  - delta_ms - wind_loss 
        if Mg_next < 0:
            print("NEGATIVE GAS MASS!! SOMETHING IS WRONG!!!")
        #to get new stellar mass we compute the suriving mass of all previous SSPs and we add that with the new stellar mass
        Ms_next = delta_ms + ms_survs
        MsX_next = np.dot(ms_survs_i,ssps_ini_zfracs) + delta_ms * (MgX/Mg)
        MsZ_next = np.sum(ms_survs_i*ssps_z) + delta_ms * (MgZ/Mg)
        #this is using assumption that metals have been completely mixed
        MgX_next = MgX + mginX_new + X_yields - delta_ms * (MgX/Mg) - wind_loss * (MgX/Mg)
        
        MgZ_next = MgZ + mginZ_new + Z_yield - delta_ms * (MgZ/Mg) - wind_loss * (MgZ/Mg)

       #this is an easy fix to solve the negative metal mass in gas issue
        if MgZ_next < 0:
            MgZ_next = Mg_next * (MgZ/Mg)

        if np.min(MgX_next) < 0:
            MgX_next = (MgX/Mg)*Mg_next

        #ADD A CHECKER THAT CHECKS IF THERE ARE ANY NEGATIVE TERMS BEING MADE..          
        if Mg_next < 0 or Ms_next < 0 or MgZ_next < 0 or MsZ_next < 0 or np.min(MgX_next) < 0 or np.min(MsX_next) <0:
            print("--")
            print("Negative values are being generated. The most likely cause is high wind_loss. Increasing nsteps will fix it. ")
            print("MgZ_next:",MgZ_next)
            print("MgX_next:",np.min(MgX_next))
            print("MgZ_old:",MgZ)
            print("Mg_next:",Mg_next)
            print("Mg_old:",Mg)            
            print("Z_yield:",Z_yield)
            print("mginZ:",mginZ_new)
            print("wind_loss:",wind_loss * (MgZ/Mg))
            print("star_loss:",delta_ms * (MgZ/Mg))
            print("--")

        #add the updated values to the track data!
        mg_tracks = np.concatenate((mg_tracks,[Mg_next]))
        ms_tracks = np.concatenate((ms_tracks,[Ms_next]))
        mgx_tracks = np.vstack((mgx_tracks,MgX_next))
        msx_tracks = np.vstack((msx_tracks,MsX_next))
        msz_tracks = np.concatenate((msz_tracks,[MsZ_next]))
        #note that msx is the TOTAL mass of element X locked in all stars of that galaxy
        mgz_tracks = np.concatenate((mgz_tracks,[MgZ_next]))
        t_tracks = np.concatenate((t_tracks,[time_steps[i+1]]))
        
        #in principle, we could add a non-zero threshold here to decrease computational time
        if delta_ms > 0:
            if verbose == True:
                print(delta_ms)
            #a new SSP was born and so we append its properties to the SSP list
            ssps_ms = np.concatenate((ssps_ms,[delta_ms]))
            ssps_tbirths = np.concatenate((ssps_tbirths,[time_steps[i+1]]))
            ssps_z = np.concatenate( (ssps_z,[MgZ/Mg]) )
            ssps_ini_zfracs = np.vstack((ssps_ini_zfracs,MgX/Mg)) #these are the individual elemental fractions in the gas.
        
        else:
            #no star is being formed and so we do not need anything to the SSP lists 
            pass 


    #time steps [-1] is the final time step
    _, ms_survs_final = compute_all_ms_surv(all_interps_dict["ms_surv"],time_steps[-1],ssps_tbirths, ssps_z, ssps_ms)

    #now we should save this resul

    temp_dict = {"t":t_tracks,"Mg":mg_tracks,"Ms":ms_tracks,"MXg":mgx_tracks,"MZg":mgz_tracks,"MZs":msz_tracks,"MXs":msx_tracks,"Z_IGM":Z_IGM,
                  "ssps_ms":ssps_ms, "ssps_tbirth":ssps_tbirths,"ssps_z":ssps_z,'ssps_ini_zfracs':ssps_ini_zfracs,"ssps_surv_ms":ms_survs_final}
        

    store_chem_results(final_dict=temp_dict,track_path=track_path,iniconf=iniconf,elements_to_track=elements_to_track,final_store_path = final_store_path)

    return 


def run_chempy_track_TRIAL(input_stuff):

    
    '''
    This function runs the chempy SSP model on a GRUMPY track to 
    
    Parameters:
    df = dataframe object, this is the dataframe that contains the track information
    Z_IGM = this is the value of the assumed IGM metallicity floor, we draw this randomly from a distribution..
            this is unlike in our fiducial model we where we assume a fixed -3. (Z_IGM IS NOT IN SOLAR UNITS)
    all_interps_dict = this is a dictionary that contains all the relevant 2d interpolations e.g. net yields, ms surv etc. 
    ini_metal_rfrac = the initial relative metal abundance fractions (no H and He in this)
    model_params and cosmo_params are the 

    One fun point ->
    We need to follow the total metal content Z for SSP etc. 
    however, as a result, we do not need to track all the elemental abundances
    We already have a separate 2d interpolation for Z yield :)
    
    Note that I will only be modelling the initial abundance of the SSP
    Assume that the abundances do not change after evolution

    (df,rpd_val=None,nsteps = 500,cosmo=None,evolve_wind = False,
                     evolve_star = False,verbose = False,elements_to_track = None,
                     all_interps_dict = None, ini_metal_rfrac = None,metals_to_track=None,
                     model_params=None,cosmo_params = None,chem_params = None):
    
    '''


    track_path = input_stuff["track_path"]
    nsteps = input_stuff["nsteps"]
    evolve_wind = input_stuff["evolve_wind"]
    evolve_star = input_stuff["evolve_star"]
    elements_to_track = input_stuff["elements_to_track"]
    all_interps_dict = input_stuff["all_interps_dict"]
    ini_metal_rfrac = input_stuff["ini_metal_rfrac"]
    metals_to_track = input_stuff["metals_to_track"]
    model_params = input_stuff["model_params"]
    cosmo_params = input_stuff["cosmo_params"]
    chem_params = input_stuff["chem_params"]
    verbose = input_stuff["verbose"]
    rpd_val = input_stuff["rpd_val"]
    iniconf = input_stuff["iniconf"]
    final_store_path = input_stuff["final_path"]

    chem_params = convert(chem_params)
    model_params = convert(model_params)
    cosmo_params = convert(cosmo_params)

    df = pd.read_csv(track_path)

    Zsun = model_params.Zsun

    if chem_params.log_zigm_sig == 0:
        Z_IGM = Zsun * 10**(chem_params.log_zigm_mean)
    else:
        Z_IGM = (10**np.random.normal(chem_params.log_zigm_mean, chem_params.log_zigm_sig))*Zsun
    
    mgin_cumu = np.array(df["Mgin"])
    mgout_cumu = np.array(df["Mgout"])

    mg_evo = np.array(df["Mg"])
    ms_evo = np.array(df["Ms"])
    mh_evo = np.array(df["Mh"])
    tt = np.array(df["t"])

    
    #we convert them to linear interpolation objects
    mgin_spl = interpolate.interp1d(tt, mgin_cumu)
    mh_spl = interpolate.interp1d(tt,mh_evo)
    if evolve_star == False:
        ms_spl = interpolate.interp1d(tt,ms_evo)
    if evolve_wind == False:
        mgout_spl = interpolate.interp1d(tt, mgout_cumu)
    
    tstart = tt[0]
    ms_start = 10 #this is the initial stellar mass within the halo
    mg_start = mg_evo[0] #this is the initial gas mass in the halo. We take this from our pre-evolved track
    #Z_IGM is drawn from a distribution, and is fed into this function
    mg_z_start = mg_start * Z_IGM
    
    mg_X = np.zeros_like(elements_to_track,dtype = float) + -99
    
    #we initially assume that the relative metal fractions/abundances are the same as solar
    #if I want to focus on [Fe/H] < -3 stars then this assumption is less valid.
    #the same assumption is made for IGM gas as well
    
    ini_element_ratios = iniconf['chem model']['ini_element_ratios']

    if ini_element_ratios == "solar":

        for ei in elements_to_track:
            if ei == "H":
                mg_X[ elements_to_track == "H" ] = 0.75*(mg_start - mg_z_start)
            elif ei == "He":
                mg_X[ elements_to_track == "He" ] = 0.25*(mg_start - mg_z_start)
            else:
                #this for all the rest of the metals 
                # print(ini_metal_rfrac[metals_to_track == ei])
                mg_X[elements_to_track == ei] = mg_z_start * ini_metal_rfrac[metals_to_track == ei]

    else:
        #we use user inputted element ratios

        ini_metal_ratios = np.array(ini_element_ratios.split(",")).astype(float)
        for j,ei in enumerate(elements_to_track): #we skip over the H and He as this is just for metals
            if ei == "H":
                mg_X[ elements_to_track == "H" ] = 0.75*(mg_start - mg_z_start)
            elif ei == "He":
                mg_X[ elements_to_track == "He" ] = 0.25*(mg_start - mg_z_start)
            else:
                # print(ini_metal_ratios[j - 2])
                mg_X[elements_to_track == ei] = mg_z_start * ini_metal_ratios[j - 2]

    if np.min(mg_X) < 0:
        print(mg_X)
        print(elements_to_track)
        print("SOME VALUE WAS NOT UPDATED!")
        
    #the time array we will evaluating the SSPs and integrating to compute the time evolution...
    time_steps = np.linspace(tt[0],tt[-1],nsteps)
    #converting this to redshifts. This is needed to be fed to MH2 function if being used
    if evolve_star == True:
        #load the pickle object
        zspl_path = iniconf['data paths']['pickle_mah_dir'] + "/zspl.pickle"
        with open(zspl_path, 'rb') as f:
            zspl_obj = pickle.load(f)
        zspl = zspl_obj["zspl"]
        #need to convert time steps to redshift 
        zred_steps = zspl(time_steps)
    
    #these are the lists where we will store the properties of the SSP
    ssps_tbirths = [] #the formation time of SSP
    ssps_z = [] #the SSP formation metallicity Z (all metal content)
    ssps_ms = [] #the mass of this SSP
    ssps_ini_zfracs = [] #the SSP initialization metal FRACTIONS (including H,He) for all the metals being tracked. 
    
    #we append the initial SSP quantities here.
    #these are the values of the first SSP
    ssps_tbirths.append(time_steps[0])
    ssps_z.append(mg_z_start/mg_start) #assume same metal properties as gas
    ssps_ms.append(ms_start)
    ssps_ini_zfracs.append( mg_X/mg_start )

    ssps_tbirths = np.array(ssps_tbirths)
    ssps_z = np.array(ssps_z)
    ssps_ms = np.array(ssps_ms)
    ssps_ini_zfracs = np.array(ssps_ini_zfracs)

    #these are the individual elemental metallicities in the IGM matter
    Z_X_IGM = mg_X / mg_start
            
    #the initial elemental abundance locked inside stars
    ms_x_start = ms_start * Z_X_IGM
    ms_z_start = ms_start * Z_IGM
    #these are the evolution tracks that we will be population
    mg_tracks = np.array([mg_start]) 
    ms_tracks = np.array([ms_start])
    msx_tracks = np.array([ms_x_start])
    msz_tracks = np.array([ms_z_start])
    mgz_tracks = np.array([mg_z_start]) #this is the initial total metal mass
    mgx_tracks = np.array([mg_X])
    
    t_tracks = np.array([tstart])
    
    #we do the euler forward integration now ... 
    
    #some relevant stuff needed in SSP yield computation
    org_elements = np.array(all_interps_dict["elements"])
    org_fracs = np.array(all_interps_dict["solar_fractions"]) #the list of elemental fractions used in 2d interpolation
    
    Rloss1 = model_params.Rloss1
    eta_norm = model_params.eta_norm
    eta_power = model_params.eta_power
    eta_c = model_params.eta_c
    eta_mass = model_params.eta_mass

    #the above are lists/arrays of same size, 1-to-1 correspondance between 
    for i in range(len(time_steps) -1 ):
        ti = time_steps[i]
        dt = time_steps[i+1] - time_steps[i] #this is the time step to be taken 
        
        #load the current values 
        Mg = mg_tracks[-1]
        Ms = ms_tracks[-1]
        MgX = mgx_tracks[-1]
        MgZ = mgz_tracks[-1]
    
        if evolve_star == True:
            z = zred_steps[i]
            rpd = rpd_val
            SigHIth = model_params.SigHIth
            tausf = model_params.tausf
            h2_model = model_params.h2_model

            h23i = cosmo_params.h23i
            Om0 = cosmo_params.Om0
            OmL = cosmo_params.OmL

            
            if i == 0:
                sfr = 0 #as their is no star formation at the very start we set it to zero (just like GRUMPY)

            M_H2, _ = MH2(Mh=mh_spl(ti), Mg=Mg, MZg=MgZ, sfr=sfr, z=z, h23i=h23i, Om0=Om0, OmL=OmL, 
                          Zsun= Zsun, Z_IGM = Z_IGM/Zsun, SigHIth=SigHIth, rpert = rpd,h2_model = h2_model )

            #compute the star formation rate and hence the new star mass formed in Delta_t step
            sfr = M_H2 / tausf
            delta_ms = sfr * dt #this is the new stellar mass formed
            
        if evolve_star == False:
            
            #we only need to know SFR from the Ms track
            sfr = (ms_spl(time_steps[i+1]) - ms_spl(time_steps[i]))/dt
            delta_ms = sfr * dt / Rloss1 #this is the new stellar mass forme
            #we correct for the assumed stellar return mass fraction assumed in org grumpy run
    
        if evolve_wind == True:
            eta_wind = 0.
            if Ms > 1:
                eta_wind = np.maximum( 0, eta_norm*(Ms/eta_mass)**(-1*eta_power) - eta_c )
                eta_wind = np.minimum(eta_wind, 2000.)
                wind_loss = dt*sfr*eta_wind
                
        if evolve_wind == False:
            #the wind evolution is based on the star formation rate and so we do not need to recompute it
            wind_loss = mgout_spl(time_steps[i+1]) - mgout_spl(time_steps[i])
            
        #let us compute the new gas that has been accreting into halo during this delta_t
        mgin_new = mgin_spl(time_steps[i+1]) - mgin_spl(time_steps[i])
        #the corresponding enrichment of elements X through IGM is
        
        mginX_new = mgin_new * Z_X_IGM
        mginZ_new = mgin_new * Z_IGM

        #total metal mass injected into ISM
        #CHECK BELOW STEP
        Z_yield = compute_yZ_gross(all_interps_dict["yZ_gross"],time_steps[i], time_steps[i+1], ssps_tbirths,ssps_z, ssps_ms)
        ms_survs, ms_survs_i = compute_all_ms_surv(all_interps_dict["ms_surv"],time_steps[i],ssps_tbirths, ssps_z, ssps_ms)
        ssp_feedbacks = compute_all_feedback_ms(all_interps_dict["ms_feedback"],time_steps[i], time_steps[i+1], ssps_tbirths,ssps_z, ssps_ms )

        #X yields are the amount of element X injected into ISM. We thus want the gross yields...
        X_yields = []
                
        #looping over each element...
        for f,xf in enumerate(elements_to_track):
            org_fracs_f = org_fracs[org_elements == xf]
            ind_f = int(np.where(elements_to_track == xf)[0])
            ssps_ini_zfracs_f = ssps_ini_zfracs[:,ind_f]

            if xf == "Mn":
                Mn_IA_interp = all_interps_dict["Mn_S21_IA_net"]
                Mn_CC_AGB_interp = all_interps_dict["Mn_S21_CC_AGB_net"]
                Mn_diff_interp = all_interps_dict["Mn_diff"]

        
                ind_Fe = int(np.where(elements_to_track == "Fe")[0])
                ind_H = int(np.where(elements_to_track == "H")[0])

                ssps_ini_Fe_fracs_f = ssps_ini_zfracs[:,ind_Fe]
                ssps_ini_H_fracs_f = ssps_ini_zfracs[:,ind_H]

                fe_solar = 0.00129197
                h_solar = 0.73739777

                feh_value = np.log10((ssps_ini_Fe_fracs_f/ssps_ini_H_fracs_f)/(fe_solar/h_solar))

                mn_s21_yield = compute_Mn_S21_gross_yield(Mn_IA_interp,Mn_CC_AGB_interp,Mn_diff_interp,time_steps[i], time_steps[i+1],ssps_tbirths, ssps_z,feh_value,  ssps_ms, ssps_ini_zfracs_f,org_fracs_f)
                X_yields.append(mn_s21_yield)

            else:
                
                #we extract the relevant column that corresponds to the element under inspection here...
    #             compute_gross_yield(yi_net,yi_diff,t_low, t_high,ssps_tbirths, ssps_z, ssps_ms, ini_frac,org_frac)
                xf_yields = compute_gross_yield(all_interps_dict[xf+"_net"],all_interps_dict[xf+"_diff"],time_steps[i], time_steps[i+1],ssps_tbirths, ssps_z, ssps_ms, ssps_ini_zfracs_f,org_fracs_f)
                X_yields.append(xf_yields)
            
        X_yields = np.array(X_yields)
 
        #WE UPDATE VALUES FOR NEXT TIME STEP NOW
        Mg_next = Mg + mgin_new + ssp_feedbacks  - delta_ms - wind_loss 
        if Mg_next < 0:
            print("NEGATIVE GAS MASS!! SOMETHING IS WRONG!!!")
        #to get new stellar mass we compute the suriving mass of all previous SSPs and we add that with the new stellar mass
        Ms_next = delta_ms + ms_survs
        MsX_next = np.dot(ms_survs_i,ssps_ini_zfracs) + delta_ms * (MgX/Mg)
        MsZ_next = np.sum(ms_survs_i*ssps_z) + delta_ms * (MgZ/Mg)
        #this is using assumption that metals have been completely mixed
        MgX_next = MgX + mginX_new + X_yields - delta_ms * (MgX/Mg) - wind_loss * (MgX/Mg)
        
        MgZ_next = MgZ + mginZ_new + Z_yield - delta_ms * (MgZ/Mg) - wind_loss * (MgZ/Mg)

       #this is an easy fix to solve the negative metal mass in gas issue
        if MgZ_next < 0:
            MgZ_next = Mg_next * (MgZ/Mg)

        if np.min(MgX_next) < 0:
            MgX_next = (MgX/Mg)*Mg_next

        #ADD A CHECKER THAT CHECKS IF THERE ARE ANY NEGATIVE TERMS BEING MADE..          
        if Mg_next < 0 or Ms_next < 0 or MgZ_next < 0 or MsZ_next < 0 or np.min(MgX_next) < 0 or np.min(MsX_next) <0:
            print("--")
            print("Negative values are being generated. The most likely cause is high wind_loss. Increasing nsteps will fix it. ")
            print("MgZ_next:",MgZ_next)
            print("MgX_next:",np.min(MgX_next))
            print("MgZ_old:",MgZ)
            print("Mg_next:",Mg_next)
            print("Mg_old:",Mg)            
            print("Z_yield:",Z_yield)
            print("mginZ:",mginZ_new)
            print("wind_loss:",wind_loss * (MgZ/Mg))
            print("star_loss:",delta_ms * (MgZ/Mg))
            print("--")

        #add the updated values to the track data!
        mg_tracks = np.concatenate((mg_tracks,[Mg_next]))
        ms_tracks = np.concatenate((ms_tracks,[Ms_next]))
        mgx_tracks = np.vstack((mgx_tracks,MgX_next))
        msx_tracks = np.vstack((msx_tracks,MsX_next))
        msz_tracks = np.concatenate((msz_tracks,[MsZ_next]))
        #note that msx is the TOTAL mass of element X locked in all stars of that galaxy
        mgz_tracks = np.concatenate((mgz_tracks,[MgZ_next]))
        t_tracks = np.concatenate((t_tracks,[time_steps[i+1]]))
        
        #in principle, we could add a non-zero threshold here to decrease computational time
        if delta_ms > 0:
            if verbose == True:
                print(delta_ms)
            #a new SSP was born and so we append its properties to the SSP list
            ssps_ms = np.concatenate((ssps_ms,[delta_ms]))
            ssps_tbirths = np.concatenate((ssps_tbirths,[time_steps[i+1]]))
            ssps_z = np.concatenate( (ssps_z,[MgZ/Mg]) )
            ssps_ini_zfracs = np.vstack((ssps_ini_zfracs,MgX/Mg)) #these are the individual elemental fractions in the gas.
        
        else:
            #no star is being formed and so we do not need anything to the SSP lists 
            pass 


    #time steps [-1] is the final time step
    _, ms_survs_final = compute_all_ms_surv(all_interps_dict["ms_surv"],time_steps[-1],ssps_tbirths, ssps_z, ssps_ms)

    #now we should save this resul

    temp_dict = {"t":t_tracks,"Mg":mg_tracks,"Ms":ms_tracks,"MXg":mgx_tracks,"MZg":mgz_tracks,"MZs":msz_tracks,"MXs":msx_tracks,"Z_IGM":Z_IGM,
                  "ssps_ms":ssps_ms, "ssps_tbirth":ssps_tbirths,"ssps_z":ssps_z,'ssps_ini_zfracs':ssps_ini_zfracs,"ssps_surv_ms":ms_survs_final}
        

    store_chem_results(final_dict=temp_dict,track_path=track_path,iniconf=iniconf,elements_to_track=elements_to_track,final_store_path = final_store_path)

    return 


def compute_ini_metal_rfrac():
    #these are a list of all the main elements
    elements_to_track = np.array(['Al', 'Ar', 'B', 'Be', 'C', 'Ca', 'Cl', 'Co', 'Cr', 'Cu', 'F','Fe', 'Ga', 'Ge', 'H', 'He', 'K', 'Li', 'Mg', 'Mn', 'N', 'Na','Ne', 'Ni', 'O', 'P', 'S', 'Sc', 'Si', 'Ti', 'V', 'Zn'])
    
    basic_solar = solar_abundances()
    getattr(basic_solar, 'Asplund09')()
        
    solar_fractions = []
    elements = np.hstack(basic_solar.all_elements)
    for elei in elements_to_track:
        solar_fractions.append(float(basic_solar.fractions[np.where(elements==elei)]))

    solar_fractions = np.array(solar_fractions)

    metals_to_track = elements_to_track[(elements_to_track != "H") & (elements_to_track != "He")]
    solar_metal_fractions = solar_fractions[(elements_to_track != "H") & (elements_to_track != "He")]

    ini_metal_rfrac = solar_metal_fractions/np.sum(solar_metal_fractions)
    #the above is relative metal adbundances (no H and He)

    return ini_metal_rfrac, metals_to_track


def check_grid(loaded_grid_obj, chem_params):

    all_chem_fields = chem_params._fields

    for i,ci in enumerate(all_chem_fields):
        if ci not in ["element_list","ini_element_ratios","log_zigm_mean","log_zigm_sig","solar_abundances","stochastic_resampling"]:
            if loaded_grid_obj[ci] != chem_params[i]:
                raise ValueError("The entry for %s in ini file does not match the corresponding chemical grid read. Make sure the appropriate pickle file is being used or generate a new chemical grid with this model configuration."%ci)
    return

if __name__ == '__main__':
    #ignore runtime warnings
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    # read in command line arguments
    args = argument_parser().parse_args()
    # read parameters and information from the run config file 
    iniconf = read_config_file(config_file=args.ini)
    cosmo_params,model_params = extract_model_params(iniconf=iniconf,print_params=False)

    chem_params = extract_chem_params(iniconf=iniconf,print_params=True)

    #to run this, I need to know the path where the GRUMPY outputs are stored
    ini_metal_rfrac,metals_to_track = compute_ini_metal_rfrac()

    chem_grid_path = iniconf['chem model']['chem_grids_path']
    check_path_existence(all_paths=[chem_grid_path])

    pickle_name = chem_grid_path + "/" + iniconf['chem model']['chem_grid_pickle']
    #generate or read the chemical grid
    if iniconf['chem model']['generate_chem_grids'] == "False":
        #read the grid file
        with open(pickle_name, 'rb') as f:
            loaded_grid_obj = pickle.load(f)
    else:
        create_chem_pickle(iniconf = iniconf,chem_params = chem_params)
        # create_chem_pickle_TRIAL(iniconf = iniconf,chem_params = chem_params)

        with open(pickle_name, 'rb') as f:
            loaded_grid_obj = pickle.load(f)


    #now we make sure this loaded pickle agrees with the chem model parameters in the ini file
    #this is to make sure that if you intend to change some chemical parameter then it is actually changed
    check_grid(loaded_grid_obj, chem_params)

    #get a list of all the GRUMPY track files
    input_track_path = iniconf['chem setup']['input_track_path']
    final_store_path = iniconf['chem setup']['store_chem_path']
    track_folder = input_track_path + "/track_data/"
    check_path_existence(all_paths=[input_track_path,track_folder,final_store_path])

    all_track_files = glob.glob(track_folder+"*_track.csv")
    #all_track_files is a list of track files of all the galaxies modelled

    print_stage("GRUMPY track files at this location are being used : %s"%track_folder)

    if len(all_track_files) == 0:
        raise ValueError("No GRUMPY track files with format '*_track.csv' are found at %s"%track_folder)
    
    run_chem_parallel = iniconf['chem setup']['run_chemical_parallel']
    nsteps = int(iniconf['chem setup']['nsteps'])
    elements_list = iniconf['chem model']['elements_list']

    #first draw the list of random

    all_element_list = chem_params.element_list
    all_element_list = np.array(all_element_list.split(","))
 
    if run_chem_parallel == "False":

        for fi in tqdm(all_track_files):
      

            inputs_i = {"nsteps":nsteps,"evolve_wind":False,"evolve_star":False,"elements_to_track":all_element_list,
                            "chem_params":chem_params._asdict(),"all_interps_dict":loaded_grid_obj,"ini_metal_rfrac":ini_metal_rfrac,
                            "metals_to_track":metals_to_track,"model_params":model_params._asdict(),"cosmo_params":cosmo_params._asdict(),
                            "verbose":False,"rpd_val":None,"track_path":fi,"iniconf":iniconf,"final_path":final_store_path}

            run_chempy_track(input_stuff = inputs_i) 
            # run_chempy_track_TRIAL(input_stuff = inputs_i)


    else:
        all_inputs = []
        for fi in all_track_files:
            inputs_i = {"nsteps":nsteps,"evolve_wind":False,"evolve_star":False,"elements_to_track":all_element_list,
                            "chem_params":chem_params._asdict(),"all_interps_dict":loaded_grid_obj,"ini_metal_rfrac":ini_metal_rfrac,
                            "metals_to_track":metals_to_track,"model_params":model_params._asdict(),"cosmo_params":cosmo_params._asdict(),
                            "verbose":False,"rpd_val":None,"track_path":fi,"iniconf":iniconf,"final_path":final_store_path}

            all_inputs.append(inputs_i)

        ncores = int(iniconf['run params']['ncores'])
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
            results = list(tqdm(executor.map(run_chempy_track,all_inputs), total = len(all_inputs)))
            # results = list(tqdm(executor.map(run_chempy_track_TRIAL,all_inputs), total = len(all_inputs)))


    #save the ini file in the final store folder
    save_code(final_store_path,args.ini)

    print_stage("The chemical model has finished running! Chemical results are stored at : %s"%final_store_path)
    



