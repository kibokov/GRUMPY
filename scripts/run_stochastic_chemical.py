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
from gen_chem_pickle import create_chem_pickle
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from chem_params import extract_chem_params
#this is only used if evolve_star = True (we usually use False anyways)
#we can use zspl here, but this was easiest during testing. Will change this later
# from colossus.cosmology import cosmology
# my_cosmo = {'flat': True, 'H0': 67.11, 'Om0': 0.32, 'Ob0': 0.05, 'sigma8': 0.83, 'ns': 0.96}
# # set my_cosmo to be the current cosmology
# cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)  


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


def check_stochastic_grid(stoc_grids, chem_params):
    #let us consider a random stoc grid and also the analytical grid to be sure

    rnd_stoc_grid = stoc_grids["M3_I2"]
    analytical_grid = stoc_grids["analytical"]

    all_chem_fields = chem_params._fields

    for i,ci in enumerate(all_chem_fields):
        if ci not in ["element_list","ini_element_ratios","log_zigm_mean","log_zigm_sig","solar_abundances","stochastic_resampling"]:
            if rnd_stoc_grid[ci] != chem_params[i]:
                raise ValueError("The entry for %s in ini file does not match the corresponding chemical grid read. Make sure the appropriate pickle file is being used or generate a new chemical grid with this model configuration."%ci)

            if analytical_grid[ci] != chem_params[i]:
                raise ValueError("The entry for %s in ini file does not match the corresponding chemical grid read. Make sure the appropriate pickle file is being used or generate a new chemical grid with this model configuration."%ci)

    return



def load_all_stoc_grids(iniconf = None):
    '''
    This function will load all the pickles into a master pickle dictionary that is indexed by its mass and iteration number
    '''
    #the template for the stochastic grids
    pickle_folder = iniconf['chem model']['chem_grids_path'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "/"
    # pickle_folder = pickle_folder + iniconf['chem model']['chem_grid_pickle'] + "_M*_I*.pickle"
    all_sto_pickles = glob.glob(pickle_folder + iniconf['chem model']['chem_grid_pickle'] + "_M*_I*.pickle")

    all_grids = {}

    for pi in all_sto_pickles:
        with open(pi, 'rb') as f:
            grid_stoc_pi = pickle.load(f)

        #isolating the name of the pickle file from its full file path
        #the below name is like ["M0_I23"]
        pi = pi.replace(pickle_folder,"").replace(iniconf['chem model']['chem_grid_pickle']+"_","").replace(".pickle","") 
        
        # all_inds = pi.split("_")
        # m_ind = int(all_inds[0].replace("M",""))  #reflective of the SSP mass
        # i_ind = int(all_inds[1].replace("I",""))#reflective the iteration (among 50) at a fixed SSP mass
        #the dictionary key is of hte format M%d_I%d
        all_grids[pi] = grid_stoc_pi

    #load the analytical grid
    pickle_analytical = iniconf['chem model']['chem_grids_path'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "_analytical.pickle"
    with open(pickle_analytical, 'rb') as f:
        grid_analytical = pickle.load(f)

    all_grids["analytical"] = grid_analytical

    return all_grids


def compute_M_index(ssp_ms):
    '''
    This function returns the index corresponding to M in the stochastic grid naming scheme
    '''

    ssp_mass_grid = np.array([1,1e1,5e1,1e2,5e2,1e3,5e3,1e4])

    diffs = np.abs(ssp_ms - ssp_mass_grid)

    m_ind = np.argmin(diffs)

    return m_ind


def generate_master_sto(stoc_grids=None,ssps_tbirths=None,ssps_z=None,
                        ssps_ms=None,master_dict = None,t_master = None,
                        ssps_ini_zfracs=None,org_fracs=None,elements_to_track=None,org_elements=None,
                        all_grid_nums=None,was_ssp_made=None):
    '''
    stoc_grids = dict, dictionary of all the grids for different SSP masses and also analytical case
    ssps_tbirths = list, time of births of SSPs
    ssps_z = list, metallicities of SSPs at time of birth 
    ssps_ms = list, masses of SSPs at formation
    master_sto_obj = dict, dictionary that contains the interpolating functions for evolutions of yZ, mass loss etc. 
    It is initially None (for first SSP) as this object has not been constructed yet
    t_master = list, time steps at which the chemical evolution is being computed at 

    I want to also return a list of all the grids used like the mass index and the iteration number

    '''
    
    if master_dict== None:
        t_ages = t_master - ssps_tbirths[0]
        #the first value of this should be zero
        if t_ages[0] != 0:
            print("SOME ISSUE HERE.")
            
        #if len of ssps lists is 1, then this is the first time this is being computed
        #the initial mass is 10 hence always going to be stochastic
        m_ind = compute_M_index(ssps_ms[0])
        #generate a random number 
        i_ind = np.random.randint(50)
        grid_name = "M%d_I%d"%(m_ind,i_ind)
        stoc_0 = stoc_grids[grid_name]
        #stoc_i is the dict that will be used
        t_ages[t_ages == 0] = 1e-3
        logz_val = np.log10(ssps_z)*np.ones(shape = len(t_ages))
        #within this dict, we want the grids of yZ_gross, ms_surv, ms_feedback
        #we will also need individual elemental grids 
        master_dict_new = {}
        all_keys = ["yZ_gross","ms_surv","ms_feedback"]
        for ki in all_keys:
            ki_0 = stoc_0[ki]
            #both inputs are in log space
            all_y_vals = ki_0(logz_val,np.log10(t_ages),grid = False)
            #do the interpolation now
            #we need to normalize all values to the mass of the SSP as grid values are for 1M SSP
            ki_interp = interpolate.interp1d(t_master,all_y_vals * ssps_ms[0])
            master_dict_new[ki] = ki_interp

        #now we compute the gross yields for each element being tracked
        for i,ei in enumerate(elements_to_track):
            org_fracs_i = org_fracs[org_elements==ei]
            ini_fracs_i = ssps_ini_zfracs[0][i]

            y_net_ei = stoc_0[ei+"_net"]
            y_delta_ei = stoc_0[ei + "_diff"]

            y_net_ei_vals = y_net_ei(logz_val, np.log10(t_ages),grid = False )
            y_delta_ei_vals = y_delta_ei(logz_val, np.log10(t_ages),grid=False)

            y_gross_ei = y_net_ei_vals + (ini_fracs_i/org_fracs_i) * y_delta_ei_vals
            #we scale the elemental yield by the SSP mass as grid values are for 1M SSP
            ei_interp = interpolate.interp1d(t_master,y_gross_ei * ssps_ms[0])

            master_dict_new[ei+"_gross"] = ei_interp

        all_grid_nums.append([ m_ind,i_ind ] )

        return master_dict_new,all_grid_nums
        
        
    else:

        #what I am concerned about here is that say I made a SSP in some time step
        #and in next time step I made no SSP
        #however when this function is fun, it does not know that? and just remakes a new SSP with that 
        #last mass? If no new SSP...we need to keep using the same interpolation

        #this is not the first time making master sto and so teh object already exists
        #we will just modify it by the latest SSP prop


        if was_ssp_made[-1] == 1:
            #a new ssp was made!
            if ssps_ms[-1] <= 5e4:
                m_ind = compute_M_index(ssps_ms[-1])
                i_ind = np.random.randint(50)
                grid_name = "M%d_I%d"%(m_ind,i_ind)
                stoc_i = stoc_grids[grid_name]
                all_grid_nums.append( [m_ind,i_ind] )

            else:
                #we use the analytical grid
                stoc_i = stoc_grids["analytical"]
                all_grid_nums.append("analytical")

            
            ### WHAT DO I DO HERE WHEN THERE IS NO NEW SSP??? 
            ### I NEED TO NOT ADD ANYTHING TO THIS MASTER LIST WHEN NO NEW SSPS ARE BEING ADDED


            all_keys = ["yZ_gross","ms_surv","ms_feedback"]

            t_ages = t_master - ssps_tbirths[-1]
            t_age_pass = t_ages[t_ages>=0]
            t_age_pass[t_age_pass == 0] = 1e-3

            logz_val = np.log10(ssps_z[-1])*np.ones(shape = len(t_ages[t_ages >= 0]))

            for ki in all_keys:
                ki_i = stoc_i[ki]
                #ki_master is the y values at t_master using the master interpolator made till now
                #we will update its values with the most recent SSP values
                ki_master = master_dict[ki](t_master)

                all_y_vals = ki_i(logz_val,np.log10(t_age_pass),grid = False)

                all_y_vals = np.concatenate((t_ages[t_ages < 0]*0,all_y_vals))
                #we update the previous values with the new SSP values
                #we scale the new values as grid values are for 1M SSP
                ki_master_new = ki_master + all_y_vals * ssps_ms[-1]
                ki_interp_new = interpolate.interp1d(t_master,ki_master_new )
                #we update the values of the dictionary
                master_dict[ki] = ki_interp_new

            #consider the latest ssps ini fracs
            ssps_ini_zfracs_i = ssps_ini_zfracs[-1]

            #now we have to loop over each element
            for i,ei in enumerate(elements_to_track):
                org_fracs_i = org_fracs[org_elements==ei]
                ini_fracs_i = ssps_ini_zfracs_i[i]

                y_net_ei = stoc_i[ei+"_net"]
                y_delta_ei = stoc_i[ei + "_diff"]

                y_net_ei_vals = y_net_ei(logz_val, np.log10(t_age_pass),grid = False )
                y_delta_ei_vals = y_delta_ei(logz_val, np.log10(t_age_pass),grid=False)

                y_gross_ei = y_net_ei_vals + (ini_fracs_i/org_fracs_i) * y_delta_ei_vals
                #we add some zeroes before hand to indicate no contribution from this SSP as it was not born then
                #we do this so it is same size as t_master. We filter ages to be >= 0 earlier hence.
                y_gross_ei_tot = np.concatenate((t_ages[t_ages < 0]*0,y_gross_ei))
                
                #these are the original gross yields evaluated at t_master
                y_gross_ei_master = master_dict[ei+"_gross"](t_master)
                #we combine the old values with values for the new SSP
                y_gross_ei_new =  y_gross_ei_master + y_gross_ei_tot * ssps_ms[-1]

                ei_interp_new = interpolate.interp1d(t_master,y_gross_ei_new)
                #we update the values of the dictionary
                master_dict[ei + "_gross"] = ei_interp_new

            #in general should make sure that the interpolating function fed to master dict should 
            # be fed time in units of age of universe
            #In contrast, the time fed to the 2D grids which we read is in age of SSP units.         
            return master_dict, all_grid_nums

        if was_ssp_made[-1] == 0:
            #a new ssp was not made

            return master_dict, all_grid_nums


def store_chem_results(final_dict=None,elements_to_track=None,track_path=None,iniconf=None,resampling_ind = None,final_store_path=None):
    '''
    function that will save the chemical model results to existing file structures or make new ones

    In case stochastic resampling is enabled, we can label the different iterations by ite0, ite1, etc. 
    '''

    sub_dict = {"t":final_dict["t"],"Ms":final_dict["Ms"],"Mg":final_dict["Mg"],
                "MZg":final_dict["MZg"]}
                # , "MZs":final_dict["MZs"]

    # MXs = final_dict['MXs']
    MXg = final_dict['MXg']

    ssps_props = {}
    ssps_ini_fracs = final_dict["ssps_ini_zfracs"]
    ssps_ms = final_dict["ssps_ms"]
    ssps_tbirth = final_dict["ssps_tbirth"]
    ssps_z = final_dict["ssps_z"]
    ssps_ms_surv = final_dict["ssps_surv_ms"]

    for i,ei in enumerate(elements_to_track):
        # sub_dict["Ms_" + ei] = MXs[:,i]
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
    chem_track_path = final_store_path + "/" + track_path_only.replace("_track.csv","") + "_chem_evo" + "_ite%d"%resampling_ind +  ".csv"

    # chem_track_path = track_path.replace("_track.csv","") + "_chem_evo" + "_ite%d"%resampling_ind +  ".csv"

    df.to_csv(chem_track_path,index = False)

    #we will also store the different ssps_ini_zfracs
    df_ssps = pd.DataFrame(ssps_props)
    ssps_track_path = final_store_path + "/" + track_path_only.replace("_track.csv","") + "_ssps_prop" + "_ite%d"%resampling_ind + ".csv"

    # ssps_track_path = track_path.replace("_track.csv","") + "_ssps_prop" + "_ite%d"%resampling_ind + ".csv"
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
    ini_metal_rfrac = input_stuff["ini_metal_rfrac"]
    metals_to_track = input_stuff["metals_to_track"]
    model_params = input_stuff["model_params"]
    cosmo_params = input_stuff["cosmo_params"]
    chem_params = input_stuff["chem_params"]
    verbose = input_stuff["verbose"]
    rpd_val = input_stuff["rpd_val"]
    iniconf = input_stuff["iniconf"]
    stoc_grids = input_stuff["stoc_grids"]
    resample_ind = input_stuff["resample_ind"]
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
    ms_start = float(iniconf['model params']['mstar_ini']) #this is the initial stellar mass within the halo
    mg_start = mg_evo[0] #this is the initial gas mass in the halo. We take this from our pre-evolved track
    #Z_IGM is drawn from a distribution, and is fed into this function
    mg_z_start = mg_start * Z_IGM
    
    mg_X = np.zeros_like(elements_to_track,dtype = float) + -99
    
    #we can initially assume that the relative metal fractions/abundances are the same as solar
    #if I want to focus on [Fe/H] < -3 stars then this assumption is less valid.
    #the same assumption is made for IGM gas as well
    #another approach I can do is to assign the yields of CC SNe at low metallicity
    
    ini_element_ratios = iniconf['chem model']['ini_element_ratios']

    if ini_element_ratios == "solar":

        for ei in elements_to_track:
            if ei == "H":
                mg_X[ elements_to_track == "H" ] = 0.75*(mg_start - mg_z_start)
            elif ei == "He":
                mg_X[ elements_to_track == "He" ] = 0.25*(mg_start - mg_z_start)
            else:
                #this for all the rest of the metals 
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
                mg_X[elements_to_track == ei] = mg_z_start * ini_metal_ratios[j - 2]

    if np.min(mg_X) < 0:
        print(mg_X)
        print(elements_to_track)
        print("SOME VALUE WAS NOT UPDATED!")
        
    #the time array we will evaluating the SSPs and integrating to compute the time evolution...
    time_steps = np.linspace(tt[0],tt[-1],nsteps)
    #converting this to redshifts. This is needed to be fed to MH2 function if being used
    if evolve_star == True:
        zred_steps = cosmo.age(time_steps,inverse=True)
    
    #these are the lists where we will store the properties of the SSP
    ssps_tbirths = [] #the formation time of SSP
    ssps_z = [] #the SSP formation metallicity Z (all metal content)
    ssps_ms = [] #the mass of this SSP
    ssps_ini_zfracs = [] #the SSP initialization metal FRACTIONS (including H,He) for all the metals being tracked. 
    #list that stores info on whether a new ssp was made or not
    was_ssp_made = []
    all_grid_nums = []


    #we append the initial SSP quantities here.
    #these are the values of the first SSP
    ssps_tbirths.append(time_steps[0])
    ssps_z.append(mg_z_start/mg_start) #assume same metal properties as gas
    ssps_ms.append(ms_start)
    ssps_ini_zfracs.append( mg_X/mg_start )
    was_ssp_made.append(1)

    ssps_tbirths = np.array(ssps_tbirths)
    ssps_z = np.array(ssps_z)
    ssps_ms = np.array(ssps_ms)
    ssps_ini_zfracs = np.array(ssps_ini_zfracs)

    #these are the individual elemental metallicities in the IGM matter
    Z_X_IGM = mg_X / mg_start
            
    #the initial elemental abundance locked inside stars
    # ms_x_start = ms_start * Z_X_IGM
    # ms_z_start = ms_start * Z_IGM
    #these are the evolution tracks that we will be populate
    mg_tracks = np.array([mg_start]) 
    ms_tracks = np.array([ms_start])
    # msx_tracks = np.array([ms_x_start])
    # msz_tracks = np.array([ms_z_start])
    mgz_tracks = np.array([mg_z_start]) #this is the initial total metal mass
    mgx_tracks = np.array([mg_X])
    
    t_tracks = np.array([tstart])
    
    #we do the euler forward integration now ... 
    
    #some relevant stuff needed in SSP yield computation
    #we just pick a random grid to get this info
    org_elements = np.array(stoc_grids["M0_I10"]["elements"])
    org_fracs = np.array(stoc_grids["M0_I10"]["solar_fractions"]) #the list of elemental fractions used in 2d interpolation
    
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
            print(delta_ms)

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

        #for all the existing SSPs, we want to compute the various yields and properties 

        

        if i == 0:
            master_dict = None

        master_dict,all_grid_nums = generate_master_sto(stoc_grids=stoc_grids,ssps_tbirths=ssps_tbirths,ssps_z=ssps_z,
                        ssps_ms=ssps_ms,master_dict = master_dict,t_master = time_steps,
                        ssps_ini_zfracs=ssps_ini_zfracs,org_fracs=org_fracs,elements_to_track=elements_to_track,
                        org_elements=org_elements,all_grid_nums=all_grid_nums,was_ssp_made=was_ssp_made)


        Z_yield = master_dict["yZ_gross"](time_steps[i+1]) - master_dict["yZ_gross"](time_steps[i])
        ms_survs = master_dict["ms_surv"](time_steps[i])
        ssp_feedbacks = master_dict["ms_feedback"](time_steps[i+1]) - master_dict["ms_feedback"](time_steps[i])

        X_yields = []
        for xf in elements_to_track:
            temp_xf = master_dict[xf+"_gross"](time_steps[i+1]) - master_dict[xf+"_gross"](time_steps[i])
            X_yields.append(temp_xf)

    
        #NEED TO MAKE SURE I AM SCALING THE YIELDS BY STELLAR MASS!

        # Z_yield = compute_yZ_gross(all_interps_dict["yZ_gross"],time_steps[i], time_steps[i+1], ssps_tbirths,ssps_z, ssps_ms)
        # ms_survs, ms_survs_i = compute_all_ms_surv(all_interps_dict["ms_surv"],time_steps[i],ssps_tbirths, ssps_z, ssps_ms)
        # ssp_feedbacks = compute_all_feedback_ms(all_interps_dict["ms_feedback"],time_steps[i], time_steps[i+1], ssps_tbirths,ssps_z, ssps_ms )

        #X yields are the amount of element X injected into ISM. We thus want the gross yields...
#         X_yields = []
                
#         #looping over each element...
#         for f,xf in enumerate(elements_to_track):
#             org_fracs_f = org_fracs[org_elements == xf]
#             ind_f = int(np.where(elements_to_track == xf)[0])
#             ssps_ini_zfracs_f = ssps_ini_zfracs[:,ind_f]
#             #we extract the relevant column that corresponds to the element under inspection here...
# #             compute_gross_yield(yi_net,yi_diff,t_low, t_high,ssps_tbirths, ssps_z, ssps_ms, ini_frac,org_frac)
#             xf_yields = compute_gross_yield(all_interps_dict[xf+"_net"],all_interps_dict[xf+"_diff"],time_steps[i], time_steps[i+1],ssps_tbirths, ssps_z, ssps_ms, ssps_ini_zfracs_f,org_fracs_f)
#             X_yields.append(xf_yields)
            
        X_yields = np.array(X_yields)
 
        #WE UPDATE VALUES FOR NEXT TIME STEP NOW
        Mg_next = Mg + mgin_new + ssp_feedbacks  - delta_ms - wind_loss 
        if Mg_next < 0:
            print("NEGATIVE GAS MASS!! SOMETHING IS WRONG!!!")
        #to get new stellar mass we compute the suriving mass of all previous SSPs and we add that with the new stellar mass
        Ms_next = delta_ms + ms_survs
        # MsX_next = np.dot(ms_survs_i,ssps_ini_zfracs) + delta_ms * (MgX/Mg)
        # MsZ_next = np.sum(ms_survs_i*ssps_z) + delta_ms * (MgZ/Mg)
        #this is using assumption that metals have been completely mixed
        MgX_next = MgX + mginX_new + X_yields - delta_ms * (MgX/Mg) - wind_loss * (MgX/Mg)
        
        MgZ_next = MgZ + mginZ_new + Z_yield - delta_ms * (MgZ/Mg) - wind_loss * (MgZ/Mg)


        #this is an easy fix to solve the negative metal mass in gas issue
        if MgZ_next < 0:
            MgZ_next = Mg_next * (MgZ/Mg)
    
        if np.min(MgX_next) < 0:
            MgX_next = (MgX/Mg)*Mg_next

        #ADD A CHECKER THAT CHECKS IF THERE ARE ANY NEGATIVE TERMS BEING MADE..          
        if Mg_next < 0 or Ms_next < 0 or MgZ_next < 0  or np.min(MgX_next) < 0: # or np.min(MsX_next) <0 or MsZ_next < 0:
            print("--")
            print("Negative values are being generated. The most likely cause is high wind_loss. Increasing nsteps will fix it. The reason for this is that for computing wind loss, we interpolate the wind loss from GRUMPY tracks - if the wind loss sharply increases then not using fine enough time steps might cause this issue.")
            print("MgZ_next:",MgZ_next)
            print("MgX_next:",MgX_next)
            print("Ms_next:",Ms_next)
            print("Mg_next:",Mg_next)
            print("MgZ_old:",MgZ)
            print("Z_yield:",Z_yield)
            print("mginZ:",mginZ_new)
            print("wind_loss:",wind_loss * (MgZ/Mg))
            print("star_loss:",delta_ms * (MgZ/Mg))
            print("--")

        #add the updated values to the track data!
        mg_tracks = np.concatenate((mg_tracks,[Mg_next]))
        ms_tracks = np.concatenate((ms_tracks,[Ms_next]))
        mgx_tracks = np.vstack((mgx_tracks,MgX_next))
        # msx_tracks = np.vstack((msx_tracks,MsX_next))
        # msz_tracks = np.concatenate((msz_tracks,[MsZ_next]))
        #note that msx is the TOTAL mass of element X locked in all stars of that galaxy
        mgz_tracks = np.concatenate((mgz_tracks,[MgZ_next]))
        t_tracks = np.concatenate((t_tracks,[time_steps[i+1]]))
        
        #in principle, we could add a non-zero threshold here to decrease computational time
        # print(ssps_ms)
        print("---")

        if delta_ms > 0:
            if verbose == True:
                print(delta_ms)
            #a new SSP was born and so we append its properties to the SSP list
            ssps_ms = np.concatenate((ssps_ms,[delta_ms]))
            ssps_tbirths = np.concatenate((ssps_tbirths,[time_steps[i+1]]))
            ssps_z = np.concatenate( (ssps_z,[MgZ/Mg]) )
            ssps_ini_zfracs = np.vstack((ssps_ini_zfracs,MgX/Mg)) #these are the individual elemental fractions in the gas.
            was_ssp_made.append(1)
        else:
            #no star is being formed and so we do not need anything to the SSP lists 
            was_ssp_made.append(0)
            pass 

    #at the very end as we have the ssps_tbirths, ssps_ms, ssps_z we can compute the surviving SSP mass
    # if len(ssps_ms)-1 != len(all_grid_nums):
    #     print(len(ssps_ms)-1,len(all_grid_nums),len(was_ssp_made)-1)
    #     print("WE HAVE AN ISSUE! SSPS ms list does not match in length with grid num list")
    # else:
    #     print("We DO MATCH!",len(ssps_ms),len(all_grid_nums),len(was_ssp_made))

    ms_survs_final = []

    for i,gi in enumerate(all_grid_nums):
        ssp_tbirth_i = ssps_tbirths[i]
        ssp_age = time_steps[-1] - ssp_tbirth_i
        ssp_ms_i = ssps_ms[i]
        ssp_z_i = ssps_z[i]

        if gi != "analytical":
            m_ind = gi[0]
            i_ind = gi[1]
            all_grid_i = stoc_grids["M%d_I%d"%(m_ind,i_ind)]
            #now we compute the surviving mass
            ms_surv_interp = all_grid_i["ms_surv"]
            ms_survs = ms_surv_interp(np.log10(ssp_z_i),np.log10(ssp_age),grid = False)
            ms_survs_final.append(ms_survs*ssp_ms_i)

        else:
            #this SSP was analytical and so we will be using the appropriate grid
            grid_ana = stoc_grids["analytical"] 
            ms_surv_interp = grid_ana["ms_surv"]
            #now we compute the surviving mass
            ms_survs = ms_surv_interp(np.log10(ssp_z_i),np.log10(ssp_age),grid = False)
            ms_survs_final.append(ms_survs*ssp_ms_i)


    #we add the ssp mass to surviving mass it was just born and so it all survives
    #we will only need to do this if a SSP was formed at the very last time step
    if len(ssps_ms) != len(ms_survs_final):
        ms_survs_final.append(ssps_ms[-1])

    # print(len(ms_survs_final),len(ssps_ms))

    # temp_dict = {"t":t_tracks,"Mg":mg_tracks,"Ms":ms_tracks,"MXg":mgx_tracks,"MZg":mgz_tracks,"MZs":msz_tracks,"MXs":msx_tracks,"Z_IGM":Z_IGM,
    #               "ssps_ms":ssps_ms, "ssps_tbirth":ssps_tbirths,"ssps_z":ssps_z,'ssps_ini_zfracs':ssps_ini_zfracs}

    temp_dict = {"t":t_tracks,"Mg":mg_tracks,"Ms":ms_tracks,"MXg":mgx_tracks,"MZg":mgz_tracks,"Z_IGM":Z_IGM,
                  "ssps_ms":ssps_ms, "ssps_tbirth":ssps_tbirths,"ssps_z":ssps_z,'ssps_ini_zfracs':ssps_ini_zfracs,'ssps_surv_ms':ms_survs_final}
        

    store_chem_results(final_dict=temp_dict,track_path=track_path,iniconf=iniconf,elements_to_track=elements_to_track,resampling_ind = resample_ind,final_store_path = final_store_path)

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


    #first read all the stochastic grids
    stoc_grids = load_all_stoc_grids(iniconf = iniconf)

    #now we make sure this loaded pickle agrees with the chem model parameters in the ini file
    #this is to make sure that if you intend to change some chemical parameter then it is actually changed
    check_stochastic_grid(stoc_grids, chem_params)

    #get a list of all the GRUMPY track files
    input_track_path = iniconf['chem setup']['input_track_path']
    final_store_path = iniconf['chem setup']['store_chem_path']

    track_folder = input_track_path + "/track_data/"
    check_path_existence(all_paths = [input_track_path,track_folder,final_store_path])
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

    resampling_num = 1

    if iniconf['chem model']["stochastic_resampling"] != "None" or iniconf['chem model']["stochastic_resampling"] != "none":
        resampling_num = int(iniconf['chem model']["stochastic_resampling"])

    if run_chem_parallel == "False":

        for fi in tqdm(all_track_files):
      
            for j in range(resampling_num):

                inputs_i = {"nsteps":nsteps,"evolve_wind":False,"evolve_star":False,"elements_to_track":all_element_list,
                                "chem_params":chem_params._asdict(),"ini_metal_rfrac":ini_metal_rfrac,
                                "metals_to_track":metals_to_track,"model_params":model_params._asdict(),"cosmo_params":cosmo_params._asdict(),
                                "verbose":False,"rpd_val":None,"track_path":fi,"iniconf":iniconf,"stoc_grids":stoc_grids,"resample_ind":j,"final_path":final_store_path}

                run_chempy_track(input_stuff = inputs_i)

    else:
        all_inputs = []
        for fi in all_track_files:
            for j in range(resampling_num):

                inputs_i = {"nsteps":nsteps,"evolve_wind":False,"evolve_star":False,"elements_to_track":all_element_list,
                                "chem_params":chem_params._asdict(),"ini_metal_rfrac":ini_metal_rfrac,
                                "metals_to_track":metals_to_track,"model_params":model_params._asdict(),"cosmo_params":cosmo_params._asdict(),
                                "verbose":False,"rpd_val":None,"track_path":fi,"iniconf":iniconf,"stoc_grids":stoc_grids,"resample_ind":j,"final_path":final_store_path}

                all_inputs.append(inputs_i)

        ncores = int(iniconf['run params']['ncores'])
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
            results = list(tqdm(executor.map(run_chempy_track,all_inputs), total = len(all_inputs)))


    print_stage("The chemical model has finished running! Chemical results are stored at : %s"%final_store_path)
    
