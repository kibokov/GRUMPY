# In this script, we collect the functions and code to generate interpolated SSP yield tables using VICE code

import pickle
import numpy as np
from tqdm import tqdm
import os
import argparse
import glob
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.interpolate import RectBivariateSpline
from configparser import ConfigParser, ExtendedInterpolation
from chem_params import extract_chem_params, mass_fractions_from_bracket_XH
import vice

vice_asplund_solar = 0.014

#### Non-stochastic (IMF-averaged) yield calculations

#functions that interpolate the CC-SNE yield tables
def ccsne_yield_table(element,yield_set = "NKT13", m_upper=40,imf="Kroupa"):
    '''
    CC-SNE yield interpolating function across metallicities for different yield tables
    '''

    if yield_set == "NKT13":
        ##get the different metallicities for that study
        nkt13_zs = np.array([-np.inf, -1.15, -0.54, -0.24, 0.15, 0.55])
        all_yi = []
        
        for zi in nkt13_zs:
            yi = vice.yields.ccsne.fractional(element, MoverH = zi, study = "NKT13",m_upper = m_upper,IMF=imf)[0]
            all_yi.append(yi)
            
        #the function that stores the 1D linear interpolation across metallicity
        interp_func = vice.toolkit.interpolation.interp_scheme_1d( vice_asplund_solar * 10**nkt13_zs, all_yi)
    
    return interp_func, vice_asplund_solar * 10**nkt13_zs, all_yi 

## the basic thing is does it tell me as a function of time, how metals are being ejected into the ISM
## also, how will gross yield be computed, we can use the return fraction?

def get_crf(imf="Kroupa"):
    '''
    In VICE, there already exists a function to compute the cumulative return fraction, however, it only takes a float.
    I want to be able to feed it an array and so interpolation is done here to get the function.
    '''
    ttr = np.linspace(0,14,1000)
    all_crfs = []
    for ti in ttr:
        all_crfs.append( vice.cumulative_return_fraction(age = ti, IMF = imf) )

    #do the linter interpolation now
    crf_func = vice.toolkit.interpolation.interp_scheme_1d( ttr, all_crfs )

    return crf_func


def get_SSP_zmetal(ssp_z,  agb_yield_set="karakas16",ccsne_yield_set = "NKT13",imf="Kroupa",delay=0.15,RIa="plaw",sn1a_norm=2.2e-3,
                  sn1a_yield_set = "seitenzahl13", sn1a_model = "N1"):
    '''
    This function returns the total net metal yield Z. This will return a function that will give the net metal yield. 

    Since VICE does every element between C and Bi, you'll have the vast majority of the metal mass. 
    The alphas dominate the metal mass budget by a landslide, so you can take that as a pretty solid proxy of total Z
    '''

    #get CC-SNE contribution
    ei_ccsne_yields,_,_ = ccsne_yield_table(element, yield_table = ccsne_yield_set,imf=imf)
    vice.yields.ccsne.settings[element] = ei_ccsne_yields(ssp_z)

    #get AGB contribution
    vice.yields.agb.settings[element] = agb_yield_set 

    #get SNe Ia contribution
    vice.yields.sneia.settings[element] = vice.yields.sneia.fractional(element,study = sn1a_yield_set,model = sn1a_model,n=sn1a_norm)




    return 


def get_SSP_vice(element, ssp_z, agb_yield_set="karakas16",ccsne_yield_set = "NKT13",imf="Kroupa",delay=0.15,RIa="plaw",sn1a_norm=2.2e-3,
                  sn1a_yield_set = "seitenzahl13", sn1a_model = "N1"):

    '''

    This function returns the net yield of a specific element (from all 3 channels) at an array of times. Furthermore, these yields are cumulative net yields, 
    that is, at a given time t, this function is computing the total amount of net yield by that time. These are not yields within certain time intervals. To compute that,
    we can just subtract the cumulative yields at two times.

    Inputs:
    ----------
    element = string, the element that we wish to model
    ssp_z = bool, the metallicity of the SSP in absolute units (not solar)

    Outputs:
    ----------
    vice_ssp = lists, lists containing information on the 

    Notes:
    ----------
    In VICE, the ssp object exists for every element separately.

    sn1a yield sets: “seitenzahl13”, “iwamoto99”, “gronow21”

    Some fun things to keep in mind for additional flexibility:
    1) The yields computed by VICE are net yields. The gross yields can be computed by accounting for the return fraction. 
        Eg. 
        rfrac_i = vice.cumulative_return_fraction(age = 7.1, IMF = "kroupa")

    2) RIa is the DTD model for Ia. It is set to power-law right now, but it can be arbitary set by user
    '''

    #CC SNE yield settings
    ei_ccsne_yields,_,_ = ccsne_yield_table(element, yield_table = ccsne_yield_set,imf=imf)
    vice.yields.ccsne.settings[element] = ei_ccsne_yields(ssp_z)

    #AGB yield settings
    vice.yields.agb.settings[element] = agb_yield_set 
    ## can also give it a function of f(m,z) m = ZAMS mass

    ## SNIa yield settings
    vice.yields.sneia.settings[element] = vice.yields.sneia.fractional(element,study = sn1a_yield_set,model = sn1a_model,n=sn1a_norm)
    
    ## putting everything together. Everything is normalized to 1 Msun SSP
    vice_ssp = vice.single_stellar_population(element, mstar = 1,Z=ssp_z, time = 13.5, dt = 0.01,delay=delay,RIa=RIa,IMF=imf)
    #the index 1 list is array of times and index 0 is the array of yields

    #we need to interpolate the cumulative net yield as a function of time
    yields_func = vice.toolkit.interpolation.interp_scheme_1d( vice_ssp[1], vice_ssp[0] )

    return yields_func


def compute_gross_yields(time, yields_func,ini_frac_ei, crf_interp):
    '''
    In this function, we compute the gross yield of an element at any given instant in time

    Inputs:
    ----------
    time: float, time in Gyr at which to compute gross yield
    yields_func: function, function that returns cumulative net yield at a given time
    ini_frac_ei: float, the initial mass fraction of the element in SSP (mass fraction out of total SSP mass)
    crf_interp: function, function that returns CRF at a given time step

    Notes:
    ----------
    gross yield = net_yield + ini_mass_of_element_in_SSP * return_frac
    '''

    #the cumulative return fraction at time 
    crf_ti = crf_interp(time)

    #get the net yield for that element at that time
    net_yield_ti = yields_func(time)

    #this is the cumulative gross yield for a 1 Msun SSP
    gross_yield_ti =  net_yield_ti + crf_ti * ini_frac_ei

    return gross_yield_ti



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
    

    if iniconf["chem setup"]["use_stochastic_sfr_track"] == "True":
        chem_track_path = final_store_path + "/" + track_path_only.replace("_track_stoch_sfr.csv","") + "_chem_evo.csv"
    else:
        chem_track_path = final_store_path + "/" + track_path_only.replace("_track.csv","") + "_chem_evo.csv"

    df.to_csv(chem_track_path,index = False)

    #we will also store the different ssps_ini_zfracs
    df_ssps = pd.DataFrame(ssps_props)
    if iniconf["chem setup"]["use_stochastic_sfr_track"] == "True":
        ssps_track_path = final_store_path + "/" + track_path_only.replace("_track_stoch_sfr.csv","") + "_ssps_prop.csv"
    else:
        ssps_track_path = final_store_path + "/" + track_path_only.replace("_track.csv","") + "_ssps_prop.csv"

    
    df_ssps.to_csv(ssps_track_path,index = False)

    return 


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)



def run_chempy_track_vice(input_stuff):
    '''
    This function runs the chempy SSP model on a GRUMPY track using vice.
    As I will not be evolving the SFR again, I do not need to keep track of the total metal content Z
    However, I would need to know the Z to get accurate yield of SSP as yields depend on it. Maybe it is an okay enough approximation to 
    just use the Z computed from IRA

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
    iniconf = input_stuff["iniconf"]
    final_store_path = input_stuff["final_path"]

    chem_params = convert(chem_params)
    model_params = convert(model_params)
    cosmo_params = convert(cosmo_params)

    df = pd.read_csv(track_path)

    Zsun = model_params.Zsun

    #loading the tracks
    mgin_cumu = np.array(df["Mgin"])
    mgout_cumu = np.array(df["Mgout"])

    mg_evo = np.array(df["Mg"])
    ms_evo = np.array(df["Ms"])
    tt = np.array(df["t"])

    if chem_params.flexible_igm == "False":
        if chem_params.log_zigm_sig == 0:
            Z_IGM = Zsun * 10**(chem_params.log_zigm_mean)
        else:
            #this is assuming that the IGM metallicity is the same for the entire galaxy's evolution.
            #and is not fluctuating over time
            Z_IGM = (10**np.random.normal(chem_params.log_zigm_mean, chem_params.log_zigm_sig))*Zsun

        def ZIGM_floor(time):
            '''
            This is a function that just returns this constant value as IGM floor
            '''
            return Z_IGM

    if chem_params.flexible_igm == "True":
        from flexible_chem_functions import ZIGM_floor
        Z_IGM = ZIGM_floor(tt[0])

    #we convert them to linear interpolation objects
    mgin_spl = interpolate.interp1d(tt, mgin_cumu)
    mg_total_spl = interpolate.interp1d(tt, mg_evo )
    if evolve_star == False:
        ms_spl = interpolate.interp1d(tt,ms_evo)
    if evolve_wind == False:
        mgout_spl = interpolate.interp1d(tt, mgout_cumu)

    
    tstart = tt[0]
    ms_start = 10 #this is the initial stellar mass within the halo
    mg_start = mg_evo[0] #this is the initial gas mass in the halo. We take this from our pre-evolved track
    mg_z_start = mg_start * Z_IGM
    #the initial abundances of elements X in gas phase
    mg_X = np.zeros_like(elements_to_track,dtype = float) + -99
    
    #we initially assume that the relative metal fractions/abundances are the same as solar
    #if I want to focus on [Fe/H] < -3 stars then this assumption is less valid.
    #the same assumption is made for IGM gas as well
    #in the future using more accurate Pop III yields (if we know) would be good
    
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
        # user-defined [X/H] in dex (one per metal, same order as elements_list)
        bracket_XH_list = np.array(ini_element_ratios.split(","), dtype=float)
        f_array = mass_fractions_from_bracket_XH(
            elements_to_track, Z_IGM, bracket_XH_list, chem_params.solar_abundance_name
        )
        mg_X = mg_start * f_array

    if np.min(mg_X) < 0:
        print(mg_X)
        print(elements_to_track)
        raise ValueError("Some initial mass of element is negative! Check initial mass fraction values.")
        
    #the time array we will evaluating the SSPs and integrating to compute the time evolution...
    time_steps = np.linspace(tt[0],tt[-1],nsteps)
    
    #these are the lists where we will store the properties of the SSP
    ssps_tbirths = [] #the formation time of SSP
    ssps_ms = [] #the mass of this SSP
    ssps_ini_zfracs = [] #the SSP initialization metal FRACTIONS (including H,He) for all the metals being tracked. 

    #we append the initial SSP quantities here.
    #these are the values of the first SSP
    ssps_tbirths.append(time_steps[0])
    ssps_ms.append(ms_start)
    ssps_ini_zfracs.append( mg_X/mg_start)

    ssps_tbirths = np.array(ssps_tbirths)
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
    zigm_tracks = np.array([Z_IGM])
    
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

        #if IGM floor is not flexible, this returns same value every time
        Z_IGM_new = ZIGM_floor(ti)
        #as Z_IGM changes in flexible case, we will also need to update Z_X_IGM. 
        #This would be simply be a proportional increase in Z_X_IGM (assuming the relative metal fracs stay the same which we are assuming for simplicity)
        Z_X_IGM = Z_X_IGM * (Z_IGM_new/zigm_tracks[-1]) 
        #zigm_tracks[-1] is the igm metallicity in the previous step
        #Z_IGM_new is the igm metallicity in the current step
        #we update the value of Z_X_IGM from its previous step. In case where IGM floor is not flexible, Z_X_IGM will not be changing 

        #load the current values 
        Mg = mg_tracks[-1]
        Ms = ms_tracks[-1]
        MgX = mgx_tracks[-1]
        MgZ = mgz_tracks[-1]

        if evolve_star == False:
            #we only need to know SFR from the Ms track
            sfr = (ms_spl(time_steps[i+1]) - ms_spl(time_steps[i]))/dt
            delta_ms = sfr * dt / Rloss1 #this is the new stellar mass formed
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
        # mginZ_new = mgin_new * Z_IGM_new

        #total metal mass injected into ISM
        #CHECK BELOW STEP
        # Z_yield = compute_yZ_gross(all_interps_dict["yZ_gross"],time_steps[i], time_steps[i+1], ssps_tbirths,ssps_z, ssps_ms)
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
            
            #If yields are flexible, the change will be implemented up here
            
            X_yields.append(xf_yields)
            
        X_yields = np.array(X_yields)
 
        #WE UPDATE VALUES FOR NEXT TIME STEP NOW
        Mg_next = Mg + mgin_new + ssp_feedbacks  - delta_ms - wind_loss 
        if Mg_next < 0:
            #if it is zero, what if we make it the true value
            Mg_next = mg_total_spl(time_steps[i+1])
            # Mg_next = 10
            # print(Mg, mgin_new, ssp_feedbacks, delta_ms, wind_loss)
            # print(Mg_next)
            # raise ValueError("UH OH! NEGATIVE GAS MASS!! SOMETHING IS WRONG!!!")
        
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
        
        #does the above fix make sense though? The above fix means that the metallicity of gas before and after is 
        #the same.

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
        zigm_tracks = np.concatenate((zigm_tracks,[Z_IGM_new]) )

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


    temp_dict = {"t":t_tracks,"Mg":mg_tracks,"Ms":ms_tracks,"MXg":mgx_tracks,"MZg":mgz_tracks,"MZs":msz_tracks,"MXs":msx_tracks,"Z_IGM":zigm_tracks,
                  "ssps_ms":ssps_ms, "ssps_tbirth":ssps_tbirths,"ssps_z":ssps_z,'ssps_ini_zfracs':ssps_ini_zfracs,"ssps_surv_ms":ms_survs_final}
        
    store_chem_results(final_dict=temp_dict,track_path=track_path,iniconf=iniconf,elements_to_track=elements_to_track,final_store_path = final_store_path)

    return 
