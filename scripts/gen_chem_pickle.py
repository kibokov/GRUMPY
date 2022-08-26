

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
import concurrent.futures
from collections import namedtuple
from scipy.interpolate import RectBivariateSpline
from configparser import ConfigParser, ExtendedInterpolation
from Chempy.imf import IMF
from Chempy.solar_abundance import solar_abundances
from Chempy.parameter import ModelParameters
from Chempy.yields import SN2_feedback, AGB_feedback, SN1a_feedback 
from Chempy.weighted_yield import SSP
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




def get_SSP(Z,net_yield = True,time_steps = np.logspace(-3,np.log10(15),50),chem_params = None,
            stochastic = False,ssp_mass = 1):
    '''
    Inputs:
    ----------
    Z = float, the metallicity of the SSP (not normed to solar value)
    net_yield = bool, if True, only net yields are computed. If False, gross yields are computed
    time_steps = array, the array of times at which we want ChemPy to compute SSP feedback
    chem_params = contains all the SSP parameters entered in the .ini file
    stochastic = bool, if we want stochastic sampling of IMF and SN Ia events
    ssp_mass = float, if stochastic is True, then we want the mass of SSP
    
    Outputs:
    ----------
    time_steps = array, the array of times at which ChemPy has computed SSP feedback
    basic_ssp = ChemPy object, contains tables of all elemental yields etc. 
    elements_to_trace = array, array of the chemical elements that will be tracked by ChemPy
    solar_fractions = array, the solar abundances of the elements tracked by ChemPy

    Notes:
    There is a weird numerical issue regarding time spacing the number of SN IA events. np.logspace(-3,np.log10(15),50) appears to be fine. 
    Using upper bound of 14 causes issues. Linear spacing does not cause issues. Have checked that with 50 spacings it works fine
    
    '''
    
    a = ModelParameters()

    basic_solar = solar_abundances()
    getattr(basic_solar, chem_params.solar_abundances)()

    a.mmax = chem_params.imf_mmax
    a.mmin = chem_params.imf_mmin
    a.sn2mmax = chem_params.sn2_mmax
    a.sn2mmin = chem_params.sn2_mmin
    a.agbmmax = chem_params.agb_mmax
    a.agbmmin = chem_params.agb_mmin

    #chempy found this to be the optimal value
    a.mass_steps = 200000

    if chem_params.imf_name == "Chabrier03":
        a.imf_type_name = 'Chabrier_1'
        a.chabrier_para1 = 0.69
        a.chabrier_para2 = 0.079
        a.high_mass_slope = -2.3
        a.imf_parameter = (a.chabrier_para1, a.chabrier_para2, a.high_mass_slope)
        basic_imf = IMF(a.mmin,a.mmax,a.mass_steps)
        getattr(basic_imf, a.imf_type_name)((a.chabrier_para1,a.chabrier_para2,a.high_mass_slope))
    else:
        raise ValueError("Other IMFs have not been implemented yet.")


    if stochastic == True:
        basic_imf.stochastic_sampling(ssp_mass)
        a.stochastic_IMF = True
        a.total_mass = ssp_mass
    else:
        a.stochastic_IMF = False

    # Load the yields of the default yield set
    basic_sn2 = SN2_feedback()
    getattr(basic_sn2, chem_params.ccsne_yields)() 
    basic_agb = AGB_feedback()
    getattr(basic_agb, chem_params.agb_yields)()
    basic_1a = SN1a_feedback()
    getattr(basic_1a, chem_params.sn1a_yields)()

    ############################################################################################
    elements_to_trace = list(np.unique(basic_agb.elements+basic_sn2.elements+basic_1a.elements))
        
    # Producing the SSP birth elemental fractions (here we use solar)
    solar_fractions = []
    elements = np.hstack(basic_solar.all_elements)
    for item in elements_to_trace:
        solar_fractions.append(float(basic_solar.fractions[np.where(elements==item)]))
    #################################################################################################

    a.yield_table_name_sn2 = chem_params.ccsne_yields
    a.yield_table_name_agb = chem_params.agb_yields

    #SN1a parameters.
    a.yield_table_name_1a =  chem_params.sn1a_yields
    # a.sn1a_time_delay = np.power(10,-1.39794)
    a.sn1a_time_delay = chem_params.sn1a_time_delay
    # SNIA normalization Maoz+2012
    # maoz2012 = 0.00130 #SNIA/Msun
    a.N_0 = chem_params.sn1a_norm
    a.sn1a_exponent = chem_params.sn1a_exponent

    a.sn1a_parameter = [a.N_0,a.sn1a_time_delay,a.sn1a_exponent,a.dummy]
               
    # Initialise the SSP class with time-steps
    #time_steps are fed in through the function
    basic_ssp = SSP(False, Z , np.copy(basic_imf.x), np.copy(basic_imf.dm), np.copy(basic_imf.dn), np.copy(time_steps), list(elements_to_trace), 'Argast_2000', 'logarithmic', net_yield)
    #In the above SSP object, the 'Argast_2000' denotes the parametrization for stellar lifetimes we use
    #'logarithmic' denotes the nature of interpolation over the metallicities in yield set
    
    #compute all the feedback now 
    basic_ssp.sn2_feedback(list(basic_sn2.elements), dict(basic_sn2.table), np.copy(basic_sn2.metallicities), float(a.sn2mmin), float(a.sn2mmax),solar_fractions)
    basic_ssp.agb_feedback(list(basic_agb.elements), dict(basic_agb.table), list(basic_agb.metallicities), float(a.agbmmin), float(a.agbmmax),solar_fractions) 
    basic_ssp.sn1a_feedback(list(basic_1a.elements), list(basic_1a.metallicities), dict(basic_1a.table), str(a.time_delay_functional_form), float(a.sn1ammin), float(a.sn1ammax), [a.N_0,a.sn1a_time_delay,a.sn1a_exponent,a.dummy],float(a.total_mass), a.stochastic_IMF)
        
    return time_steps, basic_ssp, elements_to_trace,solar_fractions


def compute_yZ(basic_ssp,elements):
    '''
    Function that computes the gross/total yield for metals (all metals)
    
    '''
    #I use Fe as the norm here as I imagine every yield set will compute this element
    
    yZ = np.zeros_like(basic_ssp.table["Fe"])
    for xi in elements:
        if xi != "H" and xi != "He":
            # yield_xi = np.cumsum(basic_ssp.agb_table[xi] + basic_ssp.sn2_table[xi] + basic_ssp.sn1a_table[xi] )
            yield_xi = np.cumsum(basic_ssp.table[xi])
            #by using the .table[] feature, we can directly get the gross yield
            yZ += yield_xi  
        
    return yZ



def create_chem_pickle(iniconf,chem_params,for_stochastic = False,verbose = True):
    '''
    In this function, we read or generate a grid of chemical yields
    '''

    # Load the yields of the default yield set
    basic_sn2 = SN2_feedback()
    getattr(basic_sn2, chem_params.ccsne_yields)() 
    basic_agb = AGB_feedback()
    getattr(basic_agb, chem_params.agb_yields)()
    basic_1a = SN1a_feedback()
    getattr(basic_1a, chem_params.sn1a_yields)()

    elements_to_trace = list(np.unique(basic_agb.elements+basic_sn2.elements+basic_1a.elements))

    all_ms_surv = []
    all_ms_feedback = []
    all_tt_cumul = []
    all_yZ_gross = []
    all_ssp_zmel = []


    z_grid = np.logspace(-6,-1.3,15)           
    #we compute the SSPs elemental feedback on this metallicity grid!

    #we make the dictionary we wish to populate
    all_yields = {}
    for ei in elements_to_trace:
        all_yields[ei+"_net"] = []
        all_yields[ei+"_diff"] = []
        
    if for_stochastic == False:
        print_stage("The chemical yield grid is being generated!")
    

    #all_yields is an empty dictionary    
    for zi in tqdm(z_grid):

        tt_i,ssp_net_i,elements_to_trace,solar_fractions =  get_SSP(zi,net_yield = True,chem_params=chem_params)
            
        tt_i,ssp_gross_i,elements_to_trace,solar_fractions =  get_SSP(zi,net_yield = False,chem_params = chem_params)
        
        yZ_gross_i = compute_yZ(ssp_gross_i,elements_to_trace)
        all_yZ_gross.append( yZ_gross_i) 
        
        #we need to loop through each element now ... 
        for ei in elements_to_trace:
            ei_net_yield = np.cumsum(ssp_net_i.agb_table[ei]+ssp_net_i.sn2_table[ei]+ssp_net_i.sn1a_table[ei]  )
            ei_gross_yield = np.cumsum(ssp_gross_i.agb_table[ei]+ssp_gross_i.sn2_table[ei]+ssp_gross_i.sn1a_table[ei]  )
            
            ei_diff = ei_gross_yield - ei_net_yield 
            
            all_yields[ei+"_net"].append(ei_net_yield)
            all_yields[ei+"_diff"].append(ei_diff)
            
        all_ms_surv.append( ssp_net_i.table['mass_in_ms_stars']) 
        all_ms_feedback.append(  np.cumsum(ssp_net_i.table['mass_of_ms_stars_dying']) - np.cumsum(ssp_net_i.table['mass_in_remnants'])     ) 
        
        all_tt_cumul.append(tt_i)
        all_ssp_zmel.append(zi + np.zeros_like(tt_i) )
                
    #let us concatenate all the terms inside the dictionary
    for ei in elements_to_trace:
        all_yields[ei+"_net"] = np.array(all_yields[ei+"_net"])
        all_yields[ei+"_diff"] = np.array(all_yields[ei+"_diff"])
            
    all_ssp_zmel = np.concatenate(all_ssp_zmel)
    all_ssp_zgrid = np.unique(all_ssp_zmel)
    all_tt_cumul = np.concatenate(all_tt_cumul)
    all_tt_grid = np.unique(all_tt_cumul)
    all_ms_surv = np.array(all_ms_surv)
    all_ms_feedback = np.array(all_ms_feedback)
    all_yZ_gross = np.array(all_yZ_gross)


    ##now we do the actual spline fitting
    ms_feedback_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), all_ms_feedback ,kx=1,ky=1)
    ms_surv_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), all_ms_surv ,kx=1,ky=1)
    yZ_gross_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), all_yZ_gross ,kx=1,ky=1)

    #now we loop over each of the dictionaries
    for ei in elements_to_trace:
        net_pts = all_yields[ei+"_net"]
        diff_pts = all_yields[ei+"_diff"]
        
        ei_net_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), net_pts ,kx=1,ky=1)
        ei_diff_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), diff_pts ,kx=1,ky=1)

        #if verbose is true, then we will print some example yields to see the values are according to expectations
        if verbose == True:
            if ei == "Fe":
                print("Total Net Fe yield, Z = 1e-3, t = 15 Gyr :",ei_net_interp([-3],[np.log10(15)],grid = False) )
                print("Total Net Fe yield, Z = 1e-3, t = 2 Gyr :",ei_net_interp([-3],[np.log10(2)] ,grid = False)  )
            if ei == "Mg":
                print("Total Net Mg yield, Z = 1e-3, t = 2 Gyr :",ei_net_interp([-3],[np.log10(2)], grid = False) )


        all_yields[ei+"_net"] = ei_net_interp
        all_yields[ei+"_diff"] = ei_diff_interp
        

    all_yields["ms_surv"] = ms_surv_interp
    all_yields["ms_feedback"] = ms_feedback_interp
    all_yields["yZ_gross"] = yZ_gross_interp


    all_yields["solar_fractions"] = solar_fractions
    all_yields["elements"] = elements_to_trace

    #now we add all the cham params
    all_yields["ccsne_yields"] = chem_params.ccsne_yields
    all_yields["agb_yields"] = chem_params.agb_yields
    all_yields["sn1a_yields"] = chem_params.sn1a_yields
    all_yields["stochastic"] = chem_params.stochastic
    all_yields["imf_mmax"] = chem_params.imf_mmax
    all_yields["imf_mmin"] = chem_params.imf_mmin
    all_yields["sn2_mmax"] = chem_params.sn2_mmax
    all_yields["sn2_mmin"] = chem_params.sn2_mmin
    all_yields["agb_mmax"] = chem_params.agb_mmax
    all_yields["agb_mmin"] = chem_params.agb_mmin
    all_yields["sn1a_time_delay"] = chem_params.sn1a_time_delay
    all_yields["sn1a_norm"] = chem_params.sn1a_norm
    all_yields["sn1a_exponent"] = chem_params.sn1a_exponent
    all_yields["imf_name"] = chem_params.imf_name


    if for_stochastic == False:

        ##let us check if this makes sense 
        save_pickle_path = iniconf['chem model']['chem_grids_path'] + "/" + iniconf['chem model']['chem_grid_pickle']
        with open(save_pickle_path, 'wb') as f:
            pickle.dump(all_yields, f)

        print_stage("The chemical yield grid has been stored here %s"%save_pickle_path)

    if for_stochastic == True:
        save_pickle_path = iniconf['chem model']['chem_grids_path'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "_analytical.pickle"
        #note that Mk refers to the kth SSP mass (in the list ssp_masses_grid) and Ig refers to the gth iteration with that SSP mass (there are a total of 75 iterations )
        with open(save_pickle_path, 'wb') as f:
            pickle.dump(all_yields, f)

    return 


def Sanders21_MnFe_SNIa(feh):
    '''
    This function returns the [Mn/Fe]_Ia yield using an approximation to Figure 8 from Sanders 21 https://arxiv.org/pdf/2106.11324.pdf
    '''
    m = 0.31431361784399675
    b = 0.24288017091128827

    return feh*m + b 


def create_chem_pickle_TRIAL(iniconf,chem_params,for_stochastic = False,verbose = True):
    '''
    In this function, we read or generate a grid of chemical yields
    '''

    # Load the yields of the default yield set
    basic_sn2 = SN2_feedback()
    getattr(basic_sn2, chem_params.ccsne_yields)() 
    basic_agb = AGB_feedback()
    getattr(basic_agb, chem_params.agb_yields)()
    basic_1a = SN1a_feedback()
    getattr(basic_1a, chem_params.sn1a_yields)()

    elements_to_trace = list(np.unique(basic_agb.elements+basic_sn2.elements+basic_1a.elements))

    all_ms_surv = []
    all_ms_feedback = []
    all_tt_cumul = []
    all_yZ_gross = []
    all_ssp_zmel = []
    all_Feh  = []
    z_grid = np.logspace(-6,-1.3,15)           
    #we compute the SSPs elemental feedback on this metallicity grid!

    #we make the dictionary we wish to populate
    all_yields = {}
    for ei in elements_to_trace:
        all_yields[ei+"_net"] = []
        all_yields[ei+"_diff"] = []

    all_yields["Mn_S21_IA_net"] = []
    all_yields["Mn_S21_CC_AGB_net"] = []

        
    if for_stochastic == False:
        print_stage("The chemical yield grid is being generated!")
    
    #all_yields is an empty dictionary    
    for zi in tqdm(z_grid):

        tt_i,ssp_net_i,elements_to_trace,solar_fractions =  get_SSP(zi,net_yield = True,chem_params=chem_params)
            
        tt_i,ssp_gross_i,elements_to_trace,solar_fractions =  get_SSP(zi,net_yield = False,chem_params = chem_params)
        
        yZ_gross_i = compute_yZ(ssp_gross_i,elements_to_trace)
        all_yZ_gross.append( yZ_gross_i) 
        
        #we need to loop through each element now ... 
        for ei in elements_to_trace:
            ei_net_yield = np.cumsum(ssp_net_i.agb_table[ei]+ssp_net_i.sn2_table[ei]+ssp_net_i.sn1a_table[ei]  )
            ei_gross_yield = np.cumsum(ssp_gross_i.agb_table[ei]+ssp_gross_i.sn2_table[ei]+ssp_gross_i.sn1a_table[ei]  )
            
            ei_diff = ei_gross_yield - ei_net_yield 
            
            all_yields[ei+"_net"].append(ei_net_yield)
            all_yields[ei+"_diff"].append(ei_diff)

            if ei == "Mn":
                #we add the modification here 
                #these grids are going to be a function of the gas [Fe/H] value and
                #I can get the 
                sn1a_tot_num = np.sum(ssp_net_i.table['sn1a'])
                Fe_1a_tot = np.sum(ssp_net_i.sn1a_table['Fe'])

                fe_per_snIa = Fe_1a_tot/sn1a_tot_num
                #we can compute how much iron is made per sn1a. And then we scale the Mn such that it agrees with Sanders 21 result 
                feh_value = np.log10(zi/0.015)
                mnfe_ia = Sanders21_MnFe_SNIa(feh_value)

                #using this we get the Mn yield in absolute terms
                mn_solar = 1.08178346e-05
                fe_solar = 0.00129197
                #this is the amount of Mn made per SnIa event according to the S21 approximation
                mn_per_snIa = (10**mnfe_ia) * (mn_solar / fe_solar) * fe_per_snIa

                Mn_sn1a_evo = ssp_net_i.table['sn1a'] * mn_per_snIa

                #we modify the CC SNE yield as well to agree with observations as well...this is more of like the starting point in the char
                #for now, let us not modify the CC SNE yields, let us just see how much SNIA can do. 
             
                all_yields["Mn_S21_IA_net"].append( np.cumsum(Mn_sn1a_evo)  )

                all_yields["Mn_S21_CC_AGB_net"].append( np.cumsum(ssp_net_i.agb_table["Mn"]+ssp_net_i.sn2_table["Mn"] ))

                all_Feh.append( feh_value + np.zeros_like(tt_i))

        all_ms_surv.append( ssp_net_i.table['mass_in_ms_stars']) 
        all_ms_feedback.append(  np.cumsum(ssp_net_i.table['mass_of_ms_stars_dying']) - np.cumsum(ssp_net_i.table['mass_in_remnants'])     ) 
        
        all_tt_cumul.append(tt_i)
        all_ssp_zmel.append(zi + np.zeros_like(tt_i) )
                
    #let us concatenate all the terms inside the dictionary
    for ei in elements_to_trace:
        all_yields[ei+"_net"] = np.array(all_yields[ei+"_net"])
        all_yields[ei+"_diff"] = np.array(all_yields[ei+"_diff"])

    all_yields["Mn_S21_IA_net"] = np.array(all_yields["Mn_S21_IA_net"])
    all_yields["Mn_S21_CC_AGB_net"] = np.array(all_yields["Mn_S21_CC_AGB_net"])

    all_feh_vals = np.concatenate(all_Feh)
    all_feh_grid = np.unique(all_feh_vals)
    print(all_feh_grid)

    all_ssp_zmel = np.concatenate(all_ssp_zmel)
    all_ssp_zgrid = np.unique(all_ssp_zmel)
    all_tt_cumul = np.concatenate(all_tt_cumul)
    all_tt_grid = np.unique(all_tt_cumul)
    all_ms_surv = np.array(all_ms_surv)
    all_ms_feedback = np.array(all_ms_feedback)
    all_yZ_gross = np.array(all_yZ_gross)


    ##now we do the actual spline fitting
    ms_feedback_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), all_ms_feedback ,kx=1,ky=1)
    ms_surv_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), all_ms_surv ,kx=1,ky=1)
    yZ_gross_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), all_yZ_gross ,kx=1,ky=1)

    #now we loop over each of the dictionaries
    for ei in elements_to_trace:
        net_pts = all_yields[ei+"_net"]
        diff_pts = all_yields[ei+"_diff"]
        
        ei_net_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), net_pts ,kx=1,ky=1)
        ei_diff_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), diff_pts ,kx=1,ky=1)

        if ei == "Mn":
            mn_ia_pts = all_yields["Mn_S21_IA_net"]
            mn_cc_agb_pts = all_yields["Mn_S21_CC_AGB_net"]

            Mn_S21_IA_interp = RectBivariateSpline(all_feh_grid,np.log10(all_tt_grid), mn_ia_pts ,kx=1,ky=1)
            Mn_S21_CC_AGB_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), mn_cc_agb_pts ,kx=1,ky=1)

            all_yields["Mn_S21_IA_net"] = Mn_S21_IA_interp
            all_yields["Mn_S21_CC_AGB_net"] = Mn_S21_CC_AGB_interp

        #if verbose is true, then we will print some example yields to see the values are according to expectations
        if verbose == True:
            if ei == "Fe":
                print("Total Net Fe yield, Z = 1e-3, t = 15 Gyr :",ei_net_interp([-3],[np.log10(15)],grid = False) )
                print("Total Net Fe yield, Z = 1e-3, t = 2 Gyr :",ei_net_interp([-3],[np.log10(2)] ,grid = False)  )
            if ei == "Mg":
                print("Total Net Mg yield, Z = 1e-3, t = 2 Gyr :",ei_net_interp([-3],[np.log10(2)], grid = False) )


        all_yields[ei+"_net"] = ei_net_interp
        all_yields[ei+"_diff"] = ei_diff_interp

    all_yields["ms_surv"] = ms_surv_interp
    all_yields["ms_feedback"] = ms_feedback_interp
    all_yields["yZ_gross"] = yZ_gross_interp


    all_yields["solar_fractions"] = solar_fractions
    all_yields["elements"] = elements_to_trace

    #now we add all the cham params
    all_yields["ccsne_yields"] = chem_params.ccsne_yields
    all_yields["agb_yields"] = chem_params.agb_yields
    all_yields["sn1a_yields"] = chem_params.sn1a_yields
    all_yields["stochastic"] = chem_params.stochastic
    all_yields["imf_mmax"] = chem_params.imf_mmax
    all_yields["imf_mmin"] = chem_params.imf_mmin
    all_yields["sn2_mmax"] = chem_params.sn2_mmax
    all_yields["sn2_mmin"] = chem_params.sn2_mmin
    all_yields["agb_mmax"] = chem_params.agb_mmax
    all_yields["agb_mmin"] = chem_params.agb_mmin
    all_yields["sn1a_time_delay"] = chem_params.sn1a_time_delay
    all_yields["sn1a_norm"] = chem_params.sn1a_norm
    all_yields["sn1a_exponent"] = chem_params.sn1a_exponent
    all_yields["imf_name"] = chem_params.imf_name


    if for_stochastic == False:

        ##let us check if this makes sense 
        save_pickle_path = iniconf['chem model']['chem_grids_path'] + "/" + iniconf['chem model']['chem_grid_pickle']
        with open(save_pickle_path, 'wb') as f:
            pickle.dump(all_yields, f)

        print_stage("The chemical yield grid has been stored here %s"%save_pickle_path)

    if for_stochastic == True:
        save_pickle_path = iniconf['chem model']['chem_grids_path'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "_analytical.pickle"
        #note that Mk refers to the kth SSP mass (in the list ssp_masses_grid) and Ig refers to the gth iteration with that SSP mass (there are a total of 75 iterations )
        with open(save_pickle_path, 'wb') as f:
            pickle.dump(all_yields, f)

    return 



###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

#the different SSP masses we will be considering
#the larger the mass, the less stochastic/noisy the IMF gets and yields tend towards the analytic approximation
#For SSP mass of 5e4 and above, the yield from analytic solution is used
#for SSP masses below 1e4, we will consider the nearest SSP mass in our grid and use that for the yield
#for each ssp mass we will have 50 different 2D grids (time vs. metallicity). Different iterations as each run is a different scenario as stochastic

ssp_masses_grid = [1,1e1,5e1,1e2,5e2,1e3,5e3,1e4]
stoc_iter = 50

def diagnostic_plots(iniconf=None):
    '''
    This function makes plots for the distribution of total SSP yield about the analytical value for some element. 
    We can use Mg as the diagnostic element here but it will be same for all elements
    '''

    ana_file = iniconf['chem model']['chem_grids_path'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "_analytical.pickle"

    stoc_files_temp = iniconf['chem model']['chem_grids_path'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "_M*_I*.pickle"
    stoc_files = glob.glob(stoc_files_temp)

    if len(stoc_files) != int(len(ssp_masses_grid)*stoc_iter):
        raise ValueError("The number of stochastic grid files does not match the expected number of 400 (8 x 50)")

    #load the analytical file
    with open(ana_file, 'rb') as f:
        grid_ana = pickle.load(f)

    #we will choose a sample element to make the diagnostic plots. We will choose Fe for this :)
    fe_grid_ana = grid_ana["Fe_net"]
    #we choose a sample metallicity and time to make the comparison
    #the below is the analytical value for net yield
    fe_val_ana = float(fe_grid_ana(np.array([-3]),[np.log10(15)],grid = False ))

    all_fe_stoc_vals = []

    all_m_inds = np.arange(len(ssp_masses_grid)) 

    for mi in all_m_inds: #we loop through the SSP masses
        fe_vals_mi = []
        for si in stoc_files: #we loop through all the files
            if "_M%d_"%mi in si: #we filter the files that correspond to the relevant mass
                with open(si, 'rb') as f:
                    grid_stoc_i = pickle.load(f)
                fe_grid_stoc = grid_stoc_i["Fe_net"]
                stoc_val_i = fe_grid_stoc(np.array([-3]),[np.log10(15)],grid = False )  
                fe_vals_mi.append(float(stoc_val_i))
        #we save the entire list of yields at a fixed SSP mass to the master list

        all_fe_stoc_vals.append(fe_vals_mi)


    #now we make the plots
    fig, ax = plt.subplots(2,4,figsize = (8,4))
    ax[0,0].hist(all_fe_stoc_vals[0],bins = "sqrt",color = "cadetblue",label = r"\rm M = 1")   
    ax[0,1].hist(all_fe_stoc_vals[1],bins = "sqrt",color = "cadetblue",label = r"\rm M = 1e1")   
    ax[0,2].hist(all_fe_stoc_vals[2],bins = "sqrt",color = "cadetblue",label = r"\rm M = 5e1")   
    ax[0,3].hist(all_fe_stoc_vals[3],bins = "sqrt",color = "cadetblue",label = r"\rm M = 1e2")   
    ax[1,0].hist(all_fe_stoc_vals[4],bins = "sqrt",color = "cadetblue",label = r"\rm M = 5e2")   
    ax[1,1].hist(all_fe_stoc_vals[5],bins = "sqrt",color = "cadetblue",label = r"\rm M = 1e3") 
    ax[1,2].hist(all_fe_stoc_vals[6],bins = "sqrt",color = "cadetblue",label = r"\rm M = 5e3")   
    ax[1,3].hist(all_fe_stoc_vals[7],bins = "sqrt",color = "cadetblue",label = r"\rm M = 1e4")   
    for axi in ax:
        for axii in axi:
            axii.set_yticks([])
            axii.vlines(x = fe_val_ana,ymin = 0,ymax = 10,lw = 3,color = "darkorange")
            axii.legend(fontsize = 6)
    ax[1,0].set_xlabel(r"\rm $y_{\rm Fe,net}$",fontsize = 8)
    ax[1,1].set_xlabel(r"\rm $y_{\rm Fe,net}$",fontsize = 8)
    ax[1,2].set_xlabel(r"\rm $y_{\rm Fe,net}$",fontsize = 8)
    ax[1,3].set_xlabel(r"\rm $y_{\rm Fe,net}$",fontsize = 8)

    plt.tight_layout()
    #these images are saved inside the folder where these pickle files are stored
    fig_save_path = iniconf['chem model']['chem_grids_path'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "/fe_yield_plots.png"
    plt.savefig(fig_save_path)
    plt.show()
        
    return

def gen_single_stoc_chem_grid(input_stuff):
    '''
    Function that generates a single 2D grid (age vs. metallicity) for a SSP. This written in function format so that it can parallelized
    '''

    #in each of these iterations we will generate a 2D grid

    g = input_stuff["g"] #index representing which iteration for a fixed SSP mass (ranges to 50)
    k = input_stuff["k"] #index representing which SSP mass 
    elements_to_trace = input_stuff["elements_to_trace"]
    z_grid = input_stuff["z_grid"]
    chem_params = input_stuff["chem_params"]
    ssp_mass_i = input_stuff["ssp_mass"]

    chem_params = convert(chem_params)

    #we make the dictionary we wish to populate
    all_yields_g_k = {} #for a specific SSP mass and for a specific iteration 
    for ei in elements_to_trace:
        all_yields_g_k[ei+"_net"] = []
        all_yields_g_k[ei+"_diff"] = []

    all_ms_surv = []
    all_ms_feedback = []
    all_tt_cumul = []
    all_yZ_gross = []
    all_ssp_zmel = []

    for zi in z_grid:

        #we generate the SSP object
        tt_i,ssp_obj_i,elements_to_trace,solar_fractions =  get_SSP(zi,net_yield = True,chem_params=chem_params,stochastic=True,ssp_mass=ssp_mass_i)
        
        #there is no need to separately compute the gross yieldss
        #the gross yields are always stored in .table[element] format
        #te net yields can be computed by adding the individual channeltables
        # tt_i,ssp_gross_i,elements_to_trace,solar_fractions =  get_SSP(zi,net_yield = False,chem_params = chem_params)
        
        yZ_gross_i = compute_yZ(ssp_obj_i,elements_to_trace)
        all_yZ_gross.append( yZ_gross_i) 
        
        #we need to loop through each element now ... 
        for ei in elements_to_trace:
            ei_net_yield = np.cumsum(ssp_obj_i.agb_table[ei]+ssp_obj_i.sn2_table[ei]+ssp_obj_i.sn1a_table[ei]  )
            ei_gross_yield = np.cumsum(ssp_obj_i.table[ei])
            
            ei_diff = ei_gross_yield - ei_net_yield 
            
            #appending net yields and the difference of the net yields and gross yields (for each element ei) to the master dictionary
            #note that the difference of net and gross yields is stored so that we can compute the gross yield for any initial elemental fraction
            all_yields_g_k[ei+"_net"].append(ei_net_yield)
            all_yields_g_k[ei+"_diff"].append(ei_diff)
            
        all_ms_surv.append( ssp_obj_i.table['mass_in_ms_stars']) 
        all_ms_feedback.append(  np.cumsum(ssp_obj_i.table['mass_of_ms_stars_dying']) - np.cumsum(ssp_obj_i.table['mass_in_remnants'])     ) 
        
        all_tt_cumul.append(tt_i)
        all_ssp_zmel.append(zi + np.zeros_like(tt_i) )
        
    #let us concatenate all the terms inside the dictionary
    for ei in elements_to_trace:
        all_yields_g_k[ei+"_net"] = np.array(all_yields_g_k[ei+"_net"])
        all_yields_g_k[ei+"_diff"] = np.array(all_yields_g_k[ei+"_diff"])
            
    all_ssp_zmel = np.concatenate(all_ssp_zmel)
    all_ssp_zgrid = np.unique(all_ssp_zmel)
    all_tt_cumul = np.concatenate(all_tt_cumul)
    all_tt_grid = np.unique(all_tt_cumul)
    all_ms_surv = np.array(all_ms_surv)
    all_ms_feedback = np.array(all_ms_feedback)
    all_yZ_gross = np.array(all_yZ_gross)


    ##now we do the actual spline fitting (linear fitting)
    ms_feedback_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), all_ms_feedback ,kx=1,ky=1)
    ms_surv_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), all_ms_surv ,kx=1,ky=1)
    yZ_gross_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), all_yZ_gross ,kx=1,ky=1)

    #now we loop over each of the dictionaries
    for ei in elements_to_trace:
        net_pts = all_yields_g_k[ei+"_net"]
        diff_pts = all_yields_g_k[ei+"_diff"]
        
        ei_net_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), net_pts ,kx=1,ky=1)
        ei_diff_interp = RectBivariateSpline(np.log10(all_ssp_zgrid),np.log10(all_tt_grid), diff_pts ,kx=1,ky=1)
        
        all_yields_g_k[ei+"_net"] = ei_net_interp
        all_yields_g_k[ei+"_diff"] = ei_diff_interp


    all_yields_g_k["ms_surv"] = ms_surv_interp
    all_yields_g_k["ms_feedback"] = ms_feedback_interp
    all_yields_g_k["yZ_gross"] = yZ_gross_interp

    all_yields_g_k["solar_fractions"] = solar_fractions
    all_yields_g_k["elements"] = elements_to_trace

    #now we add all the cham params
    all_yields_g_k["ccsne_yields"] = chem_params.ccsne_yields
    all_yields_g_k["agb_yields"] = chem_params.agb_yields
    all_yields_g_k["sn1a_yields"] = chem_params.sn1a_yields
    all_yields_g_k["stochastic"] = chem_params.stochastic
    all_yields_g_k["imf_mmax"] = chem_params.imf_mmax
    all_yields_g_k["imf_mmin"] = chem_params.imf_mmin
    all_yields_g_k["sn2_mmax"] = chem_params.sn2_mmax
    all_yields_g_k["sn2_mmin"] = chem_params.sn2_mmin
    all_yields_g_k["agb_mmax"] = chem_params.agb_mmax
    all_yields_g_k["agb_mmin"] = chem_params.agb_mmin
    all_yields_g_k["sn1a_time_delay"] = chem_params.sn1a_time_delay
    all_yields_g_k["sn1a_norm"] = chem_params.sn1a_norm
    all_yields_g_k["sn1a_exponent"] = chem_params.sn1a_exponent
    all_yields_g_k["imf_name"] = chem_params.imf_name

    #now that we have added all the information to the master dictionary let us save this dictionary as a pickle

    ##let us check if this makes sense 
    #note that iniconf['chem setup']['chem_grid_pickle'] will be the folder name and also serve as the format for sub-pickles
    #the existence of the folder has already been confirmed earlier and has been created if not earlier
    save_pickle_path = iniconf['chem model']['chem_grids_path'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "/" + iniconf['chem model']['chem_grid_pickle'] + "_M%s_I%s.pickle"%(k,g)
    #note that Mk refers to the kth SSP mass (in the list ssp_masses_grid) and Ig refers to the gth iteration with that SSP mass (there are a total of 75 iterations )
    with open(save_pickle_path, 'wb') as f:
        pickle.dump(all_yields_g_k, f)

    return 


def create_chem_pickle_stochastic(iniconf,chem_params):
    '''
    In this function, we read or generate a grid of chemical yields with stochastic IMF sampling

    things to add 

    add option to ini file if we want to do the iteration for same halo multiple times
    the choice of random iteration to use for each SSP mass might be different each time and hence result in a slightly different elemental yield
    this will just rerun the galactic chemical evolution model again. 

    I need to switch the order of things in my ini file
    need to put things needed for chemical grid generation together and rest all in other section
    NEED TO CHECK ALL THE THINGS ARE CONSISTENT AFTERWARDS

    Need to think about how to save the number of SN1A events etc. in the stochastic case (but will leave this for a future issue)

    Need to see how this stochastic treatment will change my function in run_chemical.py

    I need to add total average elemental yields to the summary csv files  
    
    DO I NEED TO GENERATE A NEW SSP object everytime? or I can repopulate those tables by computing the feedback again
    same for analytical model. so instead of get_SSP in that loop, we just have the .sn1a_feedback() and .sn2_feedback() etc.
    So turns out we need to feed in the Z and ssp mass (in case of stochastic) to get_SSP hence we cannot do the above.
    Howver, in case of stochastic grid, we can speed up the 50 iterations as it is basically at the same SSP mass and we can already create the 15 or so SSPs for differetn metallicities
    '''

    # Load the yields of the default yield set
    basic_sn2 = SN2_feedback()
    getattr(basic_sn2, chem_params.ccsne_yields)() 
    basic_agb = AGB_feedback()
    getattr(basic_agb, chem_params.agb_yields)()
    basic_1a = SN1a_feedback()
    getattr(basic_1a, chem_params.sn1a_yields)()

    elements_to_trace = list(np.unique(basic_agb.elements+basic_sn2.elements+basic_1a.elements))

    z_grid = np.logspace(-6,-1.3,15)
    #the choice for how fine of a grid I should make in metallicity will depend on how finely the metallicity grid for SN2/AGB yields is
    #if the yield grid is originally quite sparse, there is no need to make this very fine           
    #we compute the SSPs elemental feedback on this metallicity grid!
       
    print_stage("The chemical yield grid (stochastic version) is being generated!")


    master_pickle_folder = iniconf['chem model']['chem_grids_path'] + "/" + iniconf['chem model']['chem_grid_pickle']
    #we need to ensure that this folder exists.
    check_path_existence(all_paths=[master_pickle_folder])

    

    run_parallel = iniconf['chem setup']['run_chemical_parallel']


    all_inputs = []
    for k,ssp_mass_i in enumerate(ssp_masses_grid):
        for g in range(stoc_iter):
            input_stuff_i = { "g":g,"k":k,"elements_to_trace":elements_to_trace,"z_grid":z_grid,
                            "chem_params":chem_params._asdict(),"ssp_mass":ssp_mass_i}
            all_inputs.append(input_stuff_i)

    if run_parallel == "False":
        for inputs_i in tqdm(all_inputs):
            gen_single_stoc_chem_grid(inputs_i)

    else:
        ncores = int(iniconf['run params']['ncores'])
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
            _ = list(tqdm(executor.map(gen_single_stoc_chem_grid,all_inputs), total = len(all_inputs)))


    #we need to run an analytical grid as well for SSP masses that are larger than 5e4
    create_chem_pickle(iniconf = iniconf,chem_params = chem_params,for_stochastic = True) 
    #setting the for_stochastic flag to True just means that the pickle created is stored in a folder like all the above ones

    print_stage("The chemical yield grid (stochastic version) has been stored here %s"%master_pickle_folder)


    #make diagnostic plots here of how the total yield computed from these stochastic grid spreads about the analytical value.
    #this is just to make sure that the stochastic grid method is working fine
    diagnostic_plots(iniconf = iniconf)

    return 



if __name__ == '__main__':
    # read in command line arguments
    args = argument_parser().parse_args()
    # read parameters and information from the run config file 
    iniconf = read_config_file(config_file=args.ini)

    chem_params = extract_chem_params(iniconf=iniconf,print_params=True)

    if chem_params.stochastic == "True":
        create_chem_pickle_stochastic(iniconf = iniconf,chem_params = chem_params)
    else:
        create_chem_pickle(iniconf = iniconf,chem_params = chem_params)
        # create_chem_pickle_TRIAL(iniconf = iniconf,chem_params = chem_params)







