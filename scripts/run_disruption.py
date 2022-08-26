#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this script reads the relevant data and calculates the disruption / survival probabilties 
"""

import numpy as np
import math
import os
from scipy.interpolate import UnivariateSpline
from scipy.integrate import odeint
import numpy as np
import pandas as pd
from astropy.io import ascii
import pickle
from pathlib import Path
from scipy.interpolate import BPoly
from scipy.optimize import minimize
from read_tracks import unpack_data
import logging
from tqdm import tqdm

def spline_slope(x,fx,sx):
    """
    Computes the spline approximation of a set of points constrained by the derivative at each point

    Parameters:
    -----------
        x: array, x coordinates of points
        fx: array, y coordinates of points OR f(x)
        sx: array, derivative at each of these points

    Returns:
    -----------
       spl: spline object that satisfies the constraints of x,fx, and sx.
    """
    
    info = []
    for i in range(len(fx)):
        info.append( [fx[i],sx[i]] )
    
    spl = BPoly.from_derivatives(x,info)
    
    return spl


def get_orbit(ind,data):
    """
    Computes the spline approximation constrained by velocity at each point for the X,Y,Z plane orbital tracks of halos. 

    Parameters:
    -----------
        ind: integer, index of the halo
        data: dictionary, all track information for halos

    Returns:
    -----------
       splobj: list of the spline approximations for each track [x_spl,y_spl,z_spl]
    """
   
    times,aexp,x,y,z,vx,vy,vz = unpack_data(data,types = ["times","aexp","x","y","z","vx","vy","vz"])
    # times,rvir,rs,aexp,x,y,z,vx,vy,vz,mvir,m200c, upID,ID,host_ID,vmax = unpack_data(data)
        
    aexpi = aexp[ind]
    xti,yti,zti = x[ind], y[ind], z[ind]
    vxti, vyti , vzti = vx[ind], vy[ind], vz[ind]
    tti = times[ind]

    #now onto to the spline fitting
    x_spl = spline_slope(tti,xti,vxti)
    y_spl = spline_slope(tti,yti,vyti)
    z_spl = spline_slope(tti,zti,vzti)
                    
    splobj = [x_spl,y_spl,z_spl] 
    
    #the data dictionary contains all the useful stuff we will need to plot later on
    data = {'t': tti, 'a': aexpi,'x': xti,'y': yti, 'z': zti, 
        'vx': vxti,'vy': vyti , 'vz': vzti }
    
    return splobj


def first_minima(data,ind,dists,scale,time,zspl,method = "discrete"):
    """
    Computes the pericenter distance and time of pericenter for a halo using track information

    Parameters:
    -----------
        data: dictionary, all track information for halos
        ind: integer, index of the halo
        dists: list, distance of halo from host at each time step post accretion
        scale: list, scale factor of halo at each time step post accretion 
        time: list, time (Gyr) at each time step post accretion
        zspl: spline object, time (Gyr) to redshift z spline approximation
        method: "velo" or "discrete, 
                method to compute the pericenter distance. "velo" finds minimum host distance 
                of spline fitted track. "discrete" finds minimum host distance in the known finite time steps. 
    Returns:
    -----------
       dperi: float, distance (kpc) of first pericenter approach after accretion
       aperi: float, scale factor of first pericenter approach after accretion 

    Notes:
    -----------
    1. In case there is no pericenter, dperi is taken as distance of halo at z = 0. 

    """
    dists = np.array(dists)
    scale = np.array(scale)
    time = np.array(time)
    #In forward time direction for ease
    
    n = len(dists)

    if np.min(time) != time[0]:
        raise ValueError("In run_disruption.py : Time array is not in ascending order.")
    
    if len(dists) != len(scale):
        raise ValueError("In run_disruption.py : Minima Function: Array length not matching 1")
        
    if len(time) != len(scale):
        raise ValueError("In run_disruption.py : Minima Function: Array length not matching 2")
        

    if np.max(np.diff(dists)) < 0:
        #halo distance from host is monotonically decreasing
        #halo has never had a pericenter approach and is infalling for the first time
        #thus dperi is just the final distance (z = 0)
        return dists[-1], scale[-1]
    
    if np.min(np.diff(dists)) > 0:
        #halo moving away from host but still within the halo
        #such cases are rare I believe (have not seen any), but better to deal with in any case
        return dists[-1],scale[-1]

    maxind = np.argmax(dists)
    if np.max(dists) == dists[-1] or np.max(dists) == dists[0]:
        #if the maximum halo distance is the first or last element of list
        dd1 = -1
        dd2 = 1
    else:
        #we split the dists array into the 2 halves. 
        d1 = dists[0:maxind+1]
        d2 = dists[maxind:]
        dd1 = np.min(np.diff(d1))
        dd2 = np.max(np.diff(d2))
        
    if  dd1 > 0 and dd2 < 0:
    #this is the case where the distance is monotonically changing, but also there isnt a minima
    #in this case we still return the distance at final time. However, we will have to make a separate case for this
    #example case is d= [1,2,3,2,1] --> np.diff(d) = [ 1,  1, -1, -1]
    #for definition of dd1 and dd2, refer to earlier part of the function
        return dists[-1],scale[-1]
    
    #now we are left with the most common scenario, that is, the distance is not changing monotonically in time
    #what this means is that there are one or more pericenter approaches. 
    #Our goal is to find the first such approach
    #bear in mind that the inputted lists are already filtered for post-accretion times only, so no need to worry about that

    else:
        for i in range(1,n-1):
            if dists[i] < dists[i+1] and dists[i] < dists[i-1]:
            #the first pericenter approach
                if method == "discrete":
                    return dists[i],scale[i]
                if method == "velo":
                    #we will consider the 2 boundaries that is scale[i-1] and scale[i+1]. 
                    #And then consider spline in between that 
                    #first get the splines for this object
                    spls = get_orbit(ind,data)
                    xcspl,ycspl,zcspl = spls[0],spls[1],spls[2]
                    def dist_to_host(t):
                        return np.sqrt(xcspl(t)**2 + ycspl(t)**2 + zcspl(t)**2)*1e3
                    res = minimize(dist_to_host,time[i],bounds= [(time[i -1],time[i+1])])
                    tmin = res.x
                    dperi = float(dist_to_host(float(tmin)))
                    #convert tmin to scale factor
                    aperi = 1/ ( 1 +  zspl(float(tmin))  ) 
                    # aperi = 1/ ( 1 + float(cosmo.age(tmin,inverse = True)) )

                    #as a sanity check to see if it is inbetween scale[i-1] and scale[i+1]
                    if aperi < scale[i-1] or aperi > scale[i+1]:
                        print(tmin,zspl(tmin),zspl(13.72),"|||",scale[i+1],aperi,scale[i-1])
                        print("Issue with Velo Spline in first minima")
                        
                    return dperi,aperi


    ##IN THE ABOVE FUNCTION, I NEED TO MAKE SURE THAT VELOCITIES ARE BEING CONVERTED TO COMOVING BEFORE SPLINE FITITNG WITH COMOVING COORDINATES


def get_acc_time(ind,data):
    """
    Computes the time of accretion for a given halo

    Parameters:
    -----------
        ind: integer, index of the halo
        data: dictionary, all track information for halos

    Returns:
    -----------
        a_acc: float, scale factor at accretion time
        vm_acc: float, maximum circular velocity of subhalo at accretion time
        mvir_acc: float, virial mass of subhalo at accretion time

    Notes:
    -----------
    1. Accretion time is defined as the last snapshot, working backward in time from z=0, at which a 
       subhalo's host ID is equal to the main host's ID. Physically this occurs when a subhalo enters the virial
       radius of the host halo for the FINAL time. Note that a subhalo could have been contained within the host 
       halo's virial radius at an earlier time and later re-accreted. We choose the final accretion event for 
       each sub-halo. 
    2. If halo is never accreted, -99,-99,-99 is returned
    3. If upID of a subhalo is -1, then the subhalo is isolated. 
    4. An important constrain of the disruption model is that the subhalo under consideration must be a subhalo of the main host at z=0. 
    """
    
    times,aexp,mvir,ID,upID,vmax = unpack_data(data, types = ["times","aexp","mvir","ID","upID","vmax"])

    #compute the host_ID
    host_ID = ID[0][::-1]
    host_times = times[0][::-1]

    aexpi = aexp[ind][::-1]
    upIDi = upID[ind][::-1]
    mviri = mvir[ind][::-1]
    vmaxi = vmax[ind][::-1]
    
    ti = times[ind][::-1]

    t_halo_start = np.min(ti)
    t_host_start = np.min(host_times)

    t_true_start = np.maximum(t_halo_start,t_host_start)

    #filtering for times for when the halo and host are defined
    aexpi = aexpi[ ti > t_true_start]
    vmaxi = vmaxi[ ti > t_true_start]
    upIDi = upIDi[ti > t_true_start]
    host_ID_f = host_ID[host_times > t_true_start]

    if np.max(ti) != ti[0]:
        raise ValueError("In run_disruption.py : Order to time array is reserved in accretion time function")

    #THIS DISRUPTION MODELLING DOES NOT MAKE ANY SENSE FOR SUBHALOS THAT ARE NOT PART OF HOST AT Z=0
    #SO WE CAN JUST APPLY THAT CUT HERE AND MAKE THINGS SUPER SIMPLE

    if upIDi[0] == host_ID_f[0]:
        #this subhalo is part of the host at z = 0
        # thus doing disurption modelling on this makes sense

        temp = upIDi - host_ID_f 

        #this is the ind when for first time temp != 0
        try: 
            t_ind = next(t[0] for t in enumerate(temp) if t[1] != 0)
            acc_ind = t_ind - 1  

        except:
            acc_ind = len(temp) - 1

        a_acc = aexpi[acc_ind]
        vm_acc = vmaxi[acc_ind]
        mvir_acc = mviri[acc_ind]

        return a_acc, vm_acc, mvir_acc
    else:
    
        #this subhalo is not part of host at z=0
        #the host also falls under this condition as the upID of main host is equal to -1.
        return -99.,-99., -99. 
    
 
def get_first_peri(ind,data,zspl):
    """
    master function to compute the pericenter distance and time of pericenter for a halo

    Parameters:
    -----------
        ind: integer, index of the halo
        data: dictionary, all track information for halos
        zspl: spline object, time (Gyr) to redshift z spline approximation

    Returns:
    -----------
        dperi: float, scale factor at accretion time
        aperi: float, maximum circular velocity of subhalo at accretion time
    
    """

    times,aexp,x,y,z,vx,vy,vz = unpack_data(data,types=["times","aexp","x","y","z","vx","vy","vz"])
    # times,rvir,rs,aexp,x,y,z,vx,vy,vz,mvir,m200c, upID,ID,host_ID,vmax = unpack_data(data)

    #first identify if the halo is ever accreted
    a_acc,vm_acc,mvir_acc = get_acc_time(ind,data)
    
    if a_acc < 0:
        #accretion time is not defined as halo is never accreted by host
        return -99., -99.

    else:
        aexpi = aexp[ind]
        xti,yti,zti = x[ind], y[ind], z[ind]
        vxti, vyti , vzti = vx[ind], vy[ind], vz[ind]
        tti = times[ind]
        
        #filter out the arrays for times after accretion 
        xti,yti,zti = xti[aexpi >= a_acc], yti[aexpi >= a_acc], zti[aexpi >= a_acc]
        vxti, vyti, vzti = vxti[aexpi >= a_acc], vyti[aexpi >= a_acc], vzti[aexpi >= a_acc]
        tti = tti[aexpi >= a_acc]
        aexpi = aexpi[aexpi >= a_acc]

        if len(vxti) == 0:
            raise ValueError("In run_disruption.py : Halo has no time steps after accretion.")
                
        if len(vxti) == 1:
            #halo exists only for a single time step  
            dperi = float(np.sqrt(xti**2 + yti**2 + zti**2)*1e3)
            aperi = float(aexpi)

            #convert dperi to units of physical distance
            dperi = dperi*aperi
            return dperi,aperi

        else:
            dists = np.sqrt(xti**2 + yti**2 + zti**2)*1e3
            dperi,aperi = first_minima(data,ind,dists,aexpi,tti,zspl)

            #convert dperi to units of physical distance
            dperi = dperi*aperi

            return dperi,aperi
        

def get_disruption(data,zspl,subhalo_rf_dir=None):
    """
    run disruption model on halos for isolated ELVIS suites. 

    Parameters:
    -----------
        data: integer, index of the halo
        zspl: spline object, time (Gyr) to redshift z spline approximation
        subhalo_rf_dir: string, directory containing the subhalo random forest model files

    Returns:
    -----------
        survival_probs: list, suvivial probabilities for all halos
        m_acc: list, virial mass at accretion time for all halos
        v_acc: list, maximum circular velocity at accretion time for all halos.
        a_peri: list, scale factor at first pericenter for all halos
        d_peri: list, first pericenter distance for all halos

    Notes:
    -----------
    1. It is crucial to be aware whether the coordinates and velocities are in comoving or physical units. 
       We should be working in physical units.     
    """

    #load the model
    #this is the path to the subhalo random forest model 
    filename = subhalo_rf_dir + '/subhalo_randomforest/finalized_rf.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    survival_probs = []

    m_acc = []; v_acc = []; a_acc_list = []
    a_peri = []; d_peri = [] 

    n = len(data['times'])
    # mah_names = data['mah_names']

    for i in range(n):

        dperi,aperi = get_first_peri(i,data,zspl)
        a_acc,vm_acc,mvir_acc = get_acc_time(i,data)

        # if mah_names[i] == "caterpillar_26_LX14_a_1.00_3697654_2":
        #     print(a_acc,vm_acc,dperi)

        if a_acc > 0 and dperi > 0: 
            #these are the cases where the halo is accreted
            feats = np.array([dperi,a_acc,vm_acc,mvir_acc,aperi])
            feats = feats.reshape(1,-1)
            probs = loaded_model.predict_proba(feats)
            survival_probs.append(float(probs[:,0]))
            m_acc.append(mvir_acc)
            v_acc.append(vm_acc)
            a_acc_list.append(a_acc)
            a_peri.append(aperi)
            d_peri.append(dperi)

        else:
            #these are the scenarios where the halo is no accreted by host. Therefore, we assume no disruption.
            survival_probs.append(1.)
            m_acc.append(-99.)
            v_acc.append(-99.)
            a_acc_list.append(-99.)
            a_peri.append(-99.)
            d_peri.append(-99.)


    return survival_probs,m_acc,v_acc,a_acc_list,a_peri,d_peri

def create_disruption_pickle(pickleloc=None,pickle_obj=None):
    """
    create pickle object for disruption modelling statistics

    Parameters:
    -----------
        pickleloc: string, path where pickle file will be stored
        pickle_obj: dictionary, disruption modelling statistics
    Returns:
    -----------
        None
    """
    #if pickle with that name already exists. Then first delete existing one and make new now. 
    if os.path.exists(pickleloc):
        os.remove(pickleloc)
        print("Existing Pickle file has been deleted.")

    with open(pickleloc, 'wb') as f:
        pickle.dump(pickle_obj, f)

    print("Pickle file is stored here : " + pickleloc)

    return 


def run_disruption_model(pickle_name=None,data=None,zspl=None,iniconf=None):
    """
    master function that runs the entire disruption model 

    Parameters:
    -----------
        run_type: string, type of ELVIS suite (eg. "elvis_iso", "elvis_pair" etc.)
        elvis_name: string, name of the ELVIS suite
        data: dictionary, all track information for halos
        zspl: spline object, time (Gyr) to redshift z spline approximation
        subhalo_rf_dir: string, directory containing the subhalo random forest model files
        pickle_path: string, directory where pickle files are stored
        forcepickle: "yes" or "no", decision to force making a pickle file or not
    Returns:
    -----------
        survival_probs: list, suvivial probabilities for all halos
        m_acc: list, virial mass at accretion time for all halos
        v_acc: list, maximum circular velocity at accretion time for all halos.
        a_peri: list, scale factor at first pericenter for all halos
        d_peri: list, first pericenter distance for all halos
    """
    
    pickle_path = iniconf['output paths']['pickle_disruption_dir']
    pickleloc = pickle_path + "/" + pickle_name + "_disruption" + ".pickle"
    my_pickle = Path(pickleloc)

    # subhalo_rf_dir = iniconf['data paths']['subhalo_rf_dir']
    #this code is already included in the scripts folder
    subhalo_rf_dir = iniconf['main paths']['code_dir'] + "/scripts"


    forcepickle = iniconf['run params']['create_pickle']

    if my_pickle.is_file():
        # #pickle file already exists
        if forcepickle == "True":
            #force to make a new pickle file
            print("New Pickle file for disruption information is being created. Earlier pickle will be deleted.")
            survival_probs,m_acc,v_acc,a_acc,a_peri,d_peri = get_disruption(data,zspl,subhalo_rf_dir=subhalo_rf_dir)

            pickle_obj = {'surv_probs' : survival_probs, 'm_acc': m_acc, 'v_acc': v_acc, 'a_peri': a_peri, 'd_peri':d_peri,'a_acc':a_acc}
            #then store this as a pickle file
            create_disruption_pickle(pickleloc=pickleloc,pickle_obj=pickle_obj)

        # else:
        #Will not make new pickle as it already exists and is not forced to make a new one.  
        print("The pickle file for disruption information is being used")
        with open(pickleloc, 'rb') as f:
            loaded_obj = pickle.load(f)

        survival_probs = loaded_obj["surv_probs"]
        m_acc = loaded_obj["m_acc"]
        v_acc = loaded_obj["v_acc"]
        a_peri = loaded_obj["a_peri"]
        a_acc = loaded_obj["a_acc"]
        d_peri = loaded_obj["d_peri"]

    else:
        #pickle file does not exist therefore will be making one. 
        print("Pickle file for disruption information is being created")
        survival_probs,m_acc,v_acc,a_acc,a_peri,d_peri = get_disruption(data,zspl,subhalo_rf_dir=subhalo_rf_dir)

        pickle_obj = {'surv_probs' : survival_probs, 'm_acc': m_acc, 'v_acc': v_acc, 'a_peri': a_peri, 'd_peri':d_peri, 'a_acc':a_acc}
        #then store this as a pickle file
        create_disruption_pickle(pickleloc=pickleloc,pickle_obj=pickle_obj)

    #now we filter for the sub_good_ind

    disruption_dict = {  'surv_probs':survival_probs, 'm_acc': m_acc, 'v_acc' : v_acc, 'a_peri': a_peri, 'd_peri':d_peri , 'a_acc':a_acc }

    return disruption_dict



def filter_disruption(disruption_dict=None,sub_good_ind=None):
    """
    function that filters the disruption statistics for relevant halos. 

    Parameters:
    -----------
        survival_probs: list, suvivial probabilities for all halos
        m_acc: list, virial mass at accretion time for all halos
        v_acc: list, maximum circular velocity at accretion time for all halos.
        a_peri: list, scale factor at first pericenter for all halos
        d_peri: list, first pericenter distance for all halos
        sub_good_ind: list, the good indices that need to be filtered for

    Returns:
    -----------
        disruption_dict: dictionary, filtered dictionary of disruption statistics
    """

    surv_probs_filt = []
    m_acc_filt = [] #mvir of subhalo at its accretion time
    v_acc_filt = [] #vmax of subhalo at its accretion time
    a_acc_filt = [] #scale factor at time of accretion
    a_peri_filt = [] #scale factor at first pericenter approach
    d_peri_filt = [] #first pericenter distance

    survival_probs = disruption_dict["surv_probs"]
    m_acc = disruption_dict["m_acc"]
    v_acc = disruption_dict["v_acc"]
    a_acc = disruption_dict["a_acc"]
    d_peri = disruption_dict["d_peri"]
    a_peri = disruption_dict["a_peri"]

    for i, sp in enumerate(survival_probs):
        if i in sub_good_ind:
            surv_probs_filt.append(sp)
            m_acc_filt.append(m_acc[i])
            v_acc_filt.append(v_acc[i])
            a_acc_filt.append(a_acc[i])
            d_peri_filt.append(d_peri[i])
            a_peri_filt.append(a_peri[i])

    disruption_dict_filter = {  'surv_probs':surv_probs_filt, 'm_acc': m_acc_filt, 'v_acc' : v_acc_filt, 'a_peri': a_peri_filt, 'd_peri':d_peri_filt , 'a_acc':a_acc_filt }

    return disruption_dict_filter