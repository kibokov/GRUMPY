## this script contains the functions needed to spline fit the MAH_fitting and obtain relevant information to be fed into the integration.
import numpy as np
from scipy.interpolate import UnivariateSpline
import pickle
import os
from pathlib import Path
from read_tracks import unpack_data
from age_integrate import age
from read_tracks import reduce_data
from tqdm import tqdm


def print_stage(line2print, ch='-'):
    nl = len(line2print)
    print(ch*nl)
    print(line2print)
    print(ch*nl)
    print(' ')



def single_tmdlmdlt(input_stuff):

    i = input_stuff["i"]
    mah_names = input_stuff["mah_names"]
    tstart_max = input_stuff["tstart_max"]
    mpeak_min = input_stuff["mpeak_min"]

    t_i = input_stuff["times"]
    aexp_i = input_stuff["aexp"]
    m200c_i = input_stuff["m200c"] #array of m200c halo masses

    #in the reduce_data function, the above have already been filtered for bad times.
    mpeak_i = np.max(m200c_i)
    #to compute the slope and peak, we make use of M200c

    if mpeak_i > mpeak_min and np.min(t_i) < tstart_max:

        lt, lmv = np.log(t_i), np.log(m200c_i/m200c_i[-1])

        spl = UnivariateSpline(lt, lmv, s=0)
        spltd = spl(lt)
        #we constrain the spline to positive slopes only
        for ih, _ in enumerate(spltd[1:]):
            if spltd[ih+1] < spltd[ih]:
                spltd[ih+1] = spltd[ih]

        dlogmdlogt = np.diff(spltd) / np.diff(lt)
        middle_t = (t_i[1:] + t_i[:-1])*0.5

        dlmdltspl = UnivariateSpline(middle_t, dlogmdlogt, s=0.)  

        astart = aexp_i[1]
        #this has to be 1, because that is from where we start counting

        return i,mah_names, t_i[1], m200c_i[1], dlmdltspl, astart
    else:
        return -99,-99,-99,-99,-99,-99
 

def get_tmdlmdlt(data=None, pickleloc=None,iniconf=None,pickle_create=False):
    '''

    compute splines for MAH slopes and the indices of halos that pass the minimum Mpeak cut and formation time cut.

    Parameters:
    -----------
        data: dictionary, 
        pickleloc: string, 
        pickle_create: boolean, boolean representing whether pickle files for  

    Returns:
    -----------
        good_index: list, indices of halos that satisfy the Mpeak cut and formation time cut
        tmdlmdlt: list, information for integration of MAH - [t_start, m_start,MAH_slope_splines,a_start]
    '''

    times, aexp, m200c,mah_names = unpack_data(data,types = ['times','aexp','m200c','mah_name']) 

    mpeak_min = float(iniconf['run params']['mpeak_min'])
    tstart_max = float(iniconf['run params']['tstart_max'])

    #run the function either in parallel or series
    run_parallel = iniconf['run params']['run_mah_parallel']
    ncores = int(iniconf['run params']['ncores'])

    if run_parallel == "True":
        #make the input stuff dictionary
        input_stuff = []

        for i in range(len(times)):
            temp_dict = {"tstart_max":tstart_max, 'mpeak_min':mpeak_min, 'times':times[i],
                         "aexp":aexp[i],"m200c":m200c[i],"i":i,"mah_names":mah_names[i]}
            input_stuff.append(temp_dict)

        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
            results = list(tqdm(executor.map(single_tmdlmdlt,input_stuff), total = len(input_stuff)))

        all_output = np.array(list(results))
  
        all_i = all_output[:,0]
        #we filter for the -99s
        good_index = list(all_i[all_i != -99])
        good_files = list(all_output[:,1][all_i != -99])
        tmdlmdlt = list(all_output[:,2:][all_i != -99])

    else:
        tmdlmdlt = []
        #good_index is list of index numbers in mah file name list that correspond to good halos
        #aka satisfy the condition of tstart_max and mpeak_min
        good_index = []
        #same as good_index but just the corresponding list of file names
        good_files = []

        for i in range(len(times)):

            t_i = times[i]
            aexp_i = aexp[i]
            m200c_i = m200c[i] #array of m200c halo masses

            #in the reduce_data function, the above have already been filtered for bad times.
            mpeak_i = np.max(m200c_i)

            if mpeak_i > mpeak_min and np.min(t_i) < tstart_max:
                good_index.append(i)
                good_files.append(mah_names[i])
        
                lt, lmv = np.log(t_i), np.log(m200c_i/m200c_i[-1])
        
                spl = UnivariateSpline(lt, lmv, s=0)
                spltd = spl(lt)
                #we constrain the spline to positive slopes only
                for ih, _ in enumerate(spltd[1:]):
                    if spltd[ih+1] < spltd[ih]:
                        spltd[ih+1] = spltd[ih]

                dlogmdlogt = np.diff(spltd) / np.diff(lt)
                middle_t = (t_i[1:] + t_i[:-1])*0.5

                dlmdltspl = UnivariateSpline(middle_t, dlogmdlogt, s=0.)  

                astart = aexp_i[1]
                tmdlmdlt.append([t_i[1], m200c_i[1], dlmdltspl, astart])
    #         if mpeak_i > mpeak_min:
    #             temp_stuff.append(mpeak_i)

    # #calculate the fraction of halos that remain after this cut in time.
    # remain_frac = len(good_index)/len(temp_stuff)
    # print_stage("%.2f percent of halos were removed (with applied mpeak cut) with the maximum MAH start time cut of %f Gyr"%(remain_frac*100,tstart_max ))

    if pickle_create == True:
        #we save the lists good_index and tmdlmdlt to a pickle file 
        #a single pickle for a single host 
        #format of storing the data is suite name and its type
        pickle_obj = { "good_index" : good_index, "tmdlmdlt": tmdlmdlt,"good_files":good_files }
        #if pickle with that name already exists. Then first delete existing one and make new now. 
        if os.path.exists(pickleloc):
            os.remove(pickleloc)
            print_stage("Existing Pickle file has been deleted.",ch="-")
        with open(pickleloc, 'wb') as f:
            pickle.dump(pickle_obj, f)
        print_stage("Pickle file is stored here : " + pickleloc,ch = "-")        

    return good_index,tmdlmdlt,good_files


def prepare_mh_tracks(cosmo=None,iniconf=None,suite_name = None):

    mah_data_dir = iniconf['main paths']['mah_data_dir']

    Om0, OmL = cosmo.Om0, cosmo.OmL
    H0 = cosmo.H0 
    args_age = [Om0, OmL]
    iH0 = 9.77814e2 / H0 # 1/H0 in Gyrs
    # convert redshift to scale factor and age grid
    z_arr = np.linspace(0,39,500)
    a_arr = 1 / (1+z_arr)
    t_arr = []
    for _,ai in enumerate(a_arr):
        t_arr.append( iH0 * age(ai, *args_age) )
    t_arr = np.array(t_arr)
        
    #these are the 2 splines that convert time to redshift and vice-a-versa using splines
    tspl = UnivariateSpline(z_arr, t_arr, s=0.)
    zspl = UnivariateSpline(t_arr[::-1], z_arr[::-1], s=0.)


    #we should save this zspl file in a pickle file 
    zspl_loc = iniconf['output paths']['pickle_mah_dir'] + "/zspl.pickle"
    zspl_obj = { "zspl" : zspl}
    with open(zspl_loc, 'wb') as f:
        pickle.dump(zspl_obj, f)
    #this can be read in other scenarios like the chemical model

    #load all the simulation data
    data = reduce_data(mah_data_dir=mah_data_dir,tspl=tspl,iniconf=iniconf,suite_name=suite_name)

    create_pickle = iniconf['run params']['create_pickle']
    pickle_path = iniconf['output paths']['pickle_mah_dir']
    pickle_name = suite_name +"_mahs"
    picklename = pickle_name + ".pickle" 
    total_pickle = pickle_path + "/" + picklename
    my_pickle = Path(total_pickle)


    if my_pickle.is_file():
        #pickle file already exists
        if create_pickle == "True":
            #force to make a new pickle file
            print_stage("New Pickle file is being created. Earlier pickle will be deleted.",ch = "-")
            #run the get_tmdlmdlt function to get good_index and tmdlmdlt lists. Running this also creates the pickle for future use. 
            good_index, tmdlmdlt, good_files = get_tmdlmdlt(data=data,iniconf=iniconf,pickleloc=total_pickle,pickle_create=True)
        else:
            #Will not make new pickle as it already exists and is not forced to make a new one.  
            print_stage("The pickle file already exists.",ch="-")
            with open(total_pickle, 'rb') as f:
                loaded_obj = pickle.load(f)
            good_index = loaded_obj["good_index"]
            good_files = loaded_obj["good_files"]
            tmdlmdlt   = loaded_obj["tmdlmdlt"]
    else:
        #pickle file does not exist therefore will be making one. 
        print_stage("Pickle file is being created",ch = "-")
        #run the get_tmdlmdlt function to get good_index and tmdlmdlt lists. Running this also creates the pickle for future use. 
        good_index, tmdlmdlt, good_files = get_tmdlmdlt(data=data,iniconf=iniconf, pickleloc = total_pickle, pickle_create = True)
    
    #good index contains the indices of the halo that satisfy our cuts. Namely that Mpeak > 1e7 and halo track starts before 4 Gyr. 
    #This list is useful for cross-referencing with original data and also selecting appropriate survival probabilties. They correspond to the original indices in ELVIS data set. 

    print_stage("This pickle is used for MAH data: " + total_pickle, ch="-")


    return data, good_index, tmdlmdlt, zspl, good_files
