## this script reads Elvis MAHs and arranges them in lists that can be easily read.

import numpy as np
from age_integrate import age
from scipy.interpolate import UnivariateSpline
import logging
import glob
import pandas as pd

def E2(z):
    """E^2(z) where E(z) is dimensionless Hubble function"""
    return (0.266*(1+z)**3 + 0.734)

def rho_crit(z): 
    """
    critical density of the universe at redshift z in Msun/kpc^3
    assuming H0 = 71 km/s/Mpc - the value of the ELVIS simulations
    """
    return 139.894 * E2(z)

def f_nfw(x): 
    assert(x.any() != -1.)  
    return np.log(1.+x) - x / (1.+x)

pi34 = 3./(np.pi * 4.)
l200 = np.log10(200.)


def check_column_names(df=None,mh_file = None,iniconf=None):

    m200c_exists = iniconf['track format']['m200c_exists']

    all_key_names = ['scale','mvir','x_pos','y_pos','z_pos','vmax','rvir','ID','upID','rs','vx','vy','vz']

    all_keys = []

    for kyi in all_key_names:
        all_keys.append(iniconf['track format'][kyi])

    if m200c_exists == "True":
        all_key_names.append("m200c")
        m200c_key = iniconf['track format']['m200c']
        all_keys.append(m200c_key)
    #we confirm if wach of these columns exist in the dataframe

    #the keys of the dataframe
    df_keys = df.keys()

    for ki in all_keys:
        if ki in df_keys:
            pass
        else:
            raise ValueError('The key "%s" does not exist in the file %s'%(ki,mh_file)) 

    return 



def get_subsample_scales(org_scales, tar_diff = 0.014):
    final_scales_inds = []
    final_scales_vals = []

    final_scales_inds.append(0)
    final_scales_vals.append(1)

    for i in range(100):
        new_si = 1 - tar_diff*i
        #we find the nearest ind 

        use_si = float(org_scales[np.argmin(np.abs(org_scales-new_si))])
        si_ind = int(np.where(org_scales == use_si)[0])

        if new_si < 0 or use_si < 0:
            break
            
        elif use_si == final_scales_vals[-1]:
            pass
        else:
            final_scales_inds.append(si_ind)
            final_scales_vals.append(use_si)


    return np.array(final_scales_inds)




def reduce_data(mah_data_dir=None,tspl=None,iniconf=None,suite_name=None):
    '''
    Read all the MAH tract data and compile it all into a single dictionary
    '''

    all_m200c =[]
    all_mvir = []
    all_names = []
    all_aexp = []
    all_rvir = []
    all_rs = []
    all_upID = []
    all_ID = []
    all_times = []
    all_xc = []
    all_yc = []
    all_zc = []
    all_vmax = []
    all_vx = []
    all_vy = []
    all_vz = []


    subsample = iniconf["track format"]["track_subsample"]
    
    if subsample == "None":
        subsample_val = 1
    else:
        subsample_val = int(subsample)

    #get a list of all mah csv files
    mah_lists = sorted(glob.glob(mah_data_dir + "/" + suite_name + "/*.csv"))
    
    #I am doing a temporary sorting here. Will need to remove this in final verison

    mah_lists = np.array(mah_lists)

    ############################################################################################
    ############################################################################################
    ############################################################################################

    #this sorting function will work for any track file that is of the format "blah/..blah/blah_123.csv".
    #That is, the number '123' is at the very end of file name separated by an underscore
 
    all_nums = []
    for alti in mah_lists:
        alti_back = alti[::-1]
        ind = alti_back.find("_")
        number = int(alti[-1*ind:].replace(".csv",""))
        all_nums.append(number)

    satellite_modeling = iniconf["run params"]["satellite_modelling"]

    #we sort this now
    sort_inds_track = np.argsort(all_nums)
    mah_lists = mah_lists[sort_inds_track]
    # all_nums = [int(mhi.replace("/Users/radioactive/Desktop/temp_dat/mah_data_","").replace(".csv","")) for mhi in mah_lists]
    # sort_inds = np.argsort(all_nums)    
    # mah_lists = mah_lists[sort_inds]

    ############################################################################################
    ############################################################################################
    ############################################################################################

    if len(mah_lists) == 0:
        raise ValueError("No halo track data files were found in %s/ with '.csv' extension. Make sure correct path has been entered for 'data paths:mah_data_dir'"%(mah_data_dir))

    for i,mhi in enumerate(mah_lists):
        # #we find the last dot
        # mhi_inv = mhi[::-1]
        # find_dot = len(mhi) - mhi_inv.find(".") - 1 #this is the index position of the last dot
        # temp = mhi[:find_dot]
        all_names.append(mhi.replace(mah_data_dir+ "/" + suite_name +  "/","").replace(".csv","").replace("/","")  )
 
        track_data = pd.read_csv(mhi)


        #run the column check now
        check_column_names(df=track_data,mh_file = mhi,iniconf=iniconf)


        scale_key = iniconf['track format']["scale"]
        mvir_key = iniconf['track format']["mvir"]
        x_key = iniconf['track format']["x_pos"]
        y_key = iniconf['track format']["y_pos"]
        z_key = iniconf['track format']["z_pos"]
        vx_key = iniconf['track format']["vx"]
        vy_key = iniconf['track format']["vy"]
        vz_key = iniconf['track format']["vz"]
        vmax_key = iniconf['track format']["vmax"]
        ID_key = iniconf['track format']["ID"]
        upID_key = iniconf['track format']["upID"]
        rvir_key = iniconf['track format']["rvir"]
        rs_key = iniconf['track format']["rs"]

        sub_sample_inds = get_subsample_scales(np.array(track_data[scale_key]), tar_diff = 0.014)

        #we can probably make this into pandas or something to avoid confusion between columns

        aexp_i = np.array(track_data[scale_key])[sub_sample_inds]
        mvir_i = np.array(track_data[mvir_key])[sub_sample_inds]
        rvir_i = np.array(track_data[rvir_key])[sub_sample_inds]
        rs_i = np.array(track_data[rs_key])[sub_sample_inds]
        upID_i = np.array(track_data[upID_key])[sub_sample_inds]
        ID_i = np.array(track_data[ID_key])[sub_sample_inds]
        xc_i = np.array(track_data[x_key])[sub_sample_inds]
        yc_i = np.array(track_data[y_key])[sub_sample_inds]
        zc_i = np.array(track_data[z_key])[sub_sample_inds]
        vxc_i = np.array(track_data[vx_key])[sub_sample_inds]
        vyc_i = np.array(track_data[vy_key])[sub_sample_inds]
        vzc_i = np.array(track_data[vz_key])[sub_sample_inds]
        vmax_i = np.array(track_data[vmax_key])[sub_sample_inds]

        # if i < 10:
        #     print(np.diff(aexp_i),len(aexp_i))


        # aexp_i = np.array(track_data[scale_key])[::subsample_val]
        # mvir_i = np.array(track_data[mvir_key])[::subsample_val]
        # rvir_i = np.array(track_data[rvir_key])[::subsample_val]
        # rs_i = np.array(track_data[rs_key])[::subsample_val]
        # upID_i = np.array(track_data[upID_key])[::subsample_val]
        # ID_i = np.array(track_data[ID_key])[::subsample_val]
        # xc_i = np.array(track_data[x_key])[::subsample_val]
        # yc_i = np.array(track_data[y_key])[::subsample_val]
        # zc_i = np.array(track_data[z_key])[::subsample_val]
        # vxc_i = np.array(track_data[vx_key])[::subsample_val]
        # vyc_i = np.array(track_data[vy_key])[::subsample_val]
        # vzc_i = np.array(track_data[vz_key])[::subsample_val]
        # vmax_i = np.array(track_data[vmax_key])[::subsample_val]

    
        if satellite_modeling == "True":

            if i == 0:
                #this is the host stuff
                # host_scales = np.array(track_data[scale_key])[::subsample_val]
                host_scales = np.array(track_data[scale_key])[sub_sample_inds]

                scale_min_host = np.min(host_scales[host_scales>0])

                # host_xc = np.array(track_data[x_key])[::subsample_val]
                # host_yc = np.array(track_data[y_key])[::subsample_val]
                # host_zc = np.array(track_data[z_key])[::subsample_val]
                host_xc = np.array(track_data[x_key])[sub_sample_inds]
                host_yc = np.array(track_data[y_key])[sub_sample_inds]
                host_zc = np.array(track_data[z_key])[sub_sample_inds]

        #we filter all data for valid scale factors

            scale_min_sub = np.min(aexp_i[aexp_i>0])

            scale_start = np.maximum(scale_min_sub,scale_min_host)
            #we filter using this scale factor now
            host_xc_f = host_xc[host_scales > scale_start]
            host_yc_f = host_yc[host_scales > scale_start]
            host_zc_f = host_zc[host_scales > scale_start]


            aexp_if = aexp_i[aexp_i > scale_start ]
            mvir_if = mvir_i[aexp_i > scale_start ]
            rvir_if = rvir_i[aexp_i > scale_start ] 
            rs_if = rs_i[aexp_i > scale_start ]
            upID_if = upID_i[aexp_i > scale_start ] 
            ID_if = ID_i[aexp_i > scale_start ] 

            #shifting the reference frame such that host is in center
            xc_if = xc_i[aexp_i > scale_start ] - host_xc_f
            yc_if = yc_i[aexp_i > scale_start ] -  host_yc_f
            zc_if = zc_i[aexp_i > scale_start ] - host_zc_f
            vxc_if = vxc_i[aexp_i > scale_start ]
            vyc_if = vyc_i[aexp_i > scale_start ]
            vzc_if = vzc_i[aexp_i > scale_start ]
            vmax_if = vmax_i[aexp_i > scale_start ]
                
        else:
            aexp_if = aexp_i[aexp_i > 0]
            mvir_if = mvir_i[aexp_i > 0 ]
            rvir_if = rvir_i[aexp_i > 0 ] 
            rs_if = rs_i[aexp_i > 0 ]
            upID_if = upID_i[aexp_i > 0] 
            ID_if = ID_i[aexp_i > 0 ] 

            #shifting the reference frame such that host is in center
            xc_if = xc_i[aexp_i > 0]
            yc_if = yc_i[aexp_i > 0 ]
            zc_if = zc_i[aexp_i > 0 ]
            vxc_if = vxc_i[aexp_i > 0 ]
            vyc_if = vyc_i[aexp_i > 0 ]
            vzc_if = vzc_i[aexp_i > 0 ]
            vmax_if = vmax_i[aexp_i > 0 ]

        #convert scale factor to redshift
        zred_if = (1/aexp_if) - 1
        #convert redshift to time using array
        t_if = tspl(zred_if)
        #let us sort th time array in proper order, that is, in ascending order
        sort_inds = np.argsort(t_if)

        aexp_ifs = aexp_if[sort_inds]
        mvir_ifs = mvir_if[sort_inds]
        rvir_ifs = rvir_if[sort_inds]
        rs_ifs = rs_if[sort_inds]
        upID_ifs = upID_if[sort_inds]
        ID_ifs = ID_if[sort_inds]
        zred_ifs = zred_if[sort_inds]
        t_ifs = t_if[sort_inds]
        xc_ifs = xc_if[sort_inds]
        yc_ifs = yc_if[sort_inds]
        zc_ifs = zc_if[sort_inds]
        vxc_ifs = vxc_if[sort_inds]
        vyc_ifs = vyc_if[sort_inds]
        vzc_ifs = vzc_if[sort_inds]
        vmax_ifs = vmax_if[sort_inds]


        #the above are sorted in ascending order of time

        #we do the below mvir to m200c only if needed
        m200c_exists = iniconf['track format']['m200c_exists']


        if m200c_exists == "False":

            #convert mvir to m200c
            m200c_ifs = np.zeros_like(zred_ifs)

            for j, zj in enumerate(zred_ifs):
                if mvir_ifs[j] <= 0.:
                    #halo is undefined.
                    m200c_ifs[j] = -99.
                else:
                    #halos is defined
                    # NFW mass profile 
                    fdum = f_nfw( rvir_ifs[j] / rs_ifs[j] )
                    rhoc = rho_crit(zj)
                    rr = np.linspace(np.log10(1.e-1*rvir_ifs[j]), np.log10(rvir_ifs[j]), 100)
                    mr = mvir_ifs[j] * f_nfw(10.**rr/rs_ifs[j]) / fdum
                    deltar = pi34 * mr *(1.+zj)**3 / (rhoc * 10.**(3.*rr)) # (1+z)^3 factor is needed because Rvir is comoving
                    mrsp = UnivariateSpline(np.log10(deltar[::-1]), np.log10(mr[::-1]), s=0.)
                    m200c_ifs[j] = 10.**mrsp(l200)

        else:
            #in this case, the m200c column should already exist
            m200c_key = iniconf['track format']["m200c"]

            # m200c_i = np.array(track_data[m200c_key])[::subsample_val]
            m200c_i = np.array(track_data[m200c_key])[sub_sample_inds]

            m200c_if = m200c_i[aexp_i > 0]
            m200c_ifs = m200c_if[sort_inds]



        if len(aexp_ifs) != len(ID_ifs) or len(rvir_ifs) != len(m200c_ifs):
            raise ValueError("The different array lenghts for track data do not match.")

        #append all this to lists
        all_aexp.append(aexp_ifs)
        all_mvir.append(mvir_ifs)
        all_vmax.append(vmax_ifs)
        all_rvir.append(rvir_ifs)
        all_rs.append(rs_ifs) 
        all_upID.append(upID_ifs)
        all_ID.append(ID_ifs)
        all_times.append(t_ifs)
        all_m200c.append(m200c_ifs)
        all_xc.append(xc_ifs)
        all_yc.append(yc_ifs)
        all_zc.append(zc_ifs)
        all_vx.append(vxc_ifs)
        all_vy.append(vyc_ifs)
        all_vz.append(vzc_ifs)

    #this is the dictionary that contains all the useful info
    data = {'times': all_times,'rvir': all_rvir, 'rs': all_rs ,'aexp': all_aexp,'mvir': all_mvir, 'vmax':all_vmax,
            'upID': all_upID, 'ID': all_ID, 'm200c':all_m200c,'mah_name':all_names, 'x':all_xc,'y':all_yc,'z':all_zc,
            'vx':all_vx,'vy':all_vy,'vz':all_vz }


    ##save the all_names as a column text file in. Will save them in the fsps_files folder.
    # all_names_file_path = iniconf['output paths']['fsps_input_dir'][:-5] + "/all_names.txt"
    # print("=="*10)
    # print(all_names_file_path)
    # print("=="*10)

    # np.savetxt(all_names_file_path,all_names,fmt = "%s")

    return data



def unpack_data(data,types=None):
    '''
    Parameters:
    -----------
        data: dictionary, dictionary for all the track information
        types: list, list of strings of all required info. Eg. ['x','y','host_ID']

    Returns:
    -----------
        output: list, contains the relevant dictionary keys
    '''

    output = []
    for ti in types:
        output.append(data[ti])
    return output



