## this script just takes in the final dataframe and will plot some important summary plots
## Namely, the SHMR, Stellar-Mass and metallicity relation, half mass radius, the luminosity function (comparison with the weighted one observed from Drlica Wagner et. al. 2020)

import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import numpy as np
import matplotlib.backends.backend_pdf
from astropy.io import fits
#import matplotlib.patheffects as pe
#from pathlib import Path
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from os import listdir
from os.path import isfile, join
#import os
import pandas as pd

######################################################################
######################################################################

# from codes.plot_utils import plot_pretty
def plot_pretty(dpi=175,fontsize=9):
    # import pyplot and set some parameters to make plots prettier
    import matplotlib.pyplot as plt

    plt.rc("savefig", dpi=dpi)
    plt.rc("figure", dpi=dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [0.5, 1.1])
    return

plot_pretty()


#Relevant functions for plotting data

def fB13(x, alfa, delta, gamma):
    dummy = -np.log10(10.**(alfa*x)+1.) + delta*(np.log10(np.exp(x)+1.))**gamma/(1.+np.exp(10.**(-x)))
    return dummy

def MsMh_B13form(lMvir, lM1, leps, alfa, delta, gamma):
    lMsB13 = leps + lM1 + fB13(lMvir-lM1,alfa, delta, gamma) - fB13(0.,alfa, delta, gamma)
    return lMsB13


#this is just used to convert observational data.
MsLV = 2.0
lMsLV = np.log10(MsLV)
MVSun = 4.80

def Ms_to_MV(Ms):
    """
    convert input stellar mass Ms (in Msun) to V-band absolute magnitude
    """
    return 2.5*(lMsLV - np.log10(Ms)) + MVSun   

def MV_to_lMs(M_V):
    """
    convert input V-band absolute magnitude M_V to log10 of stellar mass (in Msun)
    """
    return -0.4*(M_V - MVSun) + np.log10(MsLV) 

# plot observational trend from Maiolino    
def Zmaiolino08(lMs,lM0,K0):
    """
    best fit relation from table in Maiolino et al. 2008 
    using consistent Z calibration
    """
    return -0.0864*(lMs-lM0)**2 + K0


def read_torrealba18(fname):
    f = open(fname, 'r')
    rht = []; Mst = []
    gname = []
    for il, line in enumerate(f):
        if il > 0:
            ld = line.strip()
            cols = ld[17:].split()
            not_cluster = (cols[0] != 'c') and (cols[0] != 'g/c') and (cols[0] != 'c/g') and\
                          (cols[0] != 'c?')
            if  (cols[5] != '-') and (cols[8] != '-') and not_cluster:
                galname = ld[0:16].strip()
                gname.append(galname)
                MV = float(cols[5])
                rhpc = float(cols[8]) 
                rht.append(rhpc)
                # 4.80 is absolute AB system magnitude of the Sun in V
                # see http://mips.as.arizona.edu/~cnaw/sun.html
                Mst.append(10.**MV_to_lMs(MV))
                #print dmod, dSun, rh, LV
            
    return np.array(gname), np.array(rht), np.array(Mst)



#############################
#PLOT 1 : SHMR
#############################

def plot_SHMR(Mpeak=None,Msf=None,obs_data_path=None,fig_pdf=None):
    # load data for individual galaxies from Read et al. 2017 and Read & Erkal 2019
    msre, m200re = np.loadtxt(obs_data_path+'/read_erkal19.dat', usecols=(4,9), unpack=True)
    msre *= 1e6; m200re *= 1e9
    msr, m200r = np.loadtxt(obs_data_path+'/read_etal17.dat', usecols=(3,8), unpack=True)
    msr *= 1e7; m200r *= 1e10
    # read data for M*-Mhalo constraints from Fig 6 in Nadler et al. 2020
    lmpn, lmsn = np.loadtxt(obs_data_path+'/nadler_etal_fig6.dat', usecols=(0,1), unpack=True)
    lmpn1s, lmsn1s1, lmsn1s2 = np.loadtxt(obs_data_path+'/nadler_etal_fig6_1sig.dat', usecols=(0,1,2), unpack=True)
    lmpn2s, lmsn2s1, lmsn2s2 = np.loadtxt(obs_data_path+'/nadler_etal_fig6_2sig.dat', usecols=(0,1,2), unpack=True)

    lM1 = 11.39; leps = -1.685; alfa = -2.; delta = 4.335; gamma = 0.531
    lMpe = np.linspace(7.5, 13, 200)
    lMsK18 = MsMh_B13form(lMpe, lM1, leps, alfa, delta, gamma)

    fig = plt.figure(figsize=(3, 3))
    plt.xlabel(r'$M_h\ \rm (M_\odot)$')
    plt.ylabel(r'$M_{\star}\ \rm (M_\odot)$')
    plt.xlim(1.e7,5.e12); plt.ylim(1.e2,5.e11)
    plt.xscale('log'); plt.yscale('log')
    x = [1.e9, 1.e12]; y = [1.e6, 10.**11.1]
    plt.scatter(Mpeak, Msf, marker=".", s=10, c='orangered', label=r'$z=0$')
    plt.fill_between(10.**lmpn2s, 10.**lmsn2s1, 10.**lmsn2s2, color='gray', alpha=0.2)
    plt.fill_between(10.**lmpn1s, 10.**lmsn1s1, 10.**lmsn1s2, color='gray', alpha=0.4)
    plt.scatter(10**lmpn, 10**lmsn, marker="o", s=10, c='magenta', label='DES + PS1')
    plt.scatter(m200re, msre, marker="o", s=20, c='slateblue', edgecolor='darkslateblue', label=r'Read \& Erkal 2019')
    plt.scatter(m200r, msr, marker="p", s=20, c='mediumpurple', edgecolor='darkslateblue', label=r'Read et al. 2017')
    plt.plot(x, y, '--', c='lightgray')
    plt.plot(10.**lMpe, 10.**lMsK18 )
    plt.grid(linestyle=':', c='lightgray',alpha = 0.5)
    plt.legend(frameon=False, loc='upper left',fontsize = 7)
    plt.tight_layout()
    #fig.savefig(image_path + "/" + figname)
    #plt.savefig(fig_name, bbox_inches='tight')
    fig_pdf.savefig(fig)

    return
 
#############################
#PLOT 2 : MSTAR VS. Z RELATION
#############################

def plot_massZ(Msf=None,MZsf=None,Mgf=None,MZgf=None,MHIf=None,obs_data_path=None,Zsun=None,fig_pdf=None):

    #WE ARE LATER GOING TO HAVE TO MODIFY THIS BY COLORING POINTS THAT HAVE H2 GAS OR NOT. 

    fig = plt.figure(figsize=(3, 3))
    ylabel = r'$Z_\star, Z_{\rm gas}\ (Z_\odot)$'; xlabel = r'$M_{\star}\ \rm (M_\odot)$'
    xlims = [1.e2, 8.e11]; ylims = [1.e-3,  10.]
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.xscale('log'); plt.yscale('log')
    plt.xlim(xlims); plt.ylim(ylims)

    # solar obundance of Oxygen from Asplund et al. 2009, ARAA
    OHsol = 8.68

    zcolors = ['mediumorchid', 'red', 'blue', 'darkgray']
    zmaio = np.array([0.07, 0.7, 2.2, 3.5])
    lM0 = np.array([11.18, 11.57, 12.38, 12.76])
    K0 = np.array([9.04, 9.04, 8.99, 8.79])
    nzplot = 3

    lMs = np.arange(9.0, 11.0, 0.1)
    #for i, zm in enumerate(zmaio[:nzplot]): 
    #    Zmaio = Zmaiolino08(lMs, lM0[i], K0[i]) - OHsol
    #    plt.plot(10.**lMs, 10.**Zmaio, '--', c=zcolors[i], lw=3.0, label=r'${\rm M08}, z=%.1f$'%zm)

    # plot Lee et al. 2006
    lMsL = np.linspace(5.5, 9.5, 100)
    lZLp, elZLp, lMsLp = np.loadtxt(obs_data_path+'/lee06.txt',usecols=(1,3,4),unpack=True)
    lZLp = lZLp - OHsol

    lmsb, lohb = np.loadtxt(obs_data_path+'/berg_etal16_tab1.dat',usecols=(3,7),unpack=True)
    lohb = lohb - OHsol

    lohk, lmsk = np.loadtxt(obs_data_path+'/kojima20_tab7.dat',usecols=(1,6),unpack=True)
    lohk = lohk - OHsol

    lLVk, elLVk, lZk, elZk  = np.loadtxt(obs_data_path+'/kirby13_tab4.txt',usecols=(4,5,6,7),unpack=True)

    MVs,  lZs  = np.loadtxt(obs_data_path+'/simon19_tab1.dat',usecols=(1,17),unpack=True)
    lMss = MV_to_lMs(MVs)

    plt.scatter(10**lMsLp,10.**lZLp, s=30, marker='o', c='cyan',
                edgecolor='darkslateblue', label=r'$\rm Lee\ et\ al.\ 06$')
    plt.scatter(10**lmsb,10.**lohb, s=30, marker='o', c='deepskyblue',
                edgecolor='darkslateblue', label=r'$\rm Berg\ et\ al.\ 16$')
    plt.scatter(10**lmsk,10.**lohk, s=30, marker='p', c='skyblue',
                edgecolor='darkslateblue', label=r'$\rm Kojima\ et\ al.\ 20$')

    msL, emsLp, emsLm = 7.3e5, 2.2e5, 4.3e5 # McQuinn+ 2020 
    OHL, eOHL = 7.12, 0.04
    ZgL, eZgLp = OHL-OHsol, eOHL 

    msLP = 5.6e5 # McQuinn et al. 2015
    OHLP = 7.17
    ZgLP, eZgLP = OHLP - OHsol, 0.04

    plt.scatter(msLP, 10.**ZgLP, s=25, marker='p', c='deepskyblue', edgecolor='darkslateblue',label=r'$\rm Leo\ P,\ McQuinn\ et\ al.\ 15$')
    plt.scatter(msL, 10.**ZgL, s=60, marker='*', c='deepskyblue', edgecolor='darkslateblue',label=r'$\rm Leoncino,\ McQuinn\ et\ al.\ 20$')
    # plt.scatter(Msf, (MZgf)/Mgf/Zsun, marker=".", s=10, alpha=0.5, c='g', label=r'$Z_{\rm gas}\ z=0$')

    plt.scatter(Msf[MHIf<1], (MZgf[MHIf<1])/Mgf[MHIf<1]/Zsun, marker="o", s=10, alpha=0.05, c='gray', label=r'$Z_{\rm gas}\ z=0$, no HI')
    plt.scatter(Msf[MHIf>1], (MZgf[MHIf>1])/Mgf[MHIf>1]/Zsun, marker=".", s=10, alpha=0.5, c='g', label=r'$Z_{\rm gas}\ z=0$')

    plt.scatter(10**lLVk, 10.**lZk, s=35, marker='p', c='mediumpurple', edgecolor='darkslateblue',label=r'$\rm Kirby\ et\ al.\ 13$')
    plt.scatter(10**lMss, 10.**lZs, s=20, marker='s', c='mediumpurple', edgecolor='darkslateblue',label=r'$\rm Simon\ 19$')
    plt.scatter(Msf, MZsf/Msf/Zsun, marker=".", s=10, c='orangered', label=r'$Z_\star\ z=0$')
    plt.legend(frameon=False, loc='upper left',fontsize=4,ncol = 1,scatterpoints=1)
    plt.grid(ls=':', c='lightgray', which='both',alpha = 0.5)
    # plt.savefig(image_path +  "/" + figname, bbox_inches='tight')
    plt.tight_layout()
    fig_pdf.savefig( fig )

    return



#############################
#PLOT 3: HALF MASS RADIUS 
#############################

def plot_sizes(Msf=None, rsizes=None, obs_data_path=None, fig_pdf=None):

    fname = obs_data_path+'/lg_sats_torrealba_v3.txt'
    f = open(fname, 'r')
    galname_m15, rhm15, Msm15 = read_torrealba18(fname)
    rhm15 *= 1.e-3

   
    flag3d = True

    fig, ax = plt.subplots(1,1, figsize=(4,4))
    ylim = [0.005,30.]; xlim = [1.e2, 5e11]
    ax.set_ylim(ylim); ax.set_yscale('log')
    ax.set_xlim(xlim); ax.set_xscale('log')
    ax.set_xlabel(r'$M_\star\ (M_\odot)$'); 
    if flag3d:
        ax.set_ylabel(r'$r_{\rm 50, 3D}\ (\mathrm{kpc})$')
    else:
        ax.set_ylabel(r'$R_{\rm 50, 2D}\ (\mathrm{kpc})$')

    plt.scatter(Msf, rsizes, marker='.', s=10, c='orangered', alpha=0.75, label=r'${\rm model}$')
    
    if flag3d: 
        rhm15_3d = 1.34 * rhm15
        ax.scatter(Msm15, rhm15_3d, s=85, c='mediumpurple', edgecolors='darkslateblue', 
            marker='*', label=r'$\rm LG\ dSph$')
    else:
        ax.scatter(Msm15, rhm15, s=85, c='mediumpurple', edgecolors='darkslateblue', 
            marker='*', label=r'$\rm LG\ dSph$')

    # new value from Vasily Belokurov (18/05/2019)
    lgMsAntII = np.log10(8.8e5)
    rhAII, erhAII = 2.92, 0.312 # 2D sizes in kpc  

    # Bootes IV from Homma et al. https://arxiv.org/abs/1906.07332
    M_V = -4.53
    lgMsBIV = MV_to_lMs(M_V)
    rhBIV, erhBIV = 0.462, 0.1 # 2D sizes in kpc  
    if flag3d: # correct size for projection effect, if 3D 
        rhAII, erhAII = 1.34*rhAII, 1.34*erhAII
        rhBIV, erhBIV = 1.34*rhBIV, 1.34*erhBIV        
    ax.errorbar(10.**lgMsAntII, rhAII, c='purple', xerr = 0.001, yerr = erhAII)
    ax.scatter(10.**lgMsAntII, rhAII, marker='p', s=75, c='mediumpurple',  edgecolors='purple', label=r'$\rm Antlia\ II$')
    ax.errorbar(10.**lgMsBIV, rhBIV, c='purple', xerr = 0.001, yerr = erhBIV)
    ax.scatter(10.**lgMsBIV, rhBIV, marker='o', s=50, c='mediumpurple',  edgecolors='purple', label=r'$\rm Bo\ddot{o}tes\ IV$')
    #-------------------------------------------------------
    # constant surface brightness limit line for LG dwarfs
    # MV,Sun(AB) = 4.80
    muVlim = 31
    Sb = 10.**(-0.4*(muVlim - 21.5721 - 4.80))
    Msp = 10.**(np.linspace(2,5.6,50))
    rp = 1.e-3 * np.sqrt(Msp/MsLV/np.pi/Sb)

    plt.plot(Msp, rp, '--', c='darkred', alpha=0.5, lw = 1., label=r'$\mu_V=%.1f$'%muVlim)

    #------------------------------------------------------
    ax.legend(frameon=False, loc="lower right", ncol=1, scatterpoints=1, fontsize=7)
    plt.grid(ls=':', c='lightgray', which='both',alpha = 0.5)
    # plt.savefig(image_path +  "/" + figname, bbox_inches='tight')
    plt.tight_layout()
    fig_pdf.savefig( fig )

    return


#############################
#PLOT 5: Depletion Time Scale, sSFR, SFR vs. Mstar
#############################


def plot_sfr(Msf=None, Mgf=None, sfrf=None, MHIf=None, obs_data_path=None,
             fig_pdf=None):


    url = obs_data_path + '/durbala2020-table2.21-Sep-2020.fits'
    aa = fits.getdata(url)

    #msaa = 10**aa['logMstarMcGaugh']#aa['logMstarGSWLC']#aa['logMstarTaylor']
    msaa = 10**aa['logMstarTaylor']
    mhaa = 1.33*10.**aa['logMH']
    sfraa = 10**aa['logSFRNUVIR']#aa['logSFR22'] #aa['logSFRNUVIR']


    mgms_sh, fg_sh, mb_sh, ms_sh, sfr_sh = np.loadtxt(obs_data_path+'/shield_I_II.dat', 
                                                    usecols=(1, 4, 7, 16, 19), 
                                                    unpack=True)

    sfr_sh *= 1.e-3 # convert to Msun/yr
    mg_sh = 10.**mb_sh[ms_sh>0] - 10**ms_sh[ms_sh>0]
    sfr_sh = sfr_sh[ms_sh>0]
    ms_sh = 10.**ms_sh[ms_sh>0]
    mb_sh = 10**mb_sh

    lsfr_shade, lms_shade, lmb_shade = np.loadtxt(obs_data_path+'/shade.dat.txt', 
                                                  usecols=(9,10,11), unpack=True)
    ms_shade = 10**lms_shade
    mg_shade = 10**lmb_shade - ms_shade

    lms_mq19, sfr_mq19 = np.loadtxt(obs_data_path + '/mcquinn_etal19_tab5.dat', 
                                    usecols=(1,9), unpack=True)
    ms_mq19 = 10.**lms_mq19

    lms_j15, lsfr_j15 = np.loadtxt(obs_data_path + '/jimmy_etal15_tab1.dat', 
                                   usecols=(5,11), unpack=True)
    ms_j15 = 10.**lms_j15

    #read both star, gas and sfr 
    lms_mg17, lmg_mg17, lsfr_mg17 = np.loadtxt(obs_data_path+'/mcgaugh_etal17_tab1.dat', 
                                               usecols=(2,3,5), unpack=True)
    ms_mg17 = 10.**lms_mg17
    mg_mg17 = 10.**lmg_mg17

    #lms_j17, lsfr_j17 = np.loadtxt( obs_data_path + '/james_etal17_tab5.dat', 
    #                            usecols=(1,3), unpack=True)
    #ms_j17 = 10.**lms_j17

    #lms_t16, lsfr_t16 = np.loadtxt(obs_data_path + '/teich_etal16_tab13.dat', 
    #                               usecols=(5, 1), unpack=True)
    #ms_t16 = 10.**lms_t16


    #this is plot for depletion time scale 
    fig = plt.figure(figsize=(3.,3.))
    plt.xscale('log'); plt.yscale('log')
    plt.xlim([1.e5,1.e11]); plt.ylim([0.01, 1000.])
    plt.ylabel(r'$\tau_{\rm dep}\ (\rm Gyrs)$'); plt.xlabel(r'$M_{\star}\ {(M_\odot)}$')

    plt.scatter(msaa, 1e-9*mhaa/sfraa, marker='.',c='m',s=5.,
                alpha=0.9, edgecolor='none', label=r'$\mathrm{ALFALFA}$')

    plt.scatter(ms_mg17, 1.e-9*ms_mg17/10**lsfr_mg17, marker='p',c='slateblue',s=15.,
                alpha=0.9,edgecolor='darkslateblue', label=r'$\mathrm{McGaugh+\ 17}$')

    plt.scatter(ms_sh, 1.e-9*mg_sh/sfr_sh, marker='o',c='m',s=15.,
                alpha=0.9,edgecolor='slateblue', label=r'$\mathrm{SHIELD I\&II, McQuinn+\ 20}$')
            

    plt.scatter(Msf[MHIf>1], 1.e-9*MHIf[MHIf>1]/sfrf[MHIf>1], marker='.', s=30, c='orangered', 
                alpha=0.75, label=r'${\rm model\ }\tau_{\rm dep} =M_{\rm HI}/{\rm\ SFR}$')
    plt.scatter(Msf[MHIf<1], 1.e-9*Mgf[MHIf<1]/sfrf[MHIf<1], marker='.', s=10, c='gray', 
                alpha=0.07, label=r'${\rm model\ }\tau_{\rm dep} = M_{\rm gas}/{\rm\ SFR}$')

    # plt.legend(frameon=False,loc='upper left', bbox_to_anchor=(0., 1.25), ncol=2, fontsize=5)
    plt.legend(frameon=True,loc='lower left', ncol=1, fontsize=3.5)
    plt.grid(ls=':', c='lightgray', which='major',alpha = 0.5)
    plt.tight_layout()
    fig_pdf.savefig(fig)


    #this is plot for sSFR vs. Mstar
    fig = plt.figure(figsize=(3.,3.))
    plt.xscale('log'); plt.yscale('log')
    plt.xlim([1.e5,1.e11]); plt.ylim([1.e-14,1.e-7])
    plt.ylabel(r'$\rm sSFR\ (yr^{-1})$'); plt.xlabel(r'$M_{\star}\ {(M_\odot)}$')

    plt.scatter(msaa, sfraa/msaa, marker='.', c='m', s=5.,
                alpha=0.9, edgecolor='none', label=r'$\mathrm{ALFALFA}$')

    plt.scatter(ms_mg17, 10**lsfr_mg17/ms_mg17, marker='p', c='slateblue',
                s=15., alpha=0.9, edgecolor='darkslateblue', 
                label=r'$\mathrm{McGaugh+\ 17}$')

    plt.scatter(ms_sh, sfr_sh/ms_sh, marker='o', c='m', s=15.,
                alpha=0.9, edgecolor='slateblue', 
                label=r'$\mathrm{SHIELD I\&II, McQuinn+\ 20}$')

    plt.scatter(Msf[MHIf>1], sfrf[MHIf>1]/Msf[MHIf>1], marker='.', s=40, 
                c='orangered', alpha=0.75, 
                label=r'${\rm model\ SFR=}M_{\rm H_2}/2\,\rm Gyr$')
    plt.scatter(Msf[MHIf<1], sfrf[MHIf<1]/Msf[MHIf<1], marker='.', s=40,  
                c='gray', alpha=0.07, label=r'${\rm model,\ no\ HI}$')


    # plt.legend(frameon=False,loc='upper left', bbox_to_anchor=(0., 1.25), ncol=2, fontsize=5)
    plt.legend(frameon=True,loc='lower right', ncol=1, fontsize=3.5)
    plt.grid(ls=':', c='lightgray', which='major',alpha = 0.5)
    plt.tight_layout()
    fig_pdf.savefig(fig)

    #this is plot for SFR vs. Mstar
    fig = plt.figure(figsize=(3.,3.))
    plt.xscale('log'); plt.yscale('log')
    plt.xlim([1.e5,1.e11]); plt.ylim([1.e-5,1.])
    plt.ylabel(r'$\rm SFR\ (M_\odot\,yr^{-1})$'); 
    plt.xlabel(r'$M_{\star}\ {(M_\odot)}$')

    plt.scatter(msaa, sfraa, marker='.',c='m',s=5.,
                alpha=0.5, edgecolor='none', label=r'$\mathrm{ALFALFA}$')
    plt.scatter(ms_mq19, sfr_mq19, marker='o',c='blue',s=15.,
                alpha=0.9,edgecolor='slateblue', label=r'$\mathrm{STARBIRDS}$')
    plt.scatter(ms_mg17, 10**lsfr_mg17, marker='p',c='slateblue',s=15.,
                alpha=0.9,edgecolor='darkslateblue', 
                label=r'$\mathrm{McGaugh+\ 17}$')

    plt.scatter(ms_j15, 10**lsfr_j15, marker='s',c='slateblue',s=15.,
                alpha=0.9,edgecolor='slateblue', label=r'$\mathrm{Jimmy+\ 15}$')
    plt.scatter(ms_sh, sfr_sh, marker='o',c='m',s=15.,
                alpha=0.9,edgecolor='slateblue', 
                label=r'$\mathrm{SHIELD I\&II, McQuinn+\ 20}$')

    plt.scatter(Msf[MHIf<1], sfrf[MHIf<1], marker='.', s=20, c='gray', 
                alpha=0.07, label=r'${\rm model\ SFR=}M_{\rm H_2}/2\,\rm Gyr$')
    plt.scatter(Msf[MHIf>1], sfrf[MHIf>1], marker='.', s=20, c='orangered', 
                alpha=0.75, label=r'${\rm model\ SFR=}M_{\rm H_2}/2\,\rm Gyr$')


    # plt.legend(frameon=False,loc='upper left', bbox_to_anchor=(0., 1.25), ncol=2, fontsize=5)
    plt.legend(frameon=True,loc='lower right', ncol=1, fontsize=3.5)
    plt.grid(ls=':', c='lightgray', which='major',alpha = 0.5)
    plt.tight_layout()
    fig_pdf.savefig(fig)


    #this is plot for SFR vs. Mg 
    fig = plt.figure(figsize=(3.,3.))
    plt.xscale('log'); plt.yscale('log')
    plt.xlim([1.e5,1.e11]); plt.ylim([1.e-5,1.])
    plt.ylabel(r'$\rm SFR\ (M_\odot\,yr^{-1})$'); plt.xlabel(r'$M_{\rm g}\ {(M_\odot)}$')

    plt.scatter(mhaa, sfraa, marker='.',c='m',s=5.,
                alpha=0.5, edgecolor='none', label=r'$\mathrm{ALFALFA}$')

    plt.scatter(mg_mg17, 
    10**lsfr_mg17, marker='p',c='slateblue',s=15.,
                alpha=0.9,edgecolor='darkslateblue', label=r'$\mathrm{McGaugh+\ 17}$')


    plt.scatter(mg_sh, sfr_sh, marker='o',c='m',s=15.,
                alpha=0.9,edgecolor='slateblue', label=r'$\mathrm{SHIELD I\&II, McQuinn+\ 20}$')

    plt.scatter(Mgf[MHIf<1], sfrf[MHIf<1], marker='.', s=20, c='gray', 
                alpha=0.075, label=r'${\rm model\ SFR=}M_{\rm H_2}/2\,\rm Gyr$')

    plt.scatter(Mgf[MHIf>1], sfrf[MHIf>1], marker='.', s=20, c='orangered', 
                alpha=0.75, label=r'${\rm model\ SFR=}M_{\rm H_2}/2\,\rm Gyr$')


    # plt.legend(frameon=False,loc='upper left', bbox_to_anchor=(0., 1.25), ncol=2, fontsize=5)
    plt.legend(frameon=True,loc='lower right', ncol=1, fontsize=3.5)
    plt.grid(ls=':', c='lightgray', which='major',alpha = 0.5)
    plt.tight_layout()
    fig_pdf.savefig(fig)


    #this is plot for sSFR vs. Mgas
    fig = plt.figure(figsize=(3.,3.))
    plt.xscale('log'); plt.yscale('log')
    plt.xlim([1.e5,1.e11]); plt.ylim([1.e-14,1.e-7])
    plt.ylabel(r'$\rm sSFR\ (yr^{-1})$'); plt.xlabel(r'$M_{\rm g}\ {(M_\odot)}$')

    plt.scatter(mhaa, sfraa/msaa, marker='.',c='m',s=5.,
                alpha=0.9, edgecolor='none', label=r'$\mathrm{ALFALFA}$')


    plt.scatter(mg_mg17, 10**lsfr_mg17/ms_mg17, marker='p',c='slateblue',s=15.,
                alpha=0.9,edgecolor='darkslateblue', 
                label=r'$\mathrm{McGaugh+\ 17}$')


    plt.scatter(mg_sh, sfr_sh/ms_sh, marker='o',c='m',s=15.,
                alpha=0.9,edgecolor='slateblue', label=r'$\mathrm{SHIELD I\&II, McQuinn+\ 20}$')

    plt.scatter(Mgf[MHIf>1], sfrf[MHIf>1]/Msf[MHIf>1], marker='.', s=40, c='orangered', 
                alpha=0.75, label=r'${\rm model\ SFR=}M_{\rm H_2}/2\,\rm Gyr$')
    plt.scatter(Mgf[MHIf<1], sfrf[MHIf<1]/Msf[MHIf<1], marker='.', s=40, c='gray', 
                alpha=0.075, label=r'${\rm model,\ no\ HI}$')


    # plt.legend(frameon=False,loc='upper left', bbox_to_anchor=(0., 1.25), ncol=2, fontsize=5)
    plt.legend(frameon=True,loc='lower right', ncol=1, fontsize=3.5)
    plt.grid(ls=':', c='lightgray', which='major',alpha = 0.5)
    plt.tight_layout()
    fig_pdf.savefig(fig)

    return 

#############################
#PLOT 6: gas fraction (Mg/Mstar) vs. Mstar plot 
#############################

def plot_gas_fraction(Msf=None,Mgf=None,MHIf=None,obs_data_path=None,fig_pdf=None):

    url = obs_data_path + '/durbala2020-table2.21-Sep-2020.fits'
    aa = fits.getdata(url)

    #msaa = 10**aa['logMstarMcGaugh']#aa['logMstarGSWLC']#aa['logMstarTaylor']
    msaa = 10**aa['logMstarTaylor']
    mhaa = 1.33*10.**aa['logMH']


    
    hdulist = fits.open(obs_data_path+'/figure_1_bradford_2016.fits')
    bf1 = np.asarray(hdulist[1].data)
    # if you want to see what's in the fits table, uncomment
    #bf1h = pyfits.open('data/figure_1_bradford_2016.fits')[1].header
    #print(bf1h)

    # data table in the FITS is a dictionary, which we will convert to the numpy dictionary (record)
    data = np.asarray(hdulist[1].data)
    mbbar = bf1['MBARYON']; embar = bf1['MBARYON_ERR']
    w20b = bf1['VW20I']; ew20b = bf1['VW20I_ERR']
    #mbHI = bf1['MHI'];
    lmbs = bf1['MSTAR']
    mbs = 10.**lmbs

    # note that Mgas in mbbar is already corrected for He
    # see eqs 2 and 3 in Bradford et al. 2016
    mbg = (10.**mbbar-mbs)
    mbgms = mbg/mbs

    lmsp, fgmedp, fg16p, fg84p = np.loadtxt(obs_data_path+'/peeples14_tab1.dat', 
                                            usecols=(0,1,2,3), unpack=True)

    lmhid, msd = np.loadtxt(obs_data_path+'/mgas_ms_low_mass_dwarfs.dat', 
                                            usecols=(1,2), unpack=True)

    mhid = 10**(lmhid+0.13); msd *= 1e6

    mhik, msk = np.loadtxt(obs_data_path+'/glow_sample.dat', 
                                            usecols=(1,2), unpack=True)

    mhik = 10**(mhik+0.13); msk = 10**msk 

    #mhih, msh = np.loadtxt('data/huang12_tab1.dat', 
    #                                         usecols=(1,3), unpack=True)
    #mhih = 10**mhih; msh = 10**msh

    mgms_sh, fg_sh, mb_sh, ms_sh, sfr_sh = np.loadtxt(obs_data_path+'/shield_I_II.dat', 
                                                    usecols=(1, 4, 7, 16, 19), 
                                                    unpack=True)

    sfr_sh *= 1.e-3 # convert to Msun/yr
    mg_sh = 10.**mb_sh[ms_sh>0] - 10**ms_sh[ms_sh>0]
    sfr_sh = sfr_sh[ms_sh>0]
    ms_sh = 10.**ms_sh[ms_sh>0]
    mb_sh = 10**mb_sh

    mhiad, msad = np.loadtxt(obs_data_path  + '/angst_mhi_ms.dat', 
                             usecols=(1,2), unpack=True)
    mhia = mhiad[mhiad>0]
    msa = msad[mhiad>0]
    mhia *= 1.e6
    mhia *=1.35
    msa *= 1.e7 
    mhiaul = 1e6 * np.abs(mhiad[mhiad<0])
    msaul = 1e7 * msad[mhiad<0]

    lms_mg17, lmg_mg17, lsfr_mg17 = np.loadtxt(obs_data_path+'/mcgaugh_etal17_tab1.dat', 
                                               usecols=(2,3,5), unpack=True)
    ms_mg17 = 10.**lms_mg17
    mg_mg17 = 10.**lmg_mg17


    fig  = plt.figure(figsize=(4.,4.))
    plt.xscale('log'); plt.yscale('log')
    plt.xlim([1.e2,5.e11]); plt.ylim([3.e-3,1000.])
    plt.ylabel(r'$M_{\rm g}/M_\star$'); plt.xlabel(r'$M_{\star}\ {(M_\odot)}$')

    plt.fill_between(10.**lmsp, fg16p, fg84p, color='gray', alpha=0.2)


    plt.scatter(msaa, mhaa/msaa, marker='.',c='m',s=5.,
                alpha=0.9, edgecolor='none', label=r'$\mathrm{ALFALFA}$')

    plt.scatter(mbs, mbgms, marker='.',c='darkslateblue',s=10.,
                alpha=0.9,edgecolor='none', label=r'$\mathrm{Bradford+\ 16}$')

    plt.scatter(msd, mhid/msd, marker='.',c='slateblue',s=40.,
                alpha=0.9, edgecolor='darkslateblue', 
                label=r'$\mathrm{compilation}$')
    plt.scatter(msk, mhik/msk, marker='p',c='slateblue',s=40.,
                alpha=0.9, edgecolor='darkslateblue', 
                label=r'$\mathrm{GLOW\ sample}$')
    plt.scatter(ms_sh, mgms_sh, marker='.',c='m',s=70.,
                alpha=0.9,edgecolor='darkslateblue', 
                label=r'$\mathrm{SHIELD\ I\ +\ II}$')
    plt.scatter(msa, mhia/msa, marker='s',c='skyblue',s=15.,
                alpha=0.9,edgecolor='slateblue', 
                label=r'$\mathrm{ANGST}$')
    plt.scatter(msaul, mhiaul/msaul, marker='v',c='skyblue',s=15.,
                alpha=0.9,edgecolor='slateblue', 
                label=r'$\mathrm{ANGST\ upp.\ limits}$')

 
    plt.scatter(ms_mg17, mg_mg17/ms_mg17, marker='p',c='slateblue',s=15.,
                alpha=0.9,edgecolor='darkslateblue', 
                label=r'$\mathrm{McGaugh+\ 17}$')

    plt.scatter(Msf, MHIf/Msf, marker='.', s=30, c='orangered', 
                alpha=0.75, label=r'${\rm model}$')

    plt.scatter(Msf[MHIf<1], Mgf[MHIf<1]/Msf[MHIf<1], marker='o', s=20, c='gray', 
                alpha=0.05, label=r'${\rm model}$')

    plt.plot(10**lmsp, fgmedp, c='slateblue', label='Peeples et al. 2014')
    # plt.legend(frameon=False,loc='upper left', bbox_to_anchor=(0., 1.25), ncol=2, fontsize=5)
    plt.legend(frameon=False,loc='lower left', ncol=1, fontsize=5)
    plt.grid(ls=':', c='lightgray', which='major',alpha = 0.5)
    plt.tight_layout()
    fig_pdf.savefig( fig )


    #this is plot for Mstar vs. Mg
    fig  = plt.figure(figsize=(4.,4.))
    plt.xscale('log'); plt.yscale('log')
    plt.xlim([1.e4,5.e11]); plt.ylim([1e5,1e9])
    plt.ylabel(r'$M_{\rm g}\ {(M_\odot)}$'); plt.xlabel(r'$M_{\star}\ {(M_\odot)}$')

    plt.fill_between(10.**lmsp, fg16p, fg84p, color='gray', alpha=0.2)


    plt.scatter(msaa, mhaa, marker='.',c='m',s=5.,
                alpha=0.9, edgecolor='none', label=r'$\mathrm{ALFALFA}$')

    plt.scatter(mbs, mbgms, marker='.',c='darkslateblue',s=10.,
                alpha=0.9,edgecolor='none', label=r'$\mathrm{Bradford+\ 16}$')

    plt.scatter(msd, mhid, marker='.',c='slateblue',s=40.,
                alpha=0.9, edgecolor='darkslateblue', 
                label=r'$\mathrm{compilation}$')

    plt.scatter(msk, mhik, marker='p',c='slateblue',s=40.,
                alpha=0.9, edgecolor='darkslateblue', 
                label=r'$\mathrm{GLOW\ sample}$')
    plt.scatter(ms_sh, mg_sh, marker='.',c='m',s=70.,
                alpha=0.9,edgecolor='darkslateblue', 
                label=r'$\mathrm{SHIELD\ I\ +\ II}$')
    plt.scatter(msa, mhia, marker='s',c='skyblue',s=15.,
                alpha=0.9,edgecolor='slateblue', 
                label=r'$\mathrm{ANGST}$')
    plt.scatter(msaul, mhiaul, marker='v',c='skyblue',s=15.,
                alpha=0.9,edgecolor='slateblue', 
                label=r'$\mathrm{ANGST\ upp.\ limits}$')

    plt.scatter(ms_mg17, mg_mg17, marker='p',c='slateblue',s=15.,
                alpha=0.9,edgecolor='darkslateblue', 
                label=r'$\mathrm{McGaugh+\ 17}$')

    plt.scatter(Msf, MHIf, marker='.', s=30, c='orangered', 
                alpha=0.75, label=r'${\rm model}$')

    plt.scatter(Msf[MHIf<1], Mgf[MHIf<1], marker='.', s=20, c='gray', 
                alpha=0.1, label=r'${\rm model}$')

    plt.plot(10**lmsp, fgmedp, c='slateblue', label='Peeples et al. 2014')
    plt.legend(frameon=False,loc='lower right', ncol=1, fontsize=5)
    plt.grid(ls=':', c='lightgray', which='major',alpha = 0.5)
    plt.tight_layout()
    fig_pdf.savefig( fig )

    return 


def plot_hof(Mpeak=None,mVs=None,fig_pdf=None):

    lMhg = np.linspace(6, 10, 25)
    Mhg = 10.**lMhg

    flh = []
    for i, md in enumerate(Mhg[1:]): 
        isel1 = ((Mpeak < md) & (Mpeak >Mhg[i]))
        isel2 = ((Mpeak < md) & (Mpeak >Mhg[i]) & (mVs < 0))
        fl = np.size(Mpeak[isel2]) / np.maximum(1, np.size(Mpeak[isel1]))
        flh.append(fl)



    fig, ax = plt.subplots(1,1, figsize=(4, 4))


    m200n = 8.5e7 # M200c for Nadler et al.
    fn = 0.5
    ax.errorbar(m200n, fn, yerr=0.025, lolims=True, ls=None, 
                marker='_', markersize=40, capsize=6, ecolor='steelblue', 
                mew=1., lw=2, c='white', zorder=0)

    ax.scatter(m200n, fn, marker='_', s=130, lw=3, c='steelblue')
    ax.scatter(0, 1, marker='^', s=110, lw=1, c='steelblue', label=r'\rm Nadler et al. 2020')
    ax.plot( Mhg[1:],flh,lw = 2,color = "k",ls = "dotted",label = r"\rm Model")
    ax.tick_params(axis = "x",labelsize = 12)
    ax.tick_params(axis = "y",labelsize = 12)
    ax.legend(frameon=False, loc='lower right', fontsize=10.9)
    ax.grid(linestyle=':', c='lightgray',alpha =0.7)
    ax.legend(frameon=False, loc='lower right', fontsize=10.75)
    ax.set_ylabel(r'\rm $f(M_V<0)$',fontsize = 14)
    ax.set_xlabel(r'\rm $M_{\rm h,peak}\ (M_\odot)$',fontsize = 14)
    ax.set_xscale('log')
    ax.set_xlim(1.5e6, 1e10); 
    ax.set_ylim(0., 1)
    ax.xaxis.set_ticks([1e7, 1e8, 1e9, 1e10])
    ax.yaxis.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.minorticks_on()

    plt.tight_layout()
    fig_pdf.savefig( fig )
    return 

def muV_lines(ax=None,muV=None,ls = ":",lw = 1,color = "indianred"):
    mvtr = np.linspace(5,-20,100)    
    term1 = (muV -  mvtr - 21.5721)/2.5
    
    rps = np.sqrt(10**term1/(2*np.pi))
    
    ax.plot(np.log10(rps),mvtr,ls = ls, lw = lw,color = color)

    return 


def plot_size_lumi(rsizes=None,mVs = None,obs_data_path=None,fig_pdf=None):

    mw_sat_data = np.loadtxt(obs_data_path + "/drlica_wagner20_tab2_read.txt",dtype = str)
    mw_class = mw_sat_data[:,2].astype(float)
    mw_mv = mw_sat_data[:,-2].astype(float)
    mw_r12 = mw_sat_data[:,-3].astype(float)

    mw_sat_mv = mw_mv[mw_class == 4]
    mw_sat_r12 = mw_r12[mw_class == 4]

    mw_cd_mv = mw_mv[mw_class == 3]
    mw_cd_r12 = mw_r12[mw_class == 3]

    mw_unc_mv = np.array(list([mw_mv[mw_class == 1][0]]) +  [-2.1])
    mw_unc_r12 = np.array(list([mw_r12[mw_class == 1][0]]) + [21])

    #taken from table in Grumpy paper 2
    mw_sc_mv = np.array([0,0.3,-1.6,-2.4,-1.42,0.7,0.3,-4.4,-1.21,-0.4,-0.04,-2.07,-4.3,-0.2,-4.80,-1.5,-5.0,-2,-1.,-1.6,-3.3,-1.9,-1.1,-0.3])
    mw_sc_r12 = np.array([4.1,3.9,6.65,11.24,5.5,2.29,6.9,7,7.24,7.1,2.1,5.0,22,5.4,20.7,12.8,9,3,5.6,3.45,7.45,4.69,7.58,1.31])

    fig,ax = plt.subplots(1,1,figsize = (4,4))

    ax.hist2d(np.log10(1e3*rsizes/1.34),mVs,bins=30,norm = LogNorm(),density = True,cmap = "inferno")


    ax.scatter(np.log10(mw_sc_r12),mw_sc_mv,facecolor = "cyan",edgecolor = "k",marker = "o",s=30,lw = 0.5,
              label = r"\rm Probable Star Clusters",alpha = 0.9)

    ax.scatter(np.log10(mw_unc_r12),mw_unc_mv,facecolor = "red",edgecolor = "k",marker = "*",s=120,lw = 0.5,
              label = r"\rm Unconfirmed Systems",alpha = 0.9)

    ##this is the new dwarf from Eri IV
    ax.scatter([np.log10(75)],[-4.7],facecolor = "orange",edgecolor = "k",marker = "*",s=120,lw = 0.5,alpha = 0.9   )


    ax.scatter(np.log10(mw_cd_r12),mw_cd_mv,facecolor = "orange",edgecolor = "k",marker = "*",s=120,lw = 0.5,
              label = r"\rm Candidate Dwarfs",alpha = 0.9)

    ax.scatter(np.log10(mw_sat_r12),mw_sat_mv,facecolor = "limegreen",edgecolor = "k",marker = "*",s=120,lw = 0.5,
              label = r"\rm Confirmed Dwarfs",alpha = 0.9)

    muV_lines(ax=ax,muV = 36,ls = "dashed",lw = 1,color = "grey")
    muV_lines(ax=ax,muV = 32,ls = "--",lw = 1,color = "grey")
    muV_lines(ax=ax,muV = 24,ls = "--",lw = 1,color = "grey")
    muV_lines(ax=ax,muV = 28,ls = "--",lw = 1,color = "grey")

    ax.text(3.36,-0.74-2.65,r"\rm $\mu_{\rm V} = 36$",fontsize = 11,rotation=46,color = "grey")
    ax.text(3.36,-4.74-2.65,r"\rm $\mu_{\rm V} = 32$",fontsize = 11,rotation=46,color = "grey")
    ax.text(3.36,-8.74-2.65,r"\rm $\mu_{\rm V}= 28$",fontsize = 11,rotation=46,color = "grey")
    ax.text(0.8,-5.5,r"\rm $\mu_{\rm V} = 24$",fontsize = 11,rotation=46,color = "grey")


    ax.set_xlim([0,4])
    ax.set_ylim([2.5,-18])
    ax.grid(ls = ":",color = "k",alpha = 0.2)
    ax.set_ylabel(r'$M_{\rm V}$',fontsize = 14) 
    ax.set_xlabel(r'\rm $\log_{10}(r_{\rm 1/2})$ (pc)',fontsize = 14)
    ax.legend(frameon=False,fontsize = 9.7,loc ="upper left")

    ax.tick_params(axis = "x",labelsize = 11)
    ax.tick_params(axis = "y",labelsize = 11)
    plt.tight_layout()
    fig_pdf.savefig( fig )

    return 


def run_plotting(iniconf=None):


    obs_data_path = iniconf['code paths']['code_base_dir'] + "/data"
    final_csv_path = iniconf['output paths']['final_output_dir'] + "/" + iniconf['output paths']['run_name'] +"_run.csv"

    Zsun = float(iniconf['model params']['Zsun'])

    #read the final output csv file
    df_data = pd.read_csv(final_csv_path)

    Msf = np.array(df_data["Ms"])
    Mgf = np.array(df_data["Mg"])
    Mpeak = np.array(df_data["Mpeak"])
    rsizes = np.array(df_data["rsize"])
    sfrf = np.array(df_data["sfrf"])
    MZsf = np.array(df_data["MZs"]) 
    MZgf = np.array(df_data["MZg"])
    MHIf = np.array(df_data["MHI"])

    #do all the plotting using the plotting functions and save the figures into a single pdf
    images_filename = iniconf['output paths']['final_output_dir'] + "/" + iniconf['output paths']['run_name'] +"_summary_plots.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(images_filename)

    plot_SHMR(Mpeak=Mpeak, Msf=Msf, obs_data_path=obs_data_path, fig_pdf=pdf)

    plot_massZ(Msf=Msf, MZsf=MZsf, Mgf=Mgf, MZgf=MZgf, MHIf=MHIf,
               obs_data_path=obs_data_path,Zsun = Zsun, fig_pdf=pdf)
    plot_sizes(Msf=Msf, rsizes=rsizes, obs_data_path=obs_data_path, fig_pdf=pdf)
  
    plot_sfr(Msf=Msf, Mgf=Mgf, sfrf=sfrf, MHIf=MHIf, 
             obs_data_path=obs_data_path, fig_pdf=pdf)
    plot_gas_fraction(Msf=Msf, Mgf=Mgf, MHIf=MHIf, 
                      obs_data_path=obs_data_path, fig_pdf=pdf)

    #only if FSPS was run
    if iniconf['run params']['run_fsps'] == "True":
        mVs = np.array(df_data["mV"])
        plot_hof(Mpeak=Mpeak,mVs=mVs,fig_pdf=pdf)
        plot_size_lumi(rsizes=rsizes,mVs = mVs,obs_data_path=obs_data_path,fig_pdf=pdf)



    pdf.close()

    return 
