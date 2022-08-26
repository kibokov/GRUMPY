####### reionization modelling ################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

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


def get_beta(zrei, gamma):
    """
    compute the beta parameter for the gas accretion suppression equation. 
    The beta parameter controls at what redshift the sharp increase in Mc occurs.
    It is calculated by constraining the Mc at reionization to Okamoto simulations
    given a certain z_re and gamma


    Parameters:
    -----------
        z_re: float, model redshift of reionization
        gamma: float, parameter for sharp increase in Mc
    Returns:
       beta: float
    """
    # mc_oka = 2*8.45e9 * np.exp(-0.63*9) / (1 + np.exp((9/8.7)**15)) 
    #above is the Mc at time of reionization in Okamoto+2008 simulations. 
    #the above describes the value of Mc at z = 9 in the best fit. 
    # alpha = (2*8.45e9* np.exp(-0.63*z_re) / mc_oka ) - 1
    alpha = 1820*np.exp(-0.63*zrei) - 1  #this expression is just above calc simplified
    beta = zrei * (np.log(alpha))**(-1./gamma)
    return beta

def eps(z, beta, gamma):
    '''
    
    '''
    t1 = (gamma * z**(gamma-1) / beta**gamma) * np.exp((z/beta)**gamma)
    t2 = 0.63*(1 + np.exp((z/beta)**gamma))
    t3 = (1 + np.exp((z/beta)**gamma))**2
    ret = (t1 + t2) / t3
    if math.isnan(ret) or ret < 1e-7:
        ret = 0
    return ret

def m_c_modified(z, beta, gamma):
    return  2*8.45e9 * np.exp(-0.63*z) / (1 + np.exp((z/beta)**gamma) )
    #1.69e10 normalization is 2 times the best fit value from the Okamoto+2008 data at z=0

def m_c_simple(z):
    return  8.45e9 * np.exp(-0.63*z)
    #1.69e10 normalization is 2 times the best fit value from the Okamoto+2008 data at z=0



def plot_Mc_reion():

    ztr = np.linspace(12,0,1000)
    fig, ax = plt.subplots(1,1,figsize = (4,3))
    ax.set_title(r"\rm $\texttt{okamoto}$ reionization model")
    ax.plot(ztr,m_c_modified(z=ztr,beta=get_beta(zrei = 10,gamma = 15),gamma=15),color = "mediumblue",lw = 2)
    ax.plot(ztr,m_c_modified(z=ztr,beta=get_beta(zrei = 10,gamma = 45),gamma=45),color = "mediumblue",lw = 2,ls = "--")
    ax.plot(ztr,m_c_modified(z=ztr,beta=get_beta(zrei = 10,gamma = 7),gamma=7),color = "mediumblue",lw = 2.5,ls = "dotted")


    ax.plot(ztr,m_c_modified(z=ztr,beta=get_beta(zrei = 8,gamma = 45),gamma=45),color = "darkorange",lw = 2,ls = "--")
    ax.plot(ztr,m_c_modified(z=ztr,beta=get_beta(zrei = 8,gamma = 15),gamma=15),color = "darkorange",lw = 2)
    ax.plot(ztr,m_c_modified(z=ztr,beta=get_beta(zrei = 8,gamma = 7),gamma=7),color = "darkorange",lw = 2.5,ls = "dotted")

    ax.plot(ztr,m_c_modified(z=ztr,beta=get_beta(zrei = 6,gamma = 15),gamma=15),color = "firebrick",lw = 2)
    ax.plot(ztr,m_c_modified(z=ztr,beta=get_beta(zrei = 6,gamma = 45),gamma=45),color = "firebrick",lw = 2,ls = "--")
    ax.plot(ztr,m_c_modified(z=ztr,beta=get_beta(zrei = 6,gamma = 7),gamma=7),color = "firebrick",lw = 2.5,ls = "dotted")

    ax.set_ylim([1e5,3e10])
    ax.set_xlim([0,12])
    ax.set_yscale("log")
    ax.grid(ls = ":",color = "lightgrey",alpha = 0.3)

    ax.tick_params(which='major',axis='both',direction='in',length=4,top=False,bottom=True,right=False,width = 0.6,labelsize = 10.5)


    ax.text(1,4e6*5,r"\rm $z_{\rm rei} = 6$",color = "firebrick",fontsize = 12)
    ax.text(1,4e6,r"\rm $z_{\rm rei} = 8$",color = "darkorange",fontsize = 12)
    ax.text(1,4e6/5,r"\rm $z_{\rm rei} = 10$",color = "mediumblue",fontsize = 12)


    legend_elements = [Line2D([0], [0], color='k', label=r'\rm $\gamma = 7$',lw = 2.5,ls = "dotted"),
                    Line2D([0], [0], color='k', label=r'\rm $\gamma = 15$', lw = 2), 
                    Line2D([0], [0], color='k', label=r'\rm $\gamma = 45$', lw = 2,ls="--")]

    ax.legend(frameon=False, fontsize = 11,loc = "upper right",handles = legend_elements)

    ax.set_xlabel(r"\rm $z$",fontsize = 14)
    ax.set_ylabel(r"\rm $M_c(z)$",fontsize = 12)

    plt.tight_layout()
    plt.savefig("figs/reion_models")

    plt.show()

    return 