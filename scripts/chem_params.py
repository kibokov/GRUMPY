
from collections import namedtuple


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


def extract_chem_params(iniconf=None,print_params = True):
    '''
    Function that parses the config file dictionary to read the model parameters

    Parameters:
    -------------
    iniconf: dict, dictionary of parameters in the config file

    Returns:
    -------------
    chem_params: dict, dictionary for chemical model parameters

    Notes:
    Need to add check for .pickle or not
    '''

    ccsne_vals = ["Nomoto2013","chieffi04"]
    agb_vals = ["Karakas"]
    sn1a_vals = ["Iwamoto","Thielmann","Seitenzahl"]

    solar_abundances_vals = ["Asplund09","Lodders09"]

    all_element_list = iniconf['chem model']['elements_list'].split(",")
    ini_element_ratios = iniconf['chem model']['ini_element_ratios'] 

    if "solar" not in ini_element_ratios:
        if len(ini_element_ratios.split(",")) != int(len(all_element_list) - 2):
            raise ValueError("The number of ratios in 'ini_element_ratios' does not match the number of metals in elements list (i.e. elements not including H and He)")


    if iniconf['chem model']['stochastic'] not in ["True","False"]:
        raise ValueError("Incorrect entry for stochastic in 'chem model'. Allowed entries are : [ True, False ]")

    if iniconf['chem model']['stochastic'] == "True":
        if iniconf['chem model']['stochastic_resampling'] not in ["none","None"]:
            try:
                _ = int(iniconf['chem model']['stochastic_resampling'])
            except:
                raise ValueError("The input for stochastic_resampling has either be None/none or an integer. %s is not a valid input."%(iniconf['chem model']['stochastic_resampling']))
        else:
            pass

    if iniconf['chem model']['solar_abundances'] not in solar_abundances_vals:
        raise ValueError("The solar abundance set %s is not a valid set. Allowed names are : [ Asplund09 , Lodders09 ]"%(iniconf['chem model']['solar_abundances']))


    if iniconf['chem model']['ccsne_yields'] not in ccsne_vals:
        raise ValueError('The CCSNE yield set "{:s}" is not a valid yield set. Allowed names : [ Nomoto2013 , chieffi04 ] !'.format(iniconf['chem model']['ccsne_yields']))
    
    if iniconf['chem model']['agb_yields'] not in agb_vals:
        raise ValueError('The AGB yield set "{:s}" is not a valid yield set. Allowed names : [ Karakas ] !'.format(iniconf['chem model']['ccsne_yields']))

    if iniconf['chem model']['sn1a_yields'] not in sn1a_vals:
        raise ValueError('The SN1A yield set "{:s}" is not a valid yield set. Allowed names : [ Iwamoto , Thielmann , Seitenzahl ] !'.format(iniconf['chem model']['ccsne_yields']))

    imf_types = ['Salpeter', 'Chabrier03', 'Chabrier01']

    if iniconf['chem model']['imf_name'] not in imf_types:
        raise ValueError("%s is an incorrect IMF type. Allowed options are : [Salpeter, Chabrier03, Chabrier01]"%(iniconf['chem model']['imf_name'] ))    


    if float(iniconf['chem model']['sn2_mmax']) < float(iniconf['chem model']['sn2_mmin']):
        raise ValueError("The maximum SN2 mass cannot be smaller than the minimum SN2 mass.")


    if float(iniconf['chem model']['agb_mmax']) < float(iniconf['chem model']['agb_mmin']):
        raise ValueError("The maximum AGB mass cannot be smaller than the minimum ABG mass.")

    if float(iniconf['chem model']['imf_mmax']) < float(iniconf['chem model']['imf_mmin']):
        raise ValueError("The maximum IMF mass cannot be smaller than the minimum IMF mass.")


    param_string = "element_list log_zigm_mean log_zigm_sig ccsne_yields agb_yields sn1a_yields stochastic stochastic_resampling imf_mmax imf_mmin sn2_mmax sn2_mmin agb_mmax agb_mmin sn1a_time_delay sn1a_norm sn1a_exponent imf_name solar_abundances"

    chem_params = namedtuple("chem_params",param_string)

    #assign parameter values to this named tuple
    chem_params = chem_params(element_list = iniconf['chem model']['elements_list'],
                                log_zigm_mean = float(iniconf['chem model']['log_Z_IGM_mean']),
                                log_zigm_sig = float(iniconf['chem model']['log_Z_IGM_sig']),
                                ccsne_yields = iniconf['chem model']['ccsne_yields'],
                                agb_yields = iniconf['chem model']['agb_yields'],
                                sn1a_yields = iniconf['chem model']['sn1a_yields'],
                                stochastic = iniconf['chem model']['stochastic'],
                                stochastic_resampling = iniconf['chem model']['stochastic_resampling'],
                                imf_mmax = float(iniconf['chem model']['imf_mmax']), 
                                imf_mmin = float(iniconf['chem model']['imf_mmin']), 
                                sn2_mmax = float(iniconf['chem model']['sn2_mmax']), 
                                sn2_mmin = float(iniconf['chem model']['sn2_mmin']),
                                agb_mmax = float(iniconf['chem model']['agb_mmax']),
                                agb_mmin = float(iniconf['chem model']['agb_mmin']), 
                                sn1a_time_delay = float(iniconf['chem model']['sn1a_time_delay']),
                                sn1a_norm = float(iniconf['chem model']['sn1a_norm']),
                                sn1a_exponent = float(iniconf['chem model']['sn1a_exponent']),
                                imf_name = iniconf['chem model']['imf_name'],
                                solar_abundances = iniconf['chem model']['solar_abundances'])

    ###print the parameter value summary!
    if print_params == True:
        print(' ')
        print_stage("chemical model parameter summary",ch = "-",end_space=False)
        for i,mpi in enumerate(chem_params._fields):
            if mpi == "log_zigm_mean":
                print("log_Z_IGM_mean = %.2f Zsun"%(chem_params[i]))
            else:
                print("%s = %s"%(mpi,chem_params[i]))
        print(' ')
    else:
        pass

    return chem_params