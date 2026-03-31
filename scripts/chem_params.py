from collections import namedtuple
import numpy as np


def mass_fractions_from_bracket_XH(elements_to_track, Z_IGM, bracket_XH_list, solar_abundance_name):
    """
    Convert [X/H] values (dex) to mass fractions for initial IGM composition.

    Uses [X/H] = log10((f_X/f_H)/(f_X_sun/f_H_sun)) => f_X = f_H * (f_X_sun/f_H_sun) * 10^[X/H].
    H and He use fixed fractions 0.75 and 0.25 of (1 - Z_IGM).

    Parameters
    ----------
    elements_to_track : array-like of str
        Element symbols in order (e.g. ['H','He','Fe','C']).
    Z_IGM : float
        Total metal mass fraction (e.g. Z_IGM, Z_initial_gas, or Z_initial_star from config).
    bracket_XH_list : array-like of float
        [X/H] in dex, one per metal in elements_to_track (same order, no H/He).
    solar_abundance_name : str
        Name of solar set (e.g. 'Asplund09').

    Returns
    -------
    f_array : np.ndarray
        Mass fraction for each element in elements_to_track. Caller sets mg_X = mg_start * f_array.
    """
    from Chempy.solar_abundance import solar_abundances

    elements_to_track = np.asarray(elements_to_track)
    bracket_XH_list = np.asarray(bracket_XH_list, dtype=float)

    basic_solar = solar_abundances()
    getattr(basic_solar, solar_abundance_name)()
    solar_elements = np.hstack(basic_solar.all_elements)
    solar_fractions = np.array([
        float(basic_solar.fractions[np.where(solar_elements == el)][0])
        for el in elements_to_track
    ])

    f_H = 0.75 * (1.0 - Z_IGM)
    f_He = 0.25 * (1.0 - Z_IGM)
    f_H_sun = solar_fractions[elements_to_track == "H"][0]

    print(f"H fraction in Sun = {f_H_sun}")
    print(f"Z_IGM = {Z_IGM}")

    print("Fraction in metals = ",1 - f_H - f_He)

    f_array = np.zeros_like(elements_to_track, dtype=float)

    metal_idx = 0
    for j, ei in enumerate(elements_to_track):
        if ei == "H":
            f_array[j] = f_H
        elif ei == "He":
            f_array[j] = f_He
        else:
            f_X_sun = solar_fractions[j]
            bracket_xh = bracket_XH_list[metal_idx]
            f_array[j] = f_H * (f_X_sun / f_H_sun) * (10.0 ** bracket_xh)
            metal_idx += 1

            print("ELEMENT = ", ei)
            print("Fraction = ", f_array[j])
            print("Bracket XH = ", bracket_xh)
            print("Solar fraction = ", f_X_sun/f_H_sun)
            print("---")

    f_metals_sum = np.sum(f_array[(elements_to_track != "H") & (elements_to_track != "He")])
    if f_metals_sum > Z_IGM:
        raise ValueError(
            "Initial [X/H] values imply tracked metal mass fraction (%.6e) > total Z (%.6e). "
            "Reduce [X/H] values or increase total metallicity (Z_IGM, log_Z_initial_gas, or log_Z_initial_star). "
            "Minimum total Z required for these [X/H] values: %.6e "
            "(increase the relevant Z to avoid this error)."
            % (f_metals_sum, Z_IGM, f_metals_sum)
        )

    return f_array


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

    ccsne_vals = ["Nomoto2013", "chieffi04", "Nomoto2013_net", "chieffi04_net", "CL18_net", "West17_net", "Frischknecht16_net"]
    agb_vals = ["Karakas", "Karakas_net", "Karakas_net_yield"]
    sn1a_vals = ["Iwamoto", "Thielmann", "Seitenzahl"]

    solar_abundances_vals = ["Asplund09","Lodders09"]

    all_element_list = iniconf['chem model']['elements_list'].split(",")
    n_metals = len(all_element_list) - 2

    # X_H_IGM, X_H_initial_gas, and X_H_initial_star are required
    chem_section = iniconf['chem model']
    if chem_section.get('X_H_IGM') is None:
        raise ValueError("'X_H_IGM' is required in [chem model]. Set [X/H] in dex for each tracked metal (same order as elements_list excluding H,He).")
    xh_igm_vals = chem_section.get('X_H_IGM').split(",")
    if len(xh_igm_vals) != n_metals:
        raise ValueError("The number of values in 'X_H_IGM' (%d) does not match the number of metals (%d)." % (len(xh_igm_vals), n_metals))

    if chem_section.get('X_H_initial_gas') is None:
        raise ValueError("'X_H_initial_gas' is required in [chem model]. Set [X/H] in dex for each metal, or 'same_as_IGM' to use IGM pattern.")
    xh_initial_gas = chem_section.get('X_H_initial_gas')
    if str(xh_initial_gas).strip().lower() != "same_as_igm":
        xh_gas_vals = xh_initial_gas.split(",")
        if len(xh_gas_vals) != n_metals:
            raise ValueError("The number of values in 'X_H_initial_gas' (%d) does not match the number of metals (%d)." % (len(xh_gas_vals), n_metals))

    if chem_section.get('X_H_initial_star') is None:
        raise ValueError("'X_H_initial_star' is required in [chem model]. Set [X/H] in dex for each metal, or 'same_as_IGM' to use IGM pattern.")
    xh_initial_star = chem_section.get('X_H_initial_star')
    if str(xh_initial_star).strip().lower() != "same_as_igm":
        xh_star_vals = xh_initial_star.split(",")
        if len(xh_star_vals) != n_metals:
            raise ValueError("The number of values in 'X_H_initial_star' (%d) does not match the number of metals (%d)." % (len(xh_star_vals), n_metals))

    if 'log_Z_IGM' not in chem_section:
        raise ValueError("'log_Z_IGM' is required in [chem model]. Set log10(Z_IGM/Zsun), e.g. -3.0 for 10^-3 Zsun.")
    try:
        _ = float(chem_section['log_Z_IGM'])
    except (TypeError, ValueError):
        raise ValueError("'log_Z_IGM' in [chem model] must be a number (log10(Z_IGM/Zsun)).")

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

    if iniconf['chem model']['solar_abundance_name'] not in solar_abundances_vals:
        raise ValueError("The solar abundance set %s is not a valid set. Allowed names are : [ Asplund09 , Lodders09 ]"%(iniconf['chem model']['solar_abundance_name']))


    if iniconf['chem model']['ccsne_yields'] not in ccsne_vals:
        raise ValueError('The CCSNE yield set "{:s}" is not a valid yield set. Allowed names : {!s} !'.format(iniconf['chem model']['ccsne_yields'], ccsne_vals))
    
    if iniconf['chem model']['agb_yields'] not in agb_vals:
        raise ValueError('The AGB yield set "{:s}" is not a valid yield set. Allowed names : {!s} !'.format(iniconf['chem model']['agb_yields'], agb_vals))

    if iniconf['chem model']['sn1a_yields'] not in sn1a_vals:
        raise ValueError('The SN1A yield set "{:s}" is not a valid yield set. Allowed names : {!s} !'.format(iniconf['chem model']['sn1a_yields'], sn1a_vals))

    imf_types = ['salpeter', 'Chabrier_1', 'Chabrier_2', 'normed_3slope', 'BrokenPowerLaw']

    if iniconf['chem model']['imf_type_name'] not in imf_types:
        raise ValueError("%s is an incorrect IMF type. Allowed options are : %s"%(iniconf['chem model']['imf_type_name'], imf_types))


    if float(iniconf['chem model']['sn2_mmax']) < float(iniconf['chem model']['sn2_mmin']):
        raise ValueError("The maximum SN2 mass cannot be smaller than the minimum SN2 mass.")


    if float(iniconf['chem model']['agb_mmax']) < float(iniconf['chem model']['agb_mmin']):
        raise ValueError("The maximum AGB mass cannot be smaller than the minimum ABG mass.")

    if float(iniconf['chem model']['imf_mmax']) < float(iniconf['chem model']['imf_mmin']):
        raise ValueError("The maximum IMF mass cannot be smaller than the minimum IMF mass.")

    param_string = "element_list ccsne_yields agb_yields sn1a_yields stochastic stochastic_resampling imf_mmax imf_mmin sn2_mmax sn2_mmin agb_mmax agb_mmin sn1a_time_delay sn1a_norm sn1a_exponent model_nsm nsm_time_delay nsm_norm nsm_exponent imf_type_name solar_abundance_name"

    chem_params = namedtuple("chem_params",param_string)

    #assign parameter values to this named tuple
    chem_params = chem_params(element_list = iniconf['chem model']['elements_list'],
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
                                model_nsm = iniconf['chem model']['model_nsm'],
                                nsm_time_delay = float(iniconf['chem model']['nsm_time_delay']),
                                nsm_norm = float(iniconf['chem model']['nsm_norm']),
                                nsm_exponent = float(iniconf['chem model']['nsm_exponent']),
                                imf_type_name = iniconf['chem model']['imf_type_name'],
                                solar_abundance_name = iniconf['chem model']['solar_abundance_name'])

    ###print the parameter value summary!
    if print_params == True:
        print(' ')
        print_stage("chemical model parameter summary",ch = "-",end_space=False)
        for i,mpi in enumerate(chem_params._fields):
            print("%s = %s"%(mpi,chem_params[i]))
        print(' ')
    else:
        pass

    return chem_params