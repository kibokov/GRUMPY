"""
build_yield_grids.py

Build SSP yield lookup tables as a function of time and metallicity.
Produces RectBivariateSpline interpolators for each tracked element
(net yields and compositional-correction terms), plus aggregate
quantities (total metal yield, surviving stellar mass, mass feedback).

Usage:
    python chempy_gen_yield_grids.py -ini path/to/config.ini

Or import and call create_yield_grid() directly with an SSPParams object.
"""

import pickle
import numpy as np
import time
from tqdm import tqdm
import os
import argparse
import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from scipy.interpolate import RectBivariateSpline
from configparser import ConfigParser, ExtendedInterpolation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Chempy.imf import IMF
from Chempy.solar_abundance import solar_abundances
from Chempy.parameter import ModelParameters
from Chempy.yields import SN2_feedback, AGB_feedback, SN1a_feedback, Hypernova_feedback
from Chempy.weighted_yield import SSP


# ---------------------------------------------------------------------------
#  Parameter container
# ---------------------------------------------------------------------------

@dataclass
class SSPParams:
    """
    All free parameters controlling SSP yield grid computation.

    Organizes every knob that Chempy exposes into a single flat structure
    with sensible defaults matching Chempy's own ModelParameters.
    """

    # -- Yield table choices --
    ccsne_yields: str = "Nomoto2013"
    agb_yields: str = "Karakas_net_yield"
    sn1a_yields: str = "Seitenzahl"

    # -- Solar abundance set --
    solar_abundance_name: str = "Asplund09"

    # -- IMF --
    imf_type_name: str = "Chabrier_1"
    imf_mmin: float = 0.1
    imf_mmax: float = 100.0
    high_mass_slope: float = -2.29
    chabrier_sigma: float = 0.69
    chabrier_mc: float = 0.079

    # -- Mass ranges --
    sn2_mmin: float = 8.0
    sn2_mmax: float = 100.0
    agb_mmin: float = 0.5
    agb_mmax: float = 8.0
    sn1a_mmin: float = 1.0
    sn1a_mmax: float = 8.0

    # -- BH feedback --
    # BH feedback is pure mass bookkeeping for stars too massive for the CC-SN
    # yield tables (mass between bh_mmin and bh_mmax).  It does NOT produce new
    # elements; it just returns unprocessed gas to the ISM from BH progenitors.
    #
    # By default bh_mmin = sn2_mmax and bh_mmax = imf_mmax.  With the standard
    # setup (sn2_mmax == imf_mmax == 100), the BH mass range is empty and this
    # channel has zero effect.
    #
    # To disable BH feedback entirely, simply keep sn2_mmax == imf_mmax
    # (both at 100), or explicitly set bh_mmin = bh_mmax = any equal value.
    # The code skips the call whenever bh_mmin >= bh_mmax.
    bh_mmin: Optional[float] = None
    bh_mmax: Optional[float] = None
    percentage_of_bh_mass: float = 0.25

    # -- SN Ia time-delay distribution --
    time_delay_functional_form: str = "maoz"
    # Parameters for the 'maoz' DTD (power-law; Maoz+ 2012)
    sn1a_norm: float = 0.001778  # 10^-2.75
    sn1a_time_delay: float = 0.1585  # 10^-0.8  Gyr
    sn1a_exponent: float = 1.12
    # Parameters for the 'normal' DTD (Gaussian)
    sn1a_gauss_norm: float = 0.003
    sn1a_gauss_peak: float = 1.0
    sn1a_gauss_scale: float = 3.2
    sn1a_gauss_beginning: float = 0.25
    # Parameters for the 'gamma_function' DTD
    sn1a_gamma_norm: float = 0.0024
    sn1a_gamma_a: float = 1.3
    sn1a_gamma_loc: float = 0.0
    sn1a_gamma_scale: float = 3.0

    # -- Hypernova mixing (only relevant for Nomoto2013) --
    # 1.0 = pure CC-SN yields, 0.0 = pure hypernova yields
    sn2_to_hn: float = 1.0

    # -- Stellar physics --
    stellar_lifetimes: str = "Argast_2000"
    interpolation_scheme: str = "logarithmic"

    # -- IMF mass resolution --
    mass_steps: int = 200000

    # -- Metallicity & time grids for the lookup table --
    z_grid: np.ndarray = field(
        default_factory=lambda: np.logspace(-6, -1.3, 15)
    )
    time_steps: np.ndarray = field(
        default_factory=lambda: np.logspace(-3, np.log10(15), 50)
    )

    def __post_init__(self):
        if self.bh_mmin is None:
            self.bh_mmin = self.sn2_mmax
        if self.bh_mmax is None:
            self.bh_mmax = self.imf_mmax

    @property
    def sn1a_parameter(self):
        """Build the 4-element SN Ia parameter list expected by Chempy."""
        if self.time_delay_functional_form == "maoz":
            return [self.sn1a_norm, self.sn1a_time_delay,
                    self.sn1a_exponent, 0.0]
        elif self.time_delay_functional_form == "normal":
            return [self.sn1a_gauss_norm, self.sn1a_gauss_peak,
                    self.sn1a_gauss_scale, self.sn1a_gauss_beginning]
        elif self.time_delay_functional_form == "gamma_function":
            return [self.sn1a_gamma_norm, self.sn1a_gamma_a,
                    self.sn1a_gamma_loc, self.sn1a_gamma_scale]
        else:
            raise ValueError(
                f"Unknown time_delay_functional_form: "
                f"{self.time_delay_functional_form}"
            )


# ---------------------------------------------------------------------------
#  Config file I/O
# ---------------------------------------------------------------------------

def read_config(path: str) -> ConfigParser:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(path)
    return cfg


def params_from_config(cfg: ConfigParser) -> SSPParams:
    """
    Build an SSPParams from a ConfigParser.  Any key in [chem model] that
    matches an SSPParams field name will override the default.
    """
    kw = {}
    if cfg.has_section("chem model"):
        sec = cfg["chem model"]
        float_keys = {
            "imf_mmin", "imf_mmax", "high_mass_slope",
            "chabrier_sigma", "chabrier_mc",
            "sn2_mmin", "sn2_mmax", "agb_mmin", "agb_mmax",
            "sn1a_mmin", "sn1a_mmax",
            "bh_mmin", "bh_mmax", "percentage_of_bh_mass",
            "sn1a_norm", "sn1a_time_delay", "sn1a_exponent",
            "sn1a_gauss_norm", "sn1a_gauss_peak",
            "sn1a_gauss_scale", "sn1a_gauss_beginning",
            "sn1a_gamma_norm", "sn1a_gamma_a",
            "sn1a_gamma_loc", "sn1a_gamma_scale",
            "sn2_to_hn",
        }
        int_keys = {"mass_steps"}
        str_keys = {
            "ccsne_yields", "agb_yields", "sn1a_yields",
            "solar_abundance_name", "imf_type_name",
            "time_delay_functional_form",
            "stellar_lifetimes", "interpolation_scheme",
        }
        for k in float_keys:
            if k in sec:
                kw[k] = sec.getfloat(k)
        for k in int_keys:
            if k in sec:
                kw[k] = sec.getint(k)
        for k in str_keys:
            if k in sec:
                kw[k] = sec.get(k)
    return SSPParams(**kw)


# ---------------------------------------------------------------------------
#  Yield-table loading helpers
# ---------------------------------------------------------------------------

def _build_imf(params: SSPParams) -> IMF:
    basic_imf = IMF(params.imf_mmin, params.imf_mmax, params.mass_steps)

    if params.imf_type_name == "Chabrier_1":
        imf_par = (params.chabrier_sigma, params.chabrier_mc,
                   params.high_mass_slope)
        basic_imf.Chabrier_1(imf_par)

    elif params.imf_type_name == "salpeter":
        basic_imf.salpeter(params.high_mass_slope)

    elif params.imf_type_name == "normed_3slope":
        imf_par = (params.high_mass_slope, -2.3, params.high_mass_slope,
                   0.5, 1.0)
        basic_imf.normed_3slope(imf_par)

    elif params.imf_type_name == "BrokenPowerLaw":
        basic_imf.BrokenPowerLaw(
            ((0.5, 1.39, 6), (-1.26, -1.49, -3.02, params.high_mass_slope))
        )

    elif params.imf_type_name == "Chabrier_2":
        imf_par = (22.8978, 716.4, 0.25, params.high_mass_slope)
        basic_imf.Chabrier_2(imf_par)

    else:
        raise ValueError(f"Unknown IMF: {params.imf_type_name}")

    return basic_imf


def _load_yield_sets(params: SSPParams):
    """Load the three yield-table objects and optionally mix hypernovae."""
    basic_sn2 = SN2_feedback()
    getattr(basic_sn2, params.ccsne_yields)()

    # Hypernova mixing for Nomoto2013 (following Chempy's SSP_wrap logic)
    if params.ccsne_yields == "Nomoto2013" and params.sn2_to_hn < 1.0:
        basic_hn = Hypernova_feedback()
        basic_hn.Nomoto2013()
        _mix_hypernova(basic_sn2, basic_hn, params.sn2_to_hn)

    basic_agb = AGB_feedback()
    getattr(basic_agb, params.agb_yields)()

    basic_1a = SN1a_feedback()
    getattr(basic_1a, params.sn1a_yields)()

    return basic_sn2, basic_agb, basic_1a


def _mix_hypernova(basic_sn2, basic_hn, sn2_to_hn: float):
    """
    Blend CC-SN and hypernova yields in place (replicates SSP_wrap logic).

    sn2_to_hn = 1  -> pure CC-SN
    sn2_to_hn = 0  -> pure HN
    """
    f_sn = sn2_to_hn
    f_hn = 1.0 - sn2_to_hn
    for met in basic_sn2.metallicities:
        x = copy.deepcopy(basic_sn2.table[met])
        y = copy.deepcopy(basic_hn.table[met])
        for mass in basic_hn.masses:
            idx = np.where(basic_sn2.table[met]["Mass"] == mass)
            basic_sn2.table[met]["mass_in_remnants"][idx] = (
                f_sn * x["mass_in_remnants"][idx]
                + f_hn * y["mass_in_remnants"][idx]
            )
            basic_sn2.table[met]["unprocessed_mass_in_winds"][idx] = (
                f_sn * x["unprocessed_mass_in_winds"][idx]
                + f_hn * y["unprocessed_mass_in_winds"][idx]
            )
            for elem in basic_sn2.elements:
                basic_sn2.table[met][elem][idx] = (
                    f_sn * x[elem][idx] + f_hn * y[elem][idx]
                )


# ---------------------------------------------------------------------------
#  Core SSP computation
# ---------------------------------------------------------------------------

def compute_ssp(
    Z: float,
    params: SSPParams,
    basic_imf: IMF,
    basic_sn2: SN2_feedback,
    basic_agb: AGB_feedback,
    basic_1a: SN1a_feedback,
    solar_fractions: list,
    elements_to_trace: list,
    net_yield: bool = True,
) -> SSP:
    """
    Compute SSP enrichment for a single metallicity.

    Returns the SSP object with all feedback tables populated.
    """
    ssp = SSP(
        False,
        float(Z),
        np.copy(basic_imf.x),
        np.copy(basic_imf.dm),
        np.copy(basic_imf.dn),
        np.copy(params.time_steps),
        list(elements_to_trace),
        params.stellar_lifetimes,
        params.interpolation_scheme,
        net_yield,
    )

    ssp.sn2_feedback(
        list(basic_sn2.elements),
        dict(basic_sn2.table),
        np.copy(basic_sn2.metallicities),
        float(params.sn2_mmin),
        float(params.sn2_mmax),
        list(solar_fractions),
    )
    ssp.agb_feedback(
        list(basic_agb.elements),
        dict(basic_agb.table),
        list(basic_agb.metallicities),
        float(params.agb_mmin),
        float(params.agb_mmax),
        list(solar_fractions),
    )
    ssp.sn1a_feedback(
        list(basic_1a.elements),
        list(basic_1a.metallicities),
        dict(basic_1a.table),
        params.time_delay_functional_form,
        float(params.sn1a_mmin),
        float(params.sn1a_mmax),
        params.sn1a_parameter,
        1.0,   # total_mass (normalised to 1 Msun)
        False,  # stochastic_IMF
    )

    if params.bh_mmin < params.bh_mmax:
        ssp.bh_feedback(
            float(params.bh_mmin),
            float(params.bh_mmax),
            list(elements_to_trace),
            np.array(solar_fractions),
            float(params.percentage_of_bh_mass),
        )

    return ssp


def compute_yZ(ssp: SSP, elements: list) -> np.ndarray:
    """Cumulative gross metal yield (everything except H and He)."""
    yZ = np.zeros_like(ssp.table["Fe"])
    for el in elements:
        if el not in ("H", "He"):
            yZ += np.cumsum(ssp.table[el])
    return yZ


# ---------------------------------------------------------------------------
#  Main grid builder
# ---------------------------------------------------------------------------

def create_yield_grid(
    params: SSPParams,
    save_path: Optional[str] = None,
    verbose: bool = True,
    diagnostic_plot: bool = True,
) -> Dict[str, Any]:
    """
    Build interpolated yield grids over (log Z, log t).

    For each element, stores:
        "{El}_net"  : RectBivariateSpline  -- cumulative net yield
        "{El}_diff" : RectBivariateSpline  -- gross - net (birth-composition term)

    Also stores:
        "ms_surv"     : surviving main-sequence stellar mass
        "ms_feedback" : cumulative mass returned (dying stars minus remnants)
        "yZ_gross"    : cumulative total metal yield (gross)

    Parameters
    ----------
    params : SSPParams
        All free parameters for the SSP computation.
    save_path : str, optional
        If given, pickle the result to this path.
    verbose : bool
        Print diagnostic yield values.
    diagnostic_plot : bool
        If True and save_path is set, produce an [X/Fe] vs Z diagnostic
        figure alongside the pickle file.

    Returns
    -------
    all_yields : dict
        The complete yield-grid dictionary.
    """

    t_start = time.time()
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] Starting yield grid build ...")

    # --- Set up yield tables and IMF (once) ---
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] Building IMF (mass_steps={params.mass_steps}) ...")
    basic_imf = _build_imf(params)
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] Loading yield tables (CC-SN: {params.ccsne_yields}, AGB: {params.agb_yields}, SN Ia: {params.sn1a_yields}) ...")
    basic_sn2, basic_agb, basic_1a = _load_yield_sets(params)

    elements_to_trace = list(
        np.unique(basic_agb.elements + basic_sn2.elements + basic_1a.elements)
    )
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] Yield tables loaded. Tracking {len(elements_to_trace)} elements: {elements_to_trace}")

    basic_solar = solar_abundances()
    getattr(basic_solar, params.solar_abundance_name)()
    all_elements = np.hstack(basic_solar.all_elements)
    solar_fractions = [
        float(basic_solar.fractions[np.where(all_elements == el)])
        for el in elements_to_trace
    ]

    # --- Containers ---
    z_grid = params.z_grid
    all_ms_surv = []
    all_ms_feedback = []
    all_yZ_gross = []
    all_n_sn1a = []
    all_n_sn2 = []

    all_yields: Dict[str, Any] = {}
    for el in elements_to_trace:
        all_yields[el + "_net"] = []
        all_yields[el + "_diff"] = []

    # --- Loop over metallicity grid ---
    if verbose:
        _print_banner("Generating SSP yield grid")
        print(f"[{time.strftime('%H:%M:%S')}] Grid: {len(z_grid)} metallicities x {len(params.time_steps)} time steps. Looping over Z ...")
        print()

    for zi in tqdm(z_grid, desc="Z grid"):
        ssp_net = compute_ssp(
            zi, params, basic_imf, basic_sn2, basic_agb, basic_1a,
            solar_fractions, elements_to_trace, net_yield=True,
        )
        ssp_gross = compute_ssp(
            zi, params, basic_imf, basic_sn2, basic_agb, basic_1a,
            solar_fractions, elements_to_trace, net_yield=False,
        )

        all_yZ_gross.append(compute_yZ(ssp_gross, elements_to_trace))

        for el in elements_to_trace:
            net = np.cumsum(
                ssp_net.agb_table[el]
                + ssp_net.sn2_table[el]
                + ssp_net.sn1a_table[el]
            )
            gross = np.cumsum(
                ssp_gross.agb_table[el]
                + ssp_gross.sn2_table[el]
                + ssp_gross.sn1a_table[el]
            )
            all_yields[el + "_net"].append(net)
            all_yields[el + "_diff"].append(gross - net)

        all_ms_surv.append(ssp_net.table["mass_in_ms_stars"])
        all_ms_feedback.append(
            np.cumsum(ssp_net.table["mass_of_ms_stars_dying"])
            - np.cumsum(ssp_net.table["mass_in_remnants"])
        )
        all_n_sn1a.append(np.cumsum(ssp_net.table["sn1a"]))
        all_n_sn2.append(np.cumsum(ssp_net.table["sn2"]))

    if verbose:
        print(f"\n[{time.strftime('%H:%M:%S')}] Z grid loop complete ({len(z_grid)} metallicities done). Converting to arrays ...")

    # --- Convert to arrays ---
    for el in elements_to_trace:
        all_yields[el + "_net"] = np.array(all_yields[el + "_net"])
        all_yields[el + "_diff"] = np.array(all_yields[el + "_diff"])

    all_ms_surv = np.array(all_ms_surv)
    all_ms_feedback = np.array(all_ms_feedback)
    all_yZ_gross = np.array(all_yZ_gross)
    all_n_sn1a = np.array(all_n_sn1a)
    all_n_sn2 = np.array(all_n_sn2)

    log_z = np.log10(z_grid)
    log_t = np.log10(params.time_steps)

    # --- Fit bivariate splines ---
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] Building interpolating splines (aggregate + per-element) ...")
    kw = dict(kx=1, ky=1)

    all_yields["ms_surv"] = RectBivariateSpline(log_z, log_t, all_ms_surv, **kw)
    all_yields["ms_feedback"] = RectBivariateSpline(log_z, log_t, all_ms_feedback, **kw)
    all_yields["yZ_gross"] = RectBivariateSpline(log_z, log_t, all_yZ_gross, **kw)
    all_yields["n_sn1a"] = RectBivariateSpline(log_z, log_t, all_n_sn1a, **kw)
    all_yields["n_sn2"] = RectBivariateSpline(log_z, log_t, all_n_sn2, **kw)
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] Built ms_surv, ms_feedback, yZ_gross, n_sn1a, n_sn2 splines. Fitting per-element splines ...")

    for el in tqdm(elements_to_trace, desc="Splines", disable=not verbose):
        net_interp = RectBivariateSpline(
            log_z, log_t, all_yields[el + "_net"], **kw
        )
        diff_interp = RectBivariateSpline(
            log_z, log_t, all_yields[el + "_diff"], **kw
        )

        if verbose:
            _print_diagnostic(el, net_interp)

        all_yields[el + "_net"] = net_interp
        all_yields[el + "_diff"] = diff_interp

    # --- Store metadata ---
    all_yields["solar_fractions"] = solar_fractions
    all_yields["elements"] = elements_to_trace
    all_yields["z_grid"] = z_grid
    all_yields["time_steps"] = params.time_steps

    # Store every SSP parameter for reproducibility
    all_yields["params"] = {
        "ccsne_yields": params.ccsne_yields,
        "agb_yields": params.agb_yields,
        "sn1a_yields": params.sn1a_yields,
        "solar_abundance_name": params.solar_abundance_name,
        "imf_type_name": params.imf_type_name,
        "imf_mmin": params.imf_mmin,
        "imf_mmax": params.imf_mmax,
        "high_mass_slope": params.high_mass_slope,
        "sn2_mmin": params.sn2_mmin,
        "sn2_mmax": params.sn2_mmax,
        "agb_mmin": params.agb_mmin,
        "agb_mmax": params.agb_mmax,
        "sn1a_mmin": params.sn1a_mmin,
        "sn1a_mmax": params.sn1a_mmax,
        "bh_mmin": params.bh_mmin,
        "bh_mmax": params.bh_mmax,
        "percentage_of_bh_mass": params.percentage_of_bh_mass,
        "time_delay_functional_form": params.time_delay_functional_form,
        "sn1a_parameter": params.sn1a_parameter,
        "sn2_to_hn": params.sn2_to_hn,
        "stellar_lifetimes": params.stellar_lifetimes,
        "interpolation_scheme": params.interpolation_scheme,
        "mass_steps": params.mass_steps,
    }

    # --- Diagnostic plot ---
    if diagnostic_plot and save_path is not None:
        diag_dir = os.path.dirname(save_path) or "."
        pickle_stem = os.path.basename(save_path).replace(".pickle", "")
        diag_fig_path = os.path.join(diag_dir, f"ssp_yield_diagnostics_{pickle_stem}.png")
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] Generating SSP yield diagnostic plot ...")
        plot_ssp_yield_diagnostics(
            params, basic_imf, basic_sn2, basic_agb, basic_1a,
            solar_fractions, elements_to_trace,
            save_path=diag_fig_path,
            verbose=verbose,
        )

    # --- Optionally save ---
    if save_path is not None:
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] Saving yield grid to: {save_path}")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(all_yields, f)
        if verbose:
            _print_banner(f"Yield grid saved to {save_path}")
            print(f"[{time.strftime('%H:%M:%S')}] Total time: {time.time() - t_start:.1f} s")
    elif verbose:
        print(f"[{time.strftime('%H:%M:%S')}] Done (no save path). Total time: {time.time() - t_start:.1f} s")

    return all_yields


# ---------------------------------------------------------------------------
#  NSM / delayed r-process channel
# ---------------------------------------------------------------------------

def create_nsm_grid(
    params: SSPParams,
    nsm_norm: float,
    nsm_time_delay: float,
    nsm_exponent: float,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build a separate grid for a delayed r-process enrichment channel (NSM)
    modeled with the same power-law DTD machinery as SN Ia.

    Only the number of events as a function of time is needed from this.
    """
    basic_imf = _build_imf(params)
    basic_sn2, basic_agb, basic_1a = _load_yield_sets(params)

    elements_to_trace = list(
        np.unique(basic_agb.elements + basic_sn2.elements + basic_1a.elements)
    )

    nsm_results = {}
    z_grid = params.z_grid

    for zi in tqdm(z_grid, desc="NSM Z grid"):
        ssp = SSP(
            False,
            float(zi),
            np.copy(basic_imf.x),
            np.copy(basic_imf.dm),
            np.copy(basic_imf.dn),
            np.copy(params.time_steps),
            list(elements_to_trace),
            params.stellar_lifetimes,
            params.interpolation_scheme,
        )
        ssp.sn1a_feedback(
            list(basic_1a.elements),
            list(basic_1a.metallicities),
            dict(basic_1a.table),
            "maoz",
            float(params.sn1a_mmin),
            float(params.sn1a_mmax),
            [nsm_norm, nsm_time_delay, nsm_exponent, 0.0],
            1.0,
            False,
        )
        nsm_results[zi] = ssp

    result = {
        "z_grid": z_grid,
        "time_steps": params.time_steps,
        "ssp_objects": nsm_results,
    }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
        if verbose:
            _print_banner(f"NSM grid saved to {save_path}")

    return result


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def _print_banner(msg: str, ch: str = "-"):
    n = len(msg)
    print(ch * n)
    print(msg)
    print(ch * n)
    print()


def _print_diagnostic(el: str, interp: RectBivariateSpline):
    if el == "Fe":
        v15 = interp([-3], [np.log10(15)], grid=False)
        v2 = interp([-3], [np.log10(2)], grid=False)
        print(f"  Net Fe yield  Z=1e-3  t=15 Gyr : {v15}")
        print(f"  Net Fe yield  Z=1e-3  t=2  Gyr : {v2}")
    elif el == "Mg":
        v2 = interp([-3], [np.log10(2)], grid=False)
        print(f"  Net Mg yield  Z=1e-3  t=2  Gyr : {v2}")


# ---------------------------------------------------------------------------
#  Diagnostic yield plots
# ---------------------------------------------------------------------------

def plot_ssp_yield_diagnostics(
    params: SSPParams,
    basic_imf: IMF,
    basic_sn2: SN2_feedback,
    basic_agb: AGB_feedback,
    basic_1a: SN1a_feedback,
    solar_fractions: list,
    elements_to_trace: list,
    save_path: str,
    diagnostic_elements: Optional[List[str]] = None,
    log_z_range: Tuple[float, float] = (-3.0, -1.0),
    n_z_points: int = 30,
    verbose: bool = True,
):
    """
    Two-panel diagnostic of core-collapse SSP yields vs metallicity.

    Left  : y_{CC,Fe}  -- total CC-SN Fe yield (Msun per 1 Msun SSP) vs Z.
    Right : [X/Fe]_{CC} for Mg, Ca, C (each a separate coloured line) vs Z.

    The figure is saved to *save_path* (PNG).
    """
    if diagnostic_elements is None:
        diagnostic_elements = ["Mg", "Ca", "C"]

    solar_dict = dict(zip(elements_to_trace, solar_fractions))
    fe_solar = solar_dict.get("Fe")
    if fe_solar is None or fe_solar <= 0:
        if verbose:
            print("  [diag] Fe not in tracked elements -- skipping diagnostic plot.")
        return

    plot_elements = [
        el for el in diagnostic_elements
        if el in solar_dict and el != "Fe" and solar_dict[el] > 0
    ]
    if not plot_elements:
        if verbose:
            print("  [diag] No diagnostic elements available -- skipping plot.")
        return

    z_diag = np.logspace(log_z_range[0], log_z_range[1], n_z_points)
    log_z_diag = np.log10(z_diag)

    fe_cc = np.zeros(n_z_points)
    el_cc = {el: np.zeros(n_z_points) for el in plot_elements}

    if verbose:
        print(f"  [diag] Computing CC-SN yields on diagnostic Z grid "
              f"(logZ = {log_z_range[0]} .. {log_z_range[1]}, "
              f"{n_z_points} pts) ...")

    for i, zi in enumerate(tqdm(z_diag, desc="Diag Z", disable=not verbose)):
        ssp = compute_ssp(
            zi, params, basic_imf, basic_sn2, basic_agb, basic_1a,
            solar_fractions, elements_to_trace, net_yield=True,
        )
        fe_cc[i] = np.sum(ssp.sn2_table["Fe"])
        for el in plot_elements:
            el_cc[el][i] = np.sum(ssp.sn2_table[el])

    element_colors = {"Mg": "tab:blue", "Ca": "tab:orange", "C": "tab:green",
                      "O": "tab:red", "Si": "tab:purple", "N": "tab:brown"}

    fig, (ax_fe, ax_xfe) = plt.subplots(1, 2, figsize=(11, 4.5),
                                         constrained_layout=True)

    # --- Left panel: absolute CC-SN Fe yield ---
    ax_fe.plot(log_z_diag, fe_cc, color="k", lw=2)
    ax_fe.set_xlabel(r"$\log_{10}\,Z$")
    ax_fe.set_ylabel(r"$y_{\mathrm{CC,Fe}}$  [$M_\odot\,/\,M_{\odot,\mathrm{formed}}$]")
    ax_fe.set_title("CC-SN Fe yield")

    # --- Right panel: [X/Fe]_CC for each element ---
    for el in plot_elements:
        x_solar = solar_dict[el]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (el_cc[el] / fe_cc) / (x_solar / fe_solar)
            xfe = np.where(ratio > 0, np.log10(ratio), np.nan)
        mask = np.isfinite(xfe)
        if mask.any():
            ax_xfe.plot(log_z_diag[mask], xfe[mask],
                        color=element_colors.get(el, "tab:gray"),
                        lw=1.8, label=el)
    ax_xfe.axhline(0, color="grey", lw=0.6, ls=":")
    ax_xfe.set_xlabel(r"$\log_{10}\,Z$")
    ax_xfe.set_ylabel(r"$[\mathrm{X/Fe}]_{\mathrm{CC}}$")
    ax_xfe.set_title("CC-SN [X/Fe] yield ratio")
    ax_xfe.legend(fontsize=9)

    fig.suptitle(
        f"Core-collapse SSP yield diagnostics  (CC-SN: {params.ccsne_yields})",
        fontsize=11,
    )

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  [diag] Saved diagnostic plot to {save_path}")


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build SSP yield interpolation grids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-ini", dest="ini", type=str, required=True,
                        help="Path to configuration .ini file")
    parser.add_argument("--no-diag", dest="no_diag", action="store_true",
                        help="Skip the [X/Fe] vs Z diagnostic plot")
    args = parser.parse_args()

    t_main_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Reading config: {args.ini}")
    cfg = read_config(args.ini)
    params = params_from_config(cfg)

    if not cfg.has_option("chem model", "chem_grids_path"):
        raise ValueError("Config must have 'chem_grids_path' in [chem model]")
    save_path = os.path.join(
        cfg["chem model"]["chem_grids_path"],
        cfg.get("chem model", "chem_grid_pickle", fallback="yield_grid.pickle"),
    )
    print(f"[{time.strftime('%H:%M:%S')}] Config loaded. Output: {save_path}")
    if save_path:
        print(f"[{time.strftime('%H:%M:%S')}] Yield grid will be saved to: {save_path}")
    print()

    create_yield_grid(params, save_path=save_path, verbose=True,
                      diagnostic_plot=not args.no_diag)

    print(f"[{time.strftime('%H:%M:%S')}] Script finished. Total elapsed: {time.time() - t_main_start:.1f} s")


if __name__ == "__main__":
    main()
