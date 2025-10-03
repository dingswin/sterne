#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
forecast_pulsar.py

A script to predict the future astrometric position of a pulsar based on its
known properties, including proper motion, parallax, and optional orbital
reflex motion for binary systems.

Parameters can be provided via the command line or through a target information
file specified with `--targetinfo`. A parameter must be specified in only one
location; providing it in both the file and on the command line will result
in an error.

This script produces two output files:
1.  The final pmpar.in file where statistical and systematic uncertainties
    have been added in quadrature.
2.  A preliminary file with the suffix '.preliminary' containing only the
    statistical position uncertainties.


Dependencies:
- numpy
- astropy
- The provided 'sterne' module files (positions.py, reflex_motion.py)
- novas (for parallax calculation in positions.py)
- A properly set TEMPO2 environment variable ($TEMPO2) pointing to the
  T2runtime directory, as required by positions.py for planetary ephemerides.

Example Usage:
----------------
# 1. Provide all info on the command line. The reference epoch will default to 60000.0.
#    pmra, pmdec, parallax, and uncertainties will use their default values.
python forecast_pulsar.py \
    --dates dates.txt \
    --ref-ra "10:12:13.14" \
    --ref-dec "+53:07:02.5" \
    --pm-ra 5.1 \
    --output J1012_predicted.pmpar.in

# 2. Use a target info file for most parameters.
#    (Assuming target_info.txt contains ref_ra, ref_dec, pm_ra, etc.)
python forecast_pulsar.py \
    --dates dates.txt \
    --targetinfo target_info.txt \
    --output J1012_predicted.pmpar.in

"""

import argparse
import numpy as np
import os
import sys

# This script assumes that 'positions.py' and 'reflex_motion.py' are located
# in a directory structure like 'sterne/model/'.
# Ensure this 'sterne' directory is in your Python path.
try:
    from sterne.model import positions
    from sterne.model import reflex_motion
except ImportError as e:
    print("--- Full Import Error Message ---")
    print(e)
    print("---------------------------------")
    print("\nError: Could not import the 'sterne' package.")
    print("Please ensure that 'positions.py' and 'reflex_motion.py' are in a")
    print("'sterne/model/' directory that is accessible in the Python path.")
    sys.exit(1)

try:
    from astropy.time import Time
    import astropy.units as u
    from astropy.coordinates import SkyCoord
except ImportError as e:
    print("--- Full Import Error Message ---")
    print(e)
    print("---------------------------------")
    print("\nError: The 'astropy' package is required. Please install it using 'pip install astropy'.")
    sys.exit(1)


def parse_targetinfo_file(filepath):
    """
    Parses a key-value parameter file.

    Args:
        filepath (str): The path to the target info file.

    Returns:
        dict: A dictionary of parameters with values converted to the correct types.
    """
    params = {}
    # Defines the expected type for each parameter key.
    type_map = {
        'ref_ra': str, 'ref_dec': str, 'ref_epoch': float,
        'pm_ra': float, 'pm_dec': float, 'parallax': float,
        'unc_ra_statistical': float, 'unc_dec_statistical': float,
        'unc_ra_systematic': float, 'unc_dec_systematic': float,
        'parfile': str, 'inclination': float, 'omega_asc': float
    }

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                print(f"Warning: Skipping malformed line in {filepath}: {line}")
                continue
            
            key, value = [item.strip() for item in line.split('=', 1)]
            
            if key not in type_map:
                print(f"Warning: Skipping unknown parameter '{key}' in {filepath}")
                continue

            # Convert value to the correct type
            try:
                params[key] = type_map[key](value)
            except ValueError:
                raise ValueError(f"Could not convert value '{value}' for parameter '{key}' in {filepath}.")
                
    return params


def predict_pulsar_positions(
    dates,
    ref_ra_str,
    ref_dec_str,
    ref_epoch,
    output_file,
    pm_ra=0.0,
    pm_dec=0.0,
    parallax=1.0,
    unc_ra_statistical=0.1,
    unc_dec_statistical=0.2,
    unc_ra_systematic=0.2,
    unc_dec_systematic=0.3,
    parfile=None,
    inclination=None,
    omega_asc=None,
):
    """
    Predicts pulsar positions and writes them to two pmpar.in formatted files.

    Args:
        dates (list of float): List of MJD dates to predict positions for.
        ref_ra_str (str): Reference Right Ascension in 'hh:mm:ss.s' format.
        ref_dec_str (str): Reference Declination in 'dd:mm:ss.s' format.
        ref_epoch (float): Reference epoch in MJD.
        output_file (str): Path to the base output file for the final results.
        pm_ra (float, optional): Proper motion in RA (mas/yr). Defaults to 0.0.
        pm_dec (float, optional): Proper motion in Dec (mas/yr). Defaults to 0.0.
        parallax (float, optional): Parallax in mas. Defaults to 1.0.
        unc_ra_statistical (float, optional): Statistical on-sky position uncertainty in RA (mas). Defaults to 0.1.
        unc_dec_statistical (float, optional): Statistical on-sky position uncertainty in Dec (mas). Defaults to 0.2.
        unc_ra_systematic (float, optional): Systematic on-sky position uncertainty in RA (mas). Defaults to 0.2.
        unc_dec_systematic (float, optional): Systematic on-sky position uncertainty in Dec (mas). Defaults to 0.3.
        parfile (str, optional): Path to the pulsar ephemeris (.par) file. Defaults to None.
        inclination (float, optional): Inclination angle (degrees). Required if parfile is used. Defaults to None.
        omega_asc (float, optional): Position angle of ascending node (degrees). Required if parfile is used. Defaults to None.
    """

    # --- 1. Input Validation and Preparation ---
    if parfile and (inclination is None or omega_asc is None):
        raise ValueError(
            "When a parfile is provided for reflex motion calculation, "
            "'--inclination' and '--omega-asc' must also be specified."
        )

    # Convert reference RA/Dec to radians for the calculation functions
    ref_coord = SkyCoord(ref_ra_str, ref_dec_str, unit=(u.hourangle, u.deg), frame='icrs')
    ref_ra_rad = ref_coord.ra.rad
    ref_dec_rad = ref_coord.dec.rad
    
    cos_dec = np.cos(ref_dec_rad)
    if cos_dec == 0:
        raise ValueError("Cannot calculate RA uncertainty at the celestial pole (Dec = +/- 90 degrees).")

    # --- 2. Calculate and Convert Uncertainties ---
    # Statistical uncertainties for the preliminary file
    unc_dec_stat_arcsec = unc_dec_statistical / 1000.0
    unc_ra_stat_sec_time = (unc_ra_statistical / cos_dec) / 15000.0

    # Total uncertainties (statistical + systematic in quadrature) for the final file
    total_unc_ra_mas = np.sqrt(unc_ra_statistical**2 + unc_ra_systematic**2)
    total_unc_dec_mas = np.sqrt(unc_dec_statistical**2 + unc_dec_systematic**2)

    total_unc_dec_arcsec = total_unc_dec_mas / 1000.0
    total_unc_ra_sec_time = (total_unc_ra_mas / cos_dec) / 15000.0

    # Prepare timing parameter dictionary for reflex motion
    dict_timing = {}
    if parfile:
        print(f"Reading orbital parameters from: {parfile}")
        if not os.path.exists(parfile):
             raise FileNotFoundError(f"The specified parfile does not exist: {parfile}")
        dict_timing = reflex_motion.read_parfile(parfile)
        incl_rad = np.deg2rad(inclination)
        om_asc_deg = omega_asc
    else:
        incl_rad = 0  # Default to edge-on if no parfile
        om_asc_deg = 0 # Default to 0 if no parfile

    print(f"Predicting positions for {len(dates)} epochs.")

    # --- 3. Main Calculation Loop ---
    predicted_positions = []
    for epoch in dates:
        ra_pred_rad, dec_pred_rad = positions.position(
            refepoch=ref_epoch, epoch=epoch, dec_rad=ref_dec_rad, incl=incl_rad,
            mu_a=pm_ra, mu_d=pm_dec, om_asc=om_asc_deg, px=parallax,
            ra_rad=ref_ra_rad, dict_of_timing_parameters=dict_timing,
        )
        predicted_positions.append((epoch, ra_pred_rad, dec_pred_rad))

    # --- 4. Format and Write Output Files ---
    
    # Define aligned header and data formats
    header_fmt = "# {:<13s} {:<18s} {:<13s} {:<18s} {:<12s}\n"
    data_fmt = "{:<15.4f} {:<18s} {:<13.7f} {:<18s} {:<12.6f}\n"

    # --- File 1: Preliminary (Statistical Uncertainties Only) ---
    output_file_prelim = output_file + ".preliminary"
    with open(output_file_prelim, "w") as f:
        f.write("# Predicted pulsar positions in pmpar.in format (PRELIMINARY - statistical errors only)\n")
        f.write(header_fmt.format("Epoch (MJD)", "RA (hh:mm:ss)", "errRA (s)", "DEC (dd:mm:ss)", "errDEC (as)"))

        for epoch_mjd, ra_rad, dec_rad in predicted_positions:
            pred_coord = SkyCoord(ra=ra_rad*u.rad, dec=dec_rad*u.rad, frame='icrs')

            ra_out_str = pred_coord.ra.to_string(unit=u.hourangle, sep=':', precision=7, pad=True)
            dec_out_str = pred_coord.dec.to_string(unit=u.deg, sep=':', precision=6, alwayssign=True, pad=True)
            
            f.write(
                data_fmt.format(epoch_mjd, ra_out_str, unc_ra_stat_sec_time, dec_out_str, unc_dec_stat_arcsec)
            )

    print(f"✅ Successfully wrote preliminary positions to '{output_file_prelim}'.")

    # --- File 2: Final (Total Uncertainties) ---
    with open(output_file, "w") as f:
        f.write("# Predicted pulsar positions in pmpar.in format (FINAL - total errors: statistical + systematic)\n")
        f.write(header_fmt.format("Epoch (MJD)", "RA (hh:mm:ss)", "errRA (s)", "DEC (dd:mm:ss)", "errDEC (as)"))

        for epoch_mjd, ra_rad, dec_rad in predicted_positions:
            pred_coord = SkyCoord(ra=ra_rad*u.rad, dec=dec_rad*u.rad, frame='icrs')

            ra_out_str = pred_coord.ra.to_string(unit=u.hourangle, sep=':', precision=7, pad=True)
            dec_out_str = pred_coord.dec.to_string(unit=u.deg, sep=':', precision=6, alwayssign=True, pad=True)

            f.write(
                data_fmt.format(epoch_mjd, ra_out_str, total_unc_ra_sec_time, dec_out_str, total_unc_dec_arcsec)
            )
            
    print(f"✅ Successfully wrote final positions to '{output_file}'.")


def main():
    """Main function to parse command-line arguments and run the prediction."""
    parser = argparse.ArgumentParser(
        description="Predict future astrometric positions of a pulsar.",
        formatter_class=argparse.RawTextHelpFormatter,
        prog='forecast_pulsar.py',
        epilog="""
This script requires the 'sterne' package and its dependencies.
It also relies on the NOVAS library and a correctly set $TEMPO2 environment
variable for accurate parallax calculations.
"""
    )
    
    parser.add_argument('--dates', required=True, type=str,
                        help="A list of dates in MJD for which to predict positions.\n"
                             "Can be a comma-separated string or a path to a file.")
    parser.add_argument('--output', required=True, type=str,
                        help="Name of the final output file (e.g., 'predicted.pmpar.in').\n"
                             "A second file with a '.preliminary' suffix will also be created.")
    parser.add_argument('--targetinfo', type=str, default=None,
                        help="Path to a text file containing pulsar parameters (key = value format).")
    
    # Arguments that can be in the file or on the command line
    arg_group = parser.add_argument_group('Target Parameters (can be in file or on command line)')
    arg_group.add_argument('--ref-ra', type=str, help="Reference Right Ascension in 'hh:mm:ss.s' format.")
    arg_group.add_argument('--ref-dec', type=str, help="Reference Declination in 'dd:mm:ss.s' format.")
    arg_group.add_argument('--ref-epoch', type=float, help="Reference epoch in MJD. Default: 60000.0")
    arg_group.add_argument('--pm-ra', type=float, help="Proper motion in RA (mas/yr). Default: 0.0")
    arg_group.add_argument('--pm-dec', type=float, help="Proper motion in Dec (mas/yr). Default: 0.0")
    arg_group.add_argument('--parallax', type=float, help="Parallax in mas. Default: 1.0")
    arg_group.add_argument('--unc-ra-statistical', type=float, help="Statistical on-sky position uncertainty in RA (mas). Default: 0.1")
    arg_group.add_argument('--unc-dec-statistical', type=float, help="Statistical on-sky position uncertainty in Dec (mas). Default: 0.2")
    arg_group.add_argument('--unc-ra-systematic', type=float, help="Systematic on-sky position uncertainty in RA (mas). Default: 0.2")
    arg_group.add_argument('--unc-dec-systematic', type=float, help="Systematic on-sky position uncertainty in Dec (mas). Default: 0.3")
    arg_group.add_argument('--parfile', type=str, help="Path to the pulsar ephemeris (.par) file.")
    arg_group.add_argument('--inclination', type=float, help="Orbital inclination angle (degrees). Required if --parfile is used.")
    arg_group.add_argument('--omega-asc', type=float, help="Position angle of the ascending node (Omega) in degrees. Required if --parfile is used.")

    args = parser.parse_args()

    # --- Parameter Loading and Conflict Resolution ---
    file_params = {}
    if args.targetinfo:
        if not os.path.exists(args.targetinfo):
            print(f"Error: Target info file not found at '{args.targetinfo}'")
            sys.exit(1)
        file_params = parse_targetinfo_file(args.targetinfo)
    
    # Check for conflicts
    cli_args_provided = {arg.lstrip('-').replace('-', '_') for arg in sys.argv if arg.startswith('--')}
    file_keys = set(file_params.keys())
    conflicts = cli_args_provided.intersection(file_keys)
    
    if conflicts:
        print(f"Error: The following parameters were specified in both the command line and the targetinfo file: {', '.join(conflicts)}")
        print("Please specify each parameter in one place only.")
        sys.exit(1)

    # Merge file params into args namespace, then set defaults for any remaining None values
    final_params = file_params.copy()
    for key, value in vars(args).items():
        if key not in final_params:
            final_params[key] = value

    # Set defaults for optional params that were not provided in either location
    defaults = {
        'pm_ra': 0.0, 'pm_dec': 0.0, 'parallax': 1.0, 
        'unc_ra_statistical': 0.1, 'unc_dec_statistical': 0.2, 
        'unc_ra_systematic': 0.2, 'unc_dec_systematic': 0.3, 
        'ref_epoch': 60000.0
    }
    for key, default_val in defaults.items():
        if final_params.get(key) is None:
            final_params[key] = default_val

    # Check for required parameters that might still be missing (ref-ra, ref-dec)
    required_params = ['ref_ra', 'ref_dec']
    missing = [p for p in required_params if final_params.get(p) is None]
    if missing:
        print(f"Error: Required parameter(s) missing: {', '.join(missing)}")
        print("Please provide them either on the command line or in the targetinfo file.")
        sys.exit(1)

    # --- Date Processing and Final Function Call ---
    if os.path.exists(args.dates):
        with open(args.dates, 'r') as f:
            dates = [float(line.strip()) for line in f if line.strip()]
    else:
        dates = [float(date.strip()) for date in args.dates.split(',')]

    try:
        predict_pulsar_positions(
            dates=dates,
            ref_ra_str=final_params['ref_ra'],
            ref_dec_str=final_params['ref_dec'],
            ref_epoch=final_params['ref_epoch'],
            output_file=final_params['output'],
            pm_ra=final_params['pm_ra'],
            pm_dec=final_params['pm_dec'],
            parallax=final_params['parallax'],
            unc_ra_statistical=final_params['unc_ra_statistical'],
            unc_dec_statistical=final_params['unc_dec_statistical'],
            unc_ra_systematic=final_params['unc_ra_systematic'],
            unc_dec_systematic=final_params['unc_dec_systematic'],
            parfile=final_params.get('parfile'),
            inclination=final_params.get('inclination'),
            omega_asc=final_params.get('omega_asc'),
        )
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please ensure all dependencies are met (see script help and docstrings).")
        sys.exit(1)


if __name__ == "__main__":
    main()
