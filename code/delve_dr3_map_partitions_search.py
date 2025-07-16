"""
adapting elliotts filtering code to test the map_partions function from lsdb
"""
__author__ = "Peter Ferguson"

import os
import logging 
import dask
import numpy as np
import lsdb
from scipy.interpolate import interp1d
from ugali.analysis.isochrone import factory as isochrone_factory
import healpy as hp
import pandas as pd
import time

dask.config.set({"logging.distributed": "critical"})
from dask.distributed import Client
# This also has to be done, for the above to be effective
logger = logging.getLogger("distributed")
logger.setLevel(logging.CRITICAL)


import warnings

# Finally, suppress the specific warning about Dask dashboard port usage
warnings.filterwarnings("ignore", message="Port 8787 is already in use.")


def get_filter_splines(age, mu, z, abs_mag_min=2.9, app_mag_max = 23.5, color_min=0, color_max=1, dmu=0.5, C=[0.05, 0.1], E=2., err=None):
    """
    Generate near and far spline boundaries for color-magnitude filtering.

    Parameters
    ----------
    age : float
        Stellar population age (Gyr).
    mu : float
        Distance modulus.
    z : float
        Metallicity (e.g., 0.0001 for Z=0.0001).
    abs_mag_min : float, optional
        Minimum absolute magnitude cut (default: 2.9).
    app_mag_max : float, optional
        Maximum apparent magnitude cut (default: 23.5).
    color_min : float, optional
        Minimum color value for clipping (default: 0).
    color_max : float, optional
        Maximum color value for clipping (default: 1).
    dmu : float, optional
        Half-width of distance modulus bin (default: 0.5).
    C : list, optional
        Color error offsets [near, far] (default: [0.05, 0.1]).
    E : float, optional
        Error scaling factor (default: 2.0).
    err: function to give magerr as a function of magnitude

    Returns
    -------
    spline_near : scipy.interpolate.interp1d
        Spline for the lower color boundary.
    spline_far : scipy.interpolate.interp1d
        Spline for the upper color boundary.
    """
    if err is None:
        err = lambda x: 0.0010908679647672335 + np.exp((x - 27.091072029215375) / 1.0904624484538419)
    iso = isochrone_factory('marigo2017', survey='DES', age=age, distance_modulus=mu, z=z)
    
    gsel = (iso.mag > abs_mag_min) & (iso.mag + mu < app_mag_max)
    color =iso.color[gsel]
    mag = iso.mag[gsel]
    # Spread in magnitude     
    mnear = mag + mu - dmu / 2.
    mfar = mag + mu + dmu / 2.
    color_far = np.clip(color + E * err(mfar) + C[1], color_min, color_max)
    color_near = np.clip(color - E * err(mnear) - C[0], color_min,color_max)
    spline_far = interp1d(mag + mu, color_far , bounds_error=False, fill_value=np.nan)
    spline_near = interp1d(mag + mu, color_near, bounds_error=False, fill_value=np.nan)
    return spline_near, spline_far


def filter_data(color, mag, spl_near, spl_far):
    """
    Filter data points based on spline boundaries.

    Parameters
    ----------
    color : array-like
        Array of color values (e.g., g-r).
    mag : array-like
        Array of magnitude values (e.g., g-band).
    spl_near : scipy.interpolate.interp1d
        Lower boundary spline from `get_filter_splines()`.
    spl_far : scipy.interpolate.interp1d
        Upper boundary spline from `get_filter_splines()`.

    Returns
    -------
    sel : numpy.ndarray (bool)
        Boolean mask of selected points where `color` lies between the splines.
    """
    near_vals = spl_near(mag)
    far_vals =  spl_far(mag)
    sel = (color > near_vals) & (color < far_vals)
    return sel


def table_maker(data, spline_dict, gmag_title='gmag', rmag_title='rmag', ra_title='RA', dec_title='DEC', nside=512):
    """
    Create a HEALPix-count table of filtered sources.

    Parameters
    ----------
    data : pandas.DataFrame or structured array
        Input data with photometry and coordinates.
    spline_dict : dict
        Dictionary of {name: (spline_near, spline_far)} pairs from `get_filter_splines()`.
    gmag_title : str, optional
        Column name for g-band magnitudes (default: 'gmag').
    rmag_title : str, optional
        Column name for r-band magnitudes (default: 'rmag').
    ra_title : str, optional
        Column name for Right Ascension (default: 'RA').
    dec_title : str, optional
        Column name for Declination (default: 'DEC').
    nside : int, optional
        HEALPix resolution parameter (default: 512).

    Returns
    -------
    pandas.DataFrame
        Table with HEALPix IDs (pix{nside}) and counts per filter in `spline_dict`.
    """
    ra = data[ra_title]
    dec = data[dec_title]
    
    pix = hp.ang2pix(nside, ra, dec, lonlat=True) # all pixel ids in range
    upix = np.unique(pix, return_counts=False) # all unique pixel ids in range
    
    out_col_list = [f'pix{nside}'] # this column will hold all the ids
    out_col_list += list(spline_dict.keys()) # now it's all the columns
    
    dtype_list = [(name, 'int') for name in out_col_list] # list of 'int' for each column
    hpx_array = np.recarray(shape=len(upix), dtype=dtype_list) # Make hpx array
    hpx_array.fill(0) # Make the default count value zero
    hpx_array[f'pix{nside}'] = upix # Set the hpx ids in the first column
    
    for key in spline_dict.keys():
        spl_near, spl_far = spline_dict[key]
        selector = filter_data(color=(data[gmag_title]-data[rmag_title]), 
                               mag=data[gmag_title], 
                               spl_near=spl_near,
                               spl_far=spl_far,
                               ) # Make matched-filter selection
        upix_sel, counts_sel = np.unique(pix[selector], return_counts=True)
        
        hpx_array[key][np.searchsorted(upix, upix_sel)] = counts_sel
          
    return pd.DataFrame(hpx_array)

def parse_args():
    """Parse command line arguments for the DELVE data processing script."""
    parser = argparse.ArgumentParser(description='Process DELVE DR3 data into HEALPix cubes')
    
    # Distance modulus parameters
    parser.add_argument('--mu_start', type=float, default=15.0,
                       help='Starting distance modulus (default: 15)')
    parser.add_argument('--mu_end', type=float, default=20.0,
                       help='Ending distance modulus (default: 20)')
    parser.add_argument('--mu_step', type=float, default=0.1,
                       help='Step size for distance modulus bins (default: 0.1)')
    
    # Stellar population parameters
    parser.add_argument('--age', type=float, default=12.0,
                       help='Stellar population age in Gyr (default: 12)')
    parser.add_argument('--z', type=float, default=0.0001,
                       help='Metallicity (default: 0.0001)')
    
    # File paths
    parser.add_argument('--outpath', type=str, default='./healpix_cube.parq',
                       help='Output path for HEALPix cube (default: ./healpix_cube.parq)')
    parser.add_argument('--delve_path', type=str, 
                       default='/epyc/data/delve/dr3/delve_dr3_stellar_skim/',
                       help='Path to DELVE DR3 stellar skim data (default: /epyc/data/delve/dr3/delve_dr3_stellar_skim/)')
    
    # Column selection
    parser.add_argument('--delve_cols', nargs='+', 
                       default=["COADD_OBJECT_ID", "RA", "DEC", 'MAG_PSF_SFD_G', 'MAG_PSF_SFD_R'],
                       help='Columns to load from DELVE data (default: COADD_OBJECT_ID RA DEC MAG_PSF_SFD_G MAG_PSF_SFD_R)')
    
    return parser.parse_args()


if __name__ == "__main__":
    import argparse
    
    args = parse_args()
    
    print(f"Processing distance modulus range: {args.mu_start} to {args.mu_end} with step {args.mu_step}")
    print(f"Stellar population: Age={args.age} Gyr, Z={args.z}")
    print(f"Output will be saved to: {args.outpath}")
    print(f"Using DELVE columns: {', '.join(args.delve_cols)}")
    
    mu_start=args.mu_start
    mu_end=args.mu_end
    mu_step=args.mu_step
    age =args.age
    z=args.z
    outpath = args.outpath
    delve_path = args.delve_path
    delve_cols = args.delve_cols
    muset = np.arange(mu_start, mu_end + mu_step, mu_step)
    spline_dict ={}
    start = time.perf_counter()
    for mu in muset:
        sp_near, sp_far = get_filter_splines(age=age, mu = mu, z=z)
        spline_dict[f'{mu:0.2f}'.replace('.', 'p')] = (sp_near, sp_far)
    
    delve_dr3 = lsdb.read_hats(
        delve_path,
        columns=delve_cols,
        search_filter=lsdb.ConeSearch(ra=280, dec=-60, radius_arcsec=2 * 3600),
    )
    delve_dr3
    
    unrealized = delve_dr3.map_partitions(
        table_maker,
        gmag_title='MAG_PSF_SFD_G', rmag_title='MAG_PSF_SFD_R', ra_title='RA', dec_title='DEC', nside=512,
        spline_dict = spline_dict
    )
    
    npartitions = len(delve_dr3.get_healpix_pixels())
    print(f"catalog has {npartitions} partitions")
    n_workers = min(8, npartitions)
    print(f"using {n_workers} workers to process")
    # might also want to set the threads_per_worker=1
    
    with Client(n_workers=n_workers) as client:
        result = unrealized.compute()
    print(result.head())
    end = time.perf_counter()
    print(f"Time taken: {end - start:.4f} seconds")
    if outpath:
        result.to_parquet(outpath)
    