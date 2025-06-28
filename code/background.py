'''
Different ways to fit a background to a healpix array.
fit_bkg will return BACKGROUND as another healpix array based on a deg-degree polynomial fit as a function of x and y.
spherical_harmonic_background_fit takes a healpix array and returns both residuals and background as a function of lmax
'''
import hpgeom as hp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.io import fits
import os
from astropy.table import Table, vstack
import skyproj
import pyproj
import healsparse as hsp
import healpy as hp
from numpy.polynomial import polynomial
from numpy.polynomial.polynomial import polyvander2d
import numpy.ma as ma
import matplotlib as mpl

def fit_bkg(data, proj, sigma=0.1, percent=[2, 95], deg=5):
    nside = hp.get_nside(data.mask)
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)

    vmin, vmax = np.percentile(data.compressed(), q=percent)
    data = np.clip(data, vmin, vmax)
    data.fill_value = np.ma.median(data)

    smoothed = hp.smoothing(data, sigma=np.radians(sigma), verbose=False)
    data = np.ma.array(smoothed, mask=data.mask)

    sel = ~data.mask
    x, y = proj.ang2xy(lon[sel], lat[sel], lonlat=True)

    xmin, xmax, ymin, ymax = proj.get_extent()
    sel2 = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
    # sel2 = ~np.isnan(x) & ~np.isnan(y)

    v = data[sel][sel2]
    x = x[sel2]
    y = y[sel2]

    c = polyfit2d(x, y, v, [deg, deg])

    # Evaluate the polynomial
    x, y = proj.ang2xy(lon, lat, lonlat=True)
    bkg = polynomial.polyval2d(x, y, c)
    bkg = np.ma.array(bkg, mask=data.mask, fill_value=np.nan)

    return bkg

def spherical_harmonic_background_fit(masked_array, lmax=3):
    """Fit and subtract background using spherical harmonics expansion while maintaining masks.
    
    Args:
        masked_array: Masked HEALPix array
        lmax: Maximum l value for spherical harmonics expansion
        
    Returns:
        residual_data: Masked array with background subtracted
        background_model: Masked array containing the spherical harmonic background model
    """
    nside = hp.get_nside(masked_array)
    
    # Get original alm (up to lmax)
    alm_original = hp.map2alm(masked_array, lmax=lmax)
    
    # Create unmasked background model
    full_background = hp.alm2map(alm_original, nside, verbose=False)
    
    # Convert to masked array using original mask
    background_model = np.ma.array(full_background, mask=masked_array.mask, fill_value=hp.UNSEEN)
    
    # Subtract background model from data (preserves mask)
    residual_data = masked_array - background_model
    
    return residual_data, background_model
