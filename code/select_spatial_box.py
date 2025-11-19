'''
Creates a dataframe of objects in DELVE DR3, flagged where it intersects with GAIA DR3.
    Columns:
        'coadd_object_id',
        'ra',
        'dec',
        'phi1',
        'phi2',
        'g_mag',
        'r_mag',
        'i_mag',
        'z_mag',
        'g_mag_error',
        'r_mag_error',
        'i_mag_error',
        'z_mag_error',
        'gaia',
        'pmra',
        'pmdec',
        'pmra_error',
        'pmdec_error',
        'radial_velocity',
        'radial_velocity_error',
        'pmphi1',
        'pmphi2',
        'p_pmra',
        'p_pmdec',
        'p_spatial',
        'p_photometric',
        'p_total',
        'is_background_passes_filter'
    Computing Large Datasets should be the most computationally expensive part of the script.
    In the future, LSBD should have the functionality to do a left crossmatch while leaving the datasets lazily loaded.
    P_total represents the probability that a given star is a stream member.
'''

__author__ = "Elliott Burdett"

import numpy as np
import pandas as pd
import healpy as hp
import dask
import hats
import lsdb
import astropy
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from ugali.analysis.isochrone import factory as isochrone_factory
import os
import logging
import time
import sys
code_path = "/astro/users/esb30/software/spreading_seas/code"
sys.path.append(code_path)
import argparse
parser = argparse.ArgumentParser(description="Generate DELVExGaia stream selection for a given stream.")
parser.add_argument("--stream",type=str,required=True")
args = parser.parse_args()
stream = args.stream
from filter_data import get_filter_splines, filter_data
from rotation_matrix import phi12_rotmat, pmphi12, rotation_matrix
from gaussian_membership_fits import quad_f, pmra_gaussian, pmdec_gaussian, phi2_gaussian
from stream_details import age, z, get_mu
rotmat = rotation_matrix(stream)
get_mu(stream)
delve_object = lsdb.read_hats("/epyc/data/delve/dr3/delve_dr3_gold/delve_dr3_gold/", columns=['COADD_OBJECT_ID', 'RA', 'DEC', 'PSF_MAG_APER_8_G_CORRECTED', 'PSF_MAG_APER_8_R_CORRECTED', 'PSF_MAG_APER_8_I_CORRECTED', 'PSF_MAG_APER_8_Z_CORRECTED','PSF_MAG_ERR_APER_8_G', 'PSF_MAG_ERR_APER_8_R', 'PSF_MAG_ERR_APER_8_I', 'PSF_MAG_ERR_APER_8_Z', 'EXT_XGB', 'SOURCE'])
lazy_delve = delve_object.query("PSF_MAG_APER_8_G_CORRECTED > 16 and PSF_MAG_APER_8_G_CORRECTED < 24.5 and PSF_MAG_APER_8_G_CORRECTED - PSF_MAG_APER_8_R_CORRECTED > -0.3 and PSF_MAG_APER_8_G_CORRECTED - PSF_MAG_APER_8_R_CORRECTED < 1 and EXT_XGB == 0")
lazy_gaia = lsdb.read_hats('/epyc/data3/hats/catalogs/gaia_dr3/gaia/', columns=["ra", "dec", "pm", "pmra", "pmdec", "pmra_error", "pmdec_error", "radial_velocity", "radial_velocity_error"])
print('Computing DELVE DR3')
delve = lazy_delve.compute()
print('Computing Gaia DR3')
gaia = lazy_gaia.compute()

delve['Phi1'], delve['Phi2'] = phi12_rotmat(alpha=delve['RA'].to_numpy(),delta=delve['DEC'].to_numpy(),R_phi12_radec=rotmat)
gaia['Phi1'], gaia['Phi2'] = phi12_rotmat(alpha=gaia['ra'].to_numpy(),delta=gaia['dec'].to_numpy(),R_phi12_radec=rotmat)
delve = delve[(delve['Phi1'] > -30) & (delve['Phi1'] < 30) & (delve['Phi2'] > -6) & (delve['Phi2'] < 6)]
gaia = gaia[(gaia['Phi1'] > -30) & (gaia['Phi1'] < 30) & (gaia['Phi2'] > -6) & (gaia['Phi2'] < 6)]

print('Crossmatching Data')

delve_coords = SkyCoord(ra=delve['RA'].to_numpy() * u.deg, dec=delve['DEC'].to_numpy() * u.deg)
gaia_coords = SkyCoord(ra=gaia['ra'].to_numpy() * u.deg, dec=gaia['dec'].to_numpy() * u.deg)
idx, angular_separation, _ = delve_coords.match_to_catalog_sky(gaia_coords)
match_radius = 1.0 * u.arcsec
match_mask = angular_separation < match_radius

dxg = pd.DataFrame({
    'coadd_object_id': delve['COADD_OBJECT_ID'],
    'ra': delve['RA'],
    'dec': delve['DEC'],
    'phi1': delve['Phi1'],
    'phi2': delve['Phi2'],
    'g_mag': delve['PSF_MAG_APER_8_G_CORRECTED'],
    'r_mag': delve['PSF_MAG_APER_8_R_CORRECTED'],
    'i_mag': delve['PSF_MAG_APER_8_I_CORRECTED'],
    'z_mag': delve['PSF_MAG_APER_8_Z_CORRECTED'],
    'g_mag_error': delve['PSF_MAG_ERR_APER_8_G'],
    'r_mag_error': delve['PSF_MAG_ERR_APER_8_R'],
    'i_mag_error': delve['PSF_MAG_ERR_APER_8_I'],
    'z_mag_error': delve['PSF_MAG_ERR_APER_8_Z'],
    'gaia': match_mask,
    'pmra': np.nan,
    'pmdec': np.nan,
    'pmra_error': np.nan,
    'pmdec_error': np.nan,
    'radial_velocity': np.nan,
    'radial_velocity_error': np.nan,
})

gaia_matched = gaia.iloc[idx[match_mask]].reset_index(drop=True)

dxg.loc[match_mask, 'pmra'] = gaia_matched['pmra'].values
dxg.loc[match_mask, 'pmdec'] = gaia_matched['pmdec'].values
dxg.loc[match_mask, 'pmra_error'] = gaia_matched['pmra_error'].values
dxg.loc[match_mask, 'pmdec_error'] = gaia_matched['pmdec_error'].values
dxg.loc[match_mask, 'radial_velocity'] = gaia_matched['radial_velocity'].values
dxg.loc[match_mask, 'radial_velocity_error'] = gaia_matched['radial_velocity_error'].values

dxg['pmphi1'], dxg['pmphi2'] = pmphi12(alpha=dxg['ra'],delta=dxg['dec'],mu_alpha_cos_delta=dxg['pmra'],mu_delta=dxg['pmdec'],R_phi12_radec=rotmat)

dxg['p_pmra'] = pmra_gaussian(pmra=dxg['pmra'], phi1=dxg['phi1'], pmra_error=dxg['pmra_error'], stream=stream)
dxg['p_pmdec'] = pmdec_gaussian(pmdec=dxg['pmdec'], phi1=dxg['phi1'], pmdec_error=dxg['pmdec_error'], stream=stream)
dxg['p_spatial'] = phi2_gaussian(phi2=dxg['phi2'], phi1=dxg['phi1'], stream=stream)
spl_near, spl_far = get_filter_splines(age=age(stream), mu=mu(0), z=z(stream), abs_mag_min=-1, app_mag_max = 23.5, color_min=0, color_max=1, dmu=0.5, C=[0.05, 0.1], E=2., err=None)
dxg['p_photometric'] = filter_data(color=dxg['g_mag']-dxg['r_mag'], mag=dxg['g_mag'] - mu(dxg['phi1']) + mu(0), spl_near=spl_near, spl_far=spl_far)
dxg['p_photometric'] = dxg['p_photometric'].map({True: 1, False: 0})
#dxg['p_photometric'] = filter_data_score(color=dxg['g_mag']-dxg['r_mag'], mag=dxg['g_mag'] - mu(dxg['phi1']) + mu(0), spl_near=spl_near, spl_far=spl_far, sigma=None)
dxg['p_total'] = dxg['p_pmdec'] * dxg['p_pmra'] * dxg['p_spatial']*dxg['p_photometric']
dxg['is_background_passes_filter'] = (dxg['p_photometric'] == 1) & (dxg['p_total'] < 0.7)
dxg.to_parquet(f"dxg_{stream}.parquet", compression="snappy")