'''Selection of BHB and RRL stars in the AAU region
BHB Characterization:
    -Load DELVE DR3 x Gaia [Most time-intensive step]
    -Crop to AAU Spatial Region
    -Apply color cut
    -Apply magnitude cut based on an assumed mean distance of AAU (with error)
    -Apply proper motion cut (See gaussian_membership_fits)
    -Apply spatial cut (See gaussian_membership_fits)
    -Assume distance based on BHB distance relationship
RRL Characterization:
    -Load Gaia RRL Catalog
    -Crop to AAU spatial region
    -Apply proper motion cut (See gaussian_membership_fits)
    -Apply spatial cut (See gaussian_membership_fits)
    -Assume Gaia distance
Creates a table with the following columns:
    'ra'
    'dec'
    'phi1'
    'phi2'
    'distance_modulus'
    'pmra'
    'pmdec'
    'p_pmra'
    'p_pmdec'
    'p_spatial'
    'gaia_g'
    'gaia_bp'
    'gaia_rp'
    'delve_g'
    'delve_r'
    'rrl?'
    'bhb?'
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
from rotation_matrix import phi12_rotmat, pmphi12
from gaussian_membership_fits import quad_f, pmra_gaussian, pmdec_gaussian, phi2_gaussian
atlas_rotmat = [[0.83697865, 0.29481904, -0.4610298], [0.51616778, -0.70514011, 0.4861566], [0.18176238, 0.64487142, 0.74236331]]
delve_path = "/epyc/data/delve/dr3/delve_dr3_gold/delve_dr3_gold/"
delve_object = lsdb.read_hats(delve_path, columns=['COADD_OBJECT_ID', 'RA', 'DEC', 'PSF_MAG_APER_8_G_CORRECTED', 'PSF_MAG_APER_8_R_CORRECTED', 'PSF_MAG_APER_8_I_CORRECTED', 'PSF_MAG_APER_8_Z_CORRECTED', 'EXT_XGB', 'SOURCE'])
lazy_delve = delve_object.query("PSF_MAG_APER_8_G_CORRECTED > 16 and PSF_MAG_APER_8_G_CORRECTED < 24.5 and PSF_MAG_APER_8_G_CORRECTED - PSF_MAG_APER_8_R_CORRECTED > -0.3 and PSF_MAG_APER_8_G_CORRECTED - PSF_MAG_APER_8_R_CORRECTED < 1 and EXT_XGB == 0")
lazy_gaia = lsdb.read_hats('/epyc/data3/hats/catalogs/gaia_dr3/gaia/', columns=["ra", "dec", "pm", "pmra", "pmdec", "pmra_error", "pmdec_error", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"])
lazy_dxg = lazy_delve.crossmatch(lazy_gaia, n_neighbors=1, radius_arcsec=10)
print('Computing Dataset')
dxg = lazy_dxg.compute()

dxg['phi1'], dxg['phi2'] = phi12_rotmat(alpha=dxg['RA_delve_dr3_gold'].to_numpy(),delta=dxg['DEC_delve_dr3_gold'].to_numpy(),R_phi12_radec=atlas_rotmat)
dxg_on = dxg[(dxg['phi1'] > -30) &(dxg['phi2'] < 6) &(dxg['phi2'] > -6) & (dxg['phi1'] < 30)] # Use the aau filter on 'dxg_on' for pmra vs pmdec plot -5 to 5
dxg_on['g_abs'] = dxg_on['g_mag'] - 16.66
bhb = dxg_on[
    (dxg_on['g_mag'] - dxg_on['r_mag'] > -0.3) & 
    (dxg_on['g_mag'] - dxg_on['r_mag'] < 0) & 
    (dxg_on['g_abs'] > -1) & 
    (dxg_on['g_abs'] < 2) #BHB gmags usually range from 0 to 1, and I am allowing a distance gradient of +/- 1 on each side
]

gaia_rrl_catalog = pd.read_parquet('/epyc/data/gaia_rrl/li_xmatch.parq')
gaia_rrl_catalog = gaia_rrl_catalog.rename(columns={'Gaia_gaia_rrl_w_dist': 'source_id', 'ra_gaia_rrl_w_dist': 'ra', 'dec_gaia_rrl_w_dist':'dec', 'Dist_Phot_gaia_rrl_w_dist':'dist', 'e_Dist_Phot_gaia_rrl_w_dist':'e_dist', 'pmra_gaia':'pmra', 'pmra_error_gaia':'e_pmra', 'pmdec_gaia':'pmdec', 'pmdec_error_gaia':'e_pmdec'})
gaia_rrl_catalog['phi1'], gaia_rrl_catalog['phi2'] = phi12_rotmat(alpha=gaia_rrl_catalog['ra'],delta=gaia_rrl_catalog['dec'],R_phi12_radec=atlas_rotmat)
rrl = gaia_rrl_catalog[(gaia_rrl_catalog['phi1'] > -30) & (gaia_rrl_catalog['phi1'] < 30) & (gaia_rrl_catalog['phi2'] > -6) & (gaia_rrl_catalog['phi2'] < 6)]

rrl['p_pmra'] = pmra_gaussian(pmra=rrl['pmra'], phi1=rrl['phi1'], pmra_error=rrl['e_pmra'])
rrl['p_pmdec'] = pmdec_gaussian(pmdec=rrl['pmdec'], phi1=rrl['phi1'], pmdec_error=rrl['e_pmdec'])
rrl['p_spatial'] = phi2_gaussian(phi2=rrl['phi2'], phi1=rrl['phi1'])
bhb['p_pmra'] = pmra_gaussian(pmra=bhb['pmra'], phi1=bhb['phi1'], pmra_error=bhb['pmra_error'])
bhb['p_pmdec'] = pmdec_gaussian(pmdec=bhb['pmdec'], phi1=bhb['phi1'], pmdec_error=bhb['pmdec_error'])
bhb['p_spatial'] = phi2_gaussian(phi2=bhb['phi2'], phi1=bhb['phi1'])

bhb['g_r'] = bhb['g_mag'] - bhb['r_mag']
def bhb_absolute_mag(g_r): # BHB absolute magnitude relation from Barbosa et al 2022 https://arxiv.org/pdf/2210.02820
    return 0.398 - 0.392 * (g_r) + 2.729 * (g_r)**2 + 29.1128 * (g_r)**3 + 113.569 * (g_r)**4
bhb['M_g'] = bhb_absolute_mag(bhb['g_r'])
bhb['distance_modulus'] = bhb['g_mag'] - bhb['M_g']

rrl['distance_modulus'] = (rrl['phot_g_mean_mag_gaia']-rrl['MGmag_gaia_rrl_w_dist'])

rrl_subset = pd.DataFrame({
    'ra': rrl['ra'],
    'dec': rrl['dec'],
    'phi1': rrl['phi1'],
    'phi2': rrl['phi2'],
    'distance_modulus': rrl['distance_modulus'],
    'pmra': rrl['pmra'],
    'pmdec': rrl['pmdec'],
    'p_pmra': rrl['p_pmra'],
    'p_pmdec': rrl['p_pmdec'],
    'p_spatial': rrl['p_spatial'],
    'gaia_g': rrl['MGmag_gaia_rrl_w_dist'],
    'gaia_bp': rrl['phot_bp_mean_mag_gaia'],
    'gaia_rp': rrl['phot_rp_mean_mag_gaia'],
    'delve_g': np.nan,
    'delve_r': np.nan,
    'rrl?': True,
    'bhb?': False
})

bhb_subset = pd.DataFrame({
    'ra': bhb['ra'],
    'dec': bhb['dec'],
    'phi1': bhb['phi1'],
    'phi2': bhb['phi2'],
    'distance_modulus': bhb['distance_modulus'],
    'pmra': bhb['pmra'],
    'pmdec': bhb['pmdec'],
    'p_pmra': bhb['p_pmra'],
    'p_pmdec': bhb['p_pmdec'],
    'p_spatial': bhb['p_spatial'],
    'gaia_g': bhb['gaia_g'],
    'gaia_bp': bhb['gaia_bp'],
    'gaia_rp': bhb['gaia_rp'],
    'delve_g': bhb['g_mag'],
    'delve_r': bhb['r_mag'],
    'rrl?': False,
    'bhb?': True
})

bhb_rrl= pd.concat([rrl_subset, bhb_subset], ignore_index=True)
good_mask = (bhb_rrl['p_pmdec'] * bhb_rrl['p_pmra'] * bhb_rrl['p_spatial'] > 0.1) & (bhb_rrl['distance_modulus'] < 18.75) # 18.75 is empirically and visually derived
bhb_rrl[good_mask].to_csv('aau_bhb_rrl.csv', index=False)