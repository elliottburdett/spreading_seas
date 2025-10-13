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
    -RA
    -Dec
    -Phi1
    -Phi2
    -Distance
    -RRL? (boolean)
    -BHB? (boolean)
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
lazy_gaia = lsdb.read_hats('/epyc/data3/hats/catalogs/gaia_dr3/gaia/', columns=["ra", "dec", "pm", "pmra", "pmdec", "pmra_error", "pmdec_error"])
lazy_dxg = lazy_delve.crossmatch(lazy_gaia, n_neighbors=1, radius_arcsec=10)
print('Computing Dataset')
dxg = lazy_dxg.compute()

dxg['phi1'], dxg['phi2'] = phi12_rotmat(alpha=dxg['RA_delve_dr3_gold'].to_numpy(),delta=dxg['DEC_delve_dr3_gold'].to_numpy(),R_phi12_radec=atlas_rotmat)
dxg_on = dxg[(dxg['phi1'] > -30) &(dxg['phi2'] < 4) &(dxg['phi2'] > -2) & (dxg['phi1'] < 30)] # Use the aau filter on 'dxg_on' for pmra vs pmdec plot -5 to 5
dxg_on['g_abs'] = dxg_on['PSF_MAG_APER_8_G_CORRECTED_delve_dr3_gold'] - 16.66
bhb = dxg_on[
    (dxg_on['PSF_MAG_APER_8_G_CORRECTED_delve_dr3_gold'] - dxg_on['PSF_MAG_APER_8_R_CORRECTED_delve_dr3_gold'] > -0.3) & 
    (dxg_on['PSF_MAG_APER_8_G_CORRECTED_delve_dr3_gold'] - dxg_on['PSF_MAG_APER_8_R_CORRECTED_delve_dr3_gold'] < 0) & 
    (dxg_on['g_abs'] > -1) & 
    (dxg_on['g_abs'] < 2) #BHB gmags usually range from 0 to 1, and I am allowing a distance gradient of +/- 1 on each side
]

gaia_rrl_catalog = pd.read_parquet('/epyc/data/gaia_rrl/li_xmatch.parq')
gaia_rrl_catalog = gaia_rrl_catalog.rename(columns={'Gaia_gaia_rrl_w_dist': 'source_id', 'ra_gaia_rrl_w_dist': 'ra', 'dec_gaia_rrl_w_dist':'dec', 'Dist_Phot_gaia_rrl_w_dist':'dist', 'e_Dist_Phot_gaia_rrl_w_dist':'e_dist', 'pmra_gaia':'pmra', 'pmra_error_gaia':'e_pmra', 'pmdec_gaia':'pmdec', 'pmdec_error_gaia':'e_pmdec'})
gaia_rrl_catalog['phi1'], gaia_rrl_catalog['phi2'] = phi12_rotmat(alpha=gaia_rrl_catalog['ra'],delta=gaia_rrl_catalog['dec'],R_phi12_radec=atlas_rotmat)
rrl = gaia_rrl_catalog[(gaia_rrl_catalog['phi1'] > -30) & (gaia_rrl_catalog['phi1'] < 30) & (gaia_rrl_catalog['phi2'] > -2) & (gaia_rrl_catalog['phi2'] < 4)]

rrl['pmra_score'] = pmra_gaussian(pmra=rrl['pmra'], phi1=rrl['phi1'])
rrl['pmdec_score'] = pmdec_gaussian(pmdec=rrl['pmdec'], phi1=rrl['phi1'])
rrl['spatial_score'] = phi2_gaussian(phi2=rrl['phi2'], phi1=rrl['phi1'])
bhb['pmra_score'] = pmra_gaussian(pmra=bhb['pmra_gaia'], phi1=bhb['phi1'])
bhb['pmdec_score'] = pmdec_gaussian(pmdec=bhb['pmdec_gaia'], phi1=bhb['phi1'])
bhb['spatial_score'] = phi2_gaussian(phi2=bhb['phi2'], phi1=bhb['phi1'])

rrl = rrl[(rrl['pmra_score'] * rrl['pmdec_score'] > 0.5) & rrl['spatial_score'] > 0.5]
bhb = bhb[(bhb['pmra_score'] * bhb['pmdec_score'] > 0.5) & bhb['spatial_score'] > 0.5]

bhb['g_r'] = bhb['PSF_MAG_APER_8_G_CORRECTED_delve_dr3_gold'] - bhb['PSF_MAG_APER_8_R_CORRECTED_delve_dr3_gold']
# BHB absolute magnitude relation from Barbosa et al 2022 https://arxiv.org/pdf/2210.02820
def bhb_absolute_mag(g_r):
    #return 0.178 / (0.537 + g_r)
    return 0.398 - 0.392 * (g_r) + 2.729 * (g_r)**2 + 29.1128 * (g_r)**3 + 113.569 * (g_r)**4
bhb['M_g'] = bhb_absolute_mag(bhb['g_r'])
bhb['distance_modulus'] = bhb['PSF_MAG_APER_8_G_CORRECTED_delve_dr3_gold'] - bhb['M_g']

rrl['distance_modulus'] = (rrl['phot_g_mean_mag_gaia']-rrl['MGmag_gaia_rrl_w_dist'])

rrl_subset = pd.DataFrame({
    'RA': rrl['ra'],
    'Dec': rrl['dec'],
    'Phi1': rrl['phi1'],
    'Phi2': rrl['phi2'],
    'Distance_Modulus': rrl['distance_modulus'],
    'RRL?': True,
    'BHB?': False
})

bhb_subset = pd.DataFrame({
    'RA': bhb['RA_delve_dr3_gold'],
    'Dec': bhb['DEC_delve_dr3_gold'],
    'Phi1': bhb['phi1'],
    'Phi2': bhb['phi2'],
    'Distance_Modulus': bhb['distance_modulus'],
    'RRL?': False,
    'BHB?': True
})

bhb_rrl= pd.concat([rrl_subset, bhb_subset], ignore_index=True)
bhb_rrl.to_csv('aau_bhb_rrl.csv', index=False)