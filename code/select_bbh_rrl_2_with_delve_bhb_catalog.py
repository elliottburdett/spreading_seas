'''Selection of BHB and RRL stars in the AAU region
BHB Characterization:
    -Load DELVE BHB Catalog
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

delve_bhb_catalog = pd.read_csv('delve_bhb_catalog_1.0.csv')
delve_bhb_catalog['phi1'], delve_bhb_catalog['phi2'] = phi12_rotmat(alpha=delve_bhb_catalog['ra_gaia'],delta=delve_bhb_catalog['dec_gaia'],R_phi12_radec=atlas_rotmat)
bhb = delve_bhb_catalog[(delve_bhb_catalog['phi1'] > -30) & (delve_bhb_catalog['phi1'] < 30) & (delve_bhb_catalog['phi2'] > -2) & (delve_bhb_catalog['phi2'] < 4)]

gaia_rrl_catalog = pd.read_parquet('/epyc/data/gaia_rrl/li_xmatch.parq')
gaia_rrl_catalog = gaia_rrl_catalog.rename(columns={'Gaia_gaia_rrl_w_dist': 'source_id', 'ra_gaia_rrl_w_dist': 'ra', 'dec_gaia_rrl_w_dist':'dec', 'Dist_Phot_gaia_rrl_w_dist':'dist', 'e_Dist_Phot_gaia_rrl_w_dist':'e_dist', 'pmra_gaia':'pmra', 'pmra_error_gaia':'e_pmra', 'pmdec_gaia':'pmdec', 'pmdec_error_gaia':'e_pmdec'})
gaia_rrl_catalog['phi1'], gaia_rrl_catalog['phi2'] = phi12_rotmat(alpha=gaia_rrl_catalog['ra'],delta=gaia_rrl_catalog['dec'],R_phi12_radec=atlas_rotmat)
rrl = gaia_rrl_catalog[(gaia_rrl_catalog['phi1'] > -30) & (gaia_rrl_catalog['phi1'] < 30) & (gaia_rrl_catalog['phi2'] > -2) & (gaia_rrl_catalog['phi2'] < 4)]

rrl['pmra_score'] = pmra_gaussian(pmra=rrl['pmra'], phi1=rrl['phi1'])
rrl['pmdec_score'] = pmdec_gaussian(pmdec=rrl['pmdec'], phi1=rrl['phi1'])
rrl['spatial_score'] = phi2_gaussian(phi2=rrl['phi2'], phi1=rrl['phi1'])
bhb['pmra_score'] = pmra_gaussian(pmra=bhb['pmra'], phi1=bhb['phi1'])
bhb['pmdec_score'] = pmdec_gaussian(pmdec=bhb['pmdec'], phi1=bhb['phi1'])
bhb['spatial_score'] = phi2_gaussian(phi2=bhb['phi2'], phi1=bhb['phi1'])

rrl = rrl[(rrl['pmra_score'] * rrl['pmdec_score'] > 0.5) & rrl['spatial_score'] > 0.5]
bhb = bhb[(bhb['pmra_score'] * bhb['pmdec_score'] > 0.5) & bhb['spatial_score'] > 0.5]

rrl['distance_modulus'] = (rrl['phot_g_mean_mag_gaia']-rrl['MGmag_gaia_rrl_w_dist'])

rrl_subset = pd.DataFrame({
    'ra': rrl['ra'],
    'dec': rrl['dec'],
    'phi1': rrl['phi1'],
    'phi2': rrl['phi2'],
    'distance_modulus': rrl['distance_modulus'],
    'rrl?': True,
    'bhb?': False
})

bhb_subset = pd.DataFrame({
    'ra': bhb['ra_gaia'],
    'dec': bhb['dec_gaia'],
    'phi1': bhb['phi1'],
    'phi2': bhb['phi2'],
    'distance_modulus': bhb['distance_modulus'],
    'rrl?': False,
    'bhb?': True
})

bhb_rrl= pd.concat([rrl_subset, bhb_subset], ignore_index=True)
bhb_rrl.to_csv('aau_bhb_rrl.csv', index=False)