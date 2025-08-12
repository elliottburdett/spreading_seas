from __future__ import division
import astropy
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
import ugali
import warnings
from matplotlib.path import Path
from ugali.analysis.isochrone import factory as isochrone_factory
from ugali.utils.shell import get_iso_dir
import hats
import lsdb
from matplotlib.colors import LogNorm
from astropy.table import QTable
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.polynomial import polynomial
from numpy.polynomial.polynomial import polyvander2d
import numpy.ma as ma
import glob
import astropy.io.fits as fitsio
from astropy import table
import matplotlib as mpl
from lsdb.core.search.box_search import box_filter
from scipy.interpolate import interp1d
import time
import logging
from dask.distributed import Client
import sys
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from numpy.polynomial.polynomial import polyval2d
import jax
import jax.numpy as jnp
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

code_path = "/astro/users/esb30/software/spreading_seas/code"
sys.path.append(code_path)
from delve_dr3_map_partitions_search import get_filter_splines, filter_data
from aau_mu_gradient import filter_with_mu_gradient
client = Client(n_workers=4, memory_limit="auto")
print('Created Client')
#Get the right columns: Want all the magnitude bands and don't forget PM errors when you crossmatch it with GAIA!!
delve_path = "/epyc/data/delve/dr3/delve_dr3_gold/delve_dr3_gold/"
delve_object = lsdb.read_hats(delve_path, columns=['COADD_OBJECT_ID', 'RA', 'DEC', 'PSF_MAG_APER_8_G_CORRECTED', 'PSF_MAG_APER_8_R_CORRECTED', 'PSF_MAG_APER_8_I_CORRECTED', 'PSF_MAG_APER_8_Z_CORRECTED', 'EXT_XGB', 'SOURCE'])
#delve_object
print('Conducting Query')
lazy_delve = delve_object.query("PSF_MAG_APER_8_G_CORRECTED > 16 and PSF_MAG_APER_8_G_CORRECTED < 24.5 and PSF_MAG_APER_8_G_CORRECTED - PSF_MAG_APER_8_R_CORRECTED > -0.3 and PSF_MAG_APER_8_G_CORRECTED - PSF_MAG_APER_8_R_CORRECTED < 1 and EXT_XGB == 0")
print('Computing Dataset')
delve = lazy_delve.compute()
print('Importing Rotations')
from rotation_matrix import phi12_rotmat
print('Adding AAU Phi1 and Phi2 Coordinates')
atlas_rotmat = [[0.83697865, 0.29481904, -0.4610298], [0.51616778, -0.70514011, 0.4861566], [0.18176238, 0.64487142, 0.74236331]]
phi1, phi2 = phi12_rotmat(alpha=delve['RA'].to_numpy(),delta=delve['DEC'].to_numpy(),R_phi12_radec=atlas_rotmat)
delve['phi1'] = phi1
delve['phi2'] = phi2
lazy_gaia = lsdb.read_hats('/epyc/data3/hats/catalogs/gaia_dr3/gaia/', columns=["ra", "dec", "pm", "pmra", "pmdec", "pmra_error", "pmdec_error"])
lazy_dxg = lazy_delve.crossmatch(lazy_gaia, n_neighbors=1, radius_arcsec=10)
print('Computing Gaia Crossmatch')
dxg = lazy_dxg.compute()
print('Adding Total Proper Motion Error')
dxg['pm_error_gaia'] = np.sqrt(np.square(dxg['pmra_error_gaia'].to_numpy()) + np.square(dxg['pmdec_error_gaia'].to_numpy()))
print('Importing Rotations for Proper Motions')
from rotation_matrix import pmphi12, pmphi12_reflex
print('Adding All Rotations to DXG')
phi1, phi2 = phi12_rotmat(alpha=dxg['RA_delve_dr3_gold'].to_numpy(),delta=dxg['DEC_delve_dr3_gold'].to_numpy(),R_phi12_radec=atlas_rotmat)
dxg['phi1'] = phi1
dxg['phi2'] = phi2
pmphi1, pmphi2 = pmphi12(alpha=dxg['RA_delve_dr3_gold'].to_numpy(),delta=dxg['DEC_delve_dr3_gold'].to_numpy(),mu_alpha_cos_delta=dxg['pmra_gaia'].to_numpy(),mu_delta=dxg['pmdec_gaia'].to_numpy(),R_phi12_radec=atlas_rotmat)
dxg['pmphi1'] = pmphi1
dxg['pmphi2'] = pmphi2
pmphi1r, pmphi2r = pmphi12_reflex(alpha=dxg['RA_delve_dr3_gold'].to_numpy(),delta=dxg['DEC_delve_dr3_gold'].to_numpy(),mu_alpha_cos_delta=dxg['pmra_gaia'].to_numpy(),mu_delta=dxg['pmdec_gaia'].to_numpy(),dist=(16.66 - 0.28*(dxg['phi1'] / 10) + 0.045*((dxg['phi1'] / 10) ** 2)),R_phi12_radec=atlas_rotmat)
dxg['pmphi1r'] = pmphi1r
dxg['pmphi2r'] = pmphi2r
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'serif'
print('Finding On- and Offstream Regions')
dxg_on = dxg[(dxg['phi1'] > -10) &(dxg['phi1'] < 5) &(dxg['phi2'] > 0) & (dxg['phi2'] < 1)] # Use the aau filter on 'dxg_on' for pmra vs pmdec plot -5 to 5
dxg_off = dxg[(dxg['phi1'] > -10) &(dxg['phi1'] < 5) &(dxg['phi2'] > 2) & (dxg['phi2'] < 3)]