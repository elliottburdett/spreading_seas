'''
Creates a column that is True where are members are s5 spectroscopic members and False otherwise.
'''
__author__ = "Elliott Burdett"

import numpy as np
import pandas as pd
import astropy
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u

dxg_aau = pd.read_parquet('/astro/users/esb30/software/spreading_seas/notebooks/dxg_aau.parquet')
s5_aau_members = Table.read('aau_members_full.fits', format='fits')

ra_dxg = dxg_aau['ra'].to_numpy().astype(float)
dec_dxg = dxg_aau['dec'].to_numpy().astype(float)
ra_s5 = np.array(s5_aau_members['ra'], dtype=float)
dec_s5 = np.array(s5_aau_members['dec'], dtype=float)

coords_dxg = SkyCoord(ra=ra_dxg * u.deg, dec=dec_dxg * u.deg)
coords_s5 = SkyCoord(ra=ra_s5 * u.deg, dec=dec_s5 * u.deg)

idx, separation, _ = coords_dxg.match_to_catalog_sky(coords_s5)

dxg_aau['is_s5_member'] = separation < 0.5 * u.arcsec