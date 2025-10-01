'''
Creates a dataframe of objects in DELVE DR3, flagged where it intersects with GAIA DR3.
    Columns:
        -Coadd Object ID
        -G mag (PSF)
        -R mag (PSF)
        -I mag (PSF)
        -Z mag (PSF)
        -RA
        -DEC
        -Phi1 (AAU)
        -Phi2 (AAU)
        -Gaia? (boolean)
        -PMRA (if GAIA)
        -PMDEC (if GAIA)
        -PMRA Error (if GAIA)
        -PMDEC Error (if GAIA)
        -PMPhi1 (if GAIA)
        -PMPhi2 (if GAIA)
        -PMRA Match Score (AAU, 0-1)
        -PMDEC Match Score (AAU, 0-1)
        -Spatial Match Score (AAU, 0-1)
    This should be repurposed to make for other stream data and AAU-Specific constants are flagged.
    Computing Large Datasets should be the most computationally expensive part of the script.
    In the future, LSBD should have the functionality to do a left crossmatch while leaving the datasets lazily loaded.
'''
import numpy as np
import pandas as pd
import hats
import lsdb
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
code_path = "/astro/users/esb30/software/spreading_seas/code"
sys.path.append(code_path)
from rotation_matrix import phi12_rotmat
#from gaussian_membership_fits import quad_f, pmra_gaussian, pmdec_gaussian, phi2_gaussian
atlas_rotmat = [[0.83697865, 0.29481904, -0.4610298], [0.51616778, -0.70514011, 0.4861566], [0.18176238, 0.64487142, 0.74236331]]
delve_path = "/epyc/data/delve/dr3/delve_dr3_gold/delve_dr3_gold/"
delve_object = lsdb.read_hats(delve_path, columns=['COADD_OBJECT_ID', 'RA', 'DEC', 'PSF_MAG_APER_8_G_CORRECTED', 'PSF_MAG_APER_8_R_CORRECTED', 'PSF_MAG_APER_8_I_CORRECTED', 'PSF_MAG_APER_8_Z_CORRECTED', 'EXT_XGB', 'SOURCE'])
lazy_delve = delve_object.query("PSF_MAG_APER_8_G_CORRECTED > 16 and PSF_MAG_APER_8_G_CORRECTED < 24.5 and PSF_MAG_APER_8_G_CORRECTED - PSF_MAG_APER_8_R_CORRECTED > -0.3 and PSF_MAG_APER_8_G_CORRECTED - PSF_MAG_APER_8_R_CORRECTED < 1 and EXT_XGB == 0")
lazy_gaia = lsdb.read_hats('/epyc/data3/hats/catalogs/gaia_dr3/gaia/', columns=["ra", "dec", "pm", "pmra", "pmdec", "pmra_error", "pmdec_error"])
print('Computing DELVE DR3')
delve = lazy_delve.compute()
print('Computing Gaia DR3')
gaia = lazy_gaia.compute()

delve['Phi1'], delve['Phi2'] = phi12_rotmat(alpha=delve['RA'].to_numpy(),delta=delve['DEC'].to_numpy(),R_phi12_radec=atlas_rotmat)
gaia['Phi1'], gaia['Phi2'] = phi12_rotmat(alpha=gaia['ra'].to_numpy(),delta=gaia['dec'].to_numpy(),R_phi12_radec=atlas_rotmat)
delve = delve[(delve['Phi1'] > -30) & (delve['Phi1'] < 30) & (delve['Phi2'] > -2) & (delve['Phi1'] < 4)]
gaia = gaia[(gaia['Phi1'] > -30) & (gaia['Phi1'] < 30) & (gaia['Phi2'] > -2) & (gaia['Phi1'] < 4)]

print('Crossmatching Data')

delve_coords = SkyCoord(ra=delve['RA'].to_numpy() * u.deg, dec=delve['DEC'].to_numpy() * u.deg)
gaia_coords = SkyCoord(ra=gaia['ra'].to_numpy() * u.deg, dec=gaia['dec'].to_numpy() * u.deg)
idx, angular_separation, _ = delve_coords.match_to_catalog_sky(gaia_coords)
match_radius = 1.0 * u.arcsec
match_mask = angular_separation < match_radius

dxg = pd.DataFrame({
    'COADD_OBJECT_ID': delve['COADD_OBJECT_ID'],
    'RA': delve['RA'],
    'DEC': delve['DEC'],
    'Phi1': delve['Phi1'],
    'Phi2': delve['Phi2'],
    'G_mag': delve['PSF_MAG_APER_8_G_CORRECTED'],
    'R_mag': delve['PSF_MAG_APER_8_R_CORRECTED'],
    'I_mag': delve['PSF_MAG_APER_8_I_CORRECTED'],
    'Z_mag': delve['PSF_MAG_APER_8_Z_CORRECTED'],
    'Gaia': match_mask,
    'PMRA': np.nan,
    'PMDEC': np.nan,
    'PMRA_Error': np.nan,
    'PMDEC_Error': np.nan,
})

gaia_matched = gaia.iloc[idx[match_mask]].reset_index(drop=True)

dxg.loc[match_mask, 'PMRA'] = gaia_matched['pmra'].values
dxg.loc[match_mask, 'PMDEC'] = gaia_matched['pmdec'].values
dxg.loc[match_mask, 'PMRA_Error'] = gaia_matched['pmra_error'].values
dxg.loc[match_mask, 'PMDEC_Error'] = gaia_matched['pmdec_error'].values

dxg['PMPhi1'], dxg['PMPhi2'] = pmphi12(alpha=dxg['RA'],delta=dxg['DEC'],mu_alpha_cos_delta=dxg['PMRA'],mu_delta=dxg['PMDEC'],R_phi12_radec=atlas_rotmat)

dxg['PMAR_Score'] = pmra_gaussian(pmra=dxg['PMRA'], phi1=dxg['Phi1'])
dxg['PMDEC_Score'] = pmdec_gaussian(pmdec=dxg['PMDEC'], phi1=dxg['Phi1'])
dxg['Spatial_Score'] = phi2_gaussian(phi2=dxg['Phi2'], phi1=dxg['Phi1'])

dxg.to_csv('dxg_aau.csv', index=False)