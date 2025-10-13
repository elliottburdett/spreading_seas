'''
Conducts operations on large healpix cubes.
'''

__author__ = "Elliott Burdett"

import pandas as pd
import healpy as hp
import numpy as np

def scale_hpx_cube(cube_to_modify, other_cube):
    '''Scales a healpix cube based on the counts per healpixel of another healpix cube'''
    scaled = cube_to_modify.copy()
    for col in scaled.columns:
        if col != 'pix512':
            low, high = other_cube[col].quantile([0, 0.95])
            trimmed_mean1 = other_cube[col][(other_cube[col] >= low) & (other_cube[col] <= high)].mean()
            
            low, high = cube_to_modify[col].quantile([0, 0.95])
            trimmed_mean2 = cube_to_modify[col][(cube_to_modify[col] >= low) & (cube_to_modify[col] <= high)].mean()
            
            scaled[col] = scaled[col] + trimmed_mean1 - trimmed_mean2

    return scaled

def combine_hpx_cubes(df1, df2, fill=True):
    '''Returns a combined healpix cube object.
    if fill, creates zero rows for pixel ids with no data.'''
    
    combined = pd.concat([df1.set_index('pix512'), df2.set_index('pix512')])
    df = combined.groupby('pix512').sum().reset_index()

    if fill:
        all_pix_values = pd.Series(np.arange(0, hp.nside2npix(512)))
        missing_pix = all_pix_values[~all_pix_values.isin(df['pix512'])]
        missing_df = pd.DataFrame({'pix512': missing_pix})
        
        for col in df.columns:
            if col != 'pix512':
                missing_df[col] = hp.UNSEEN
        
        df_complete = pd.concat([df, missing_df], ignore_index=True)
        df = df_complete.sort_values('pix512').reset_index(drop=True)

    return df
    
def fill_hpx_cube(df):
    '''Creates zero rows for pixel ids with no data.'''
    
    all_pix_values = pd.Series(np.arange(0, hp.nside2npix(512)))
    missing_pix = all_pix_values[~all_pix_values.isin(df['pix512'])]
    missing_df = pd.DataFrame({'pix512': missing_pix})
    
    for col in df.columns:
        if col != 'pix512':
            missing_df[col] = hp.UNSEEN
    
    df_complete = pd.concat([df, missing_df], ignore_index=True)
    df = df_complete.sort_values('pix512').reset_index(drop=True)

    return df