"""
I have hardcoded my 2nd-order polynomial fits to AAU's parameters as a function of Phi1 in each other file.
They are flagged, so new ones may be created with new data.
This is how they were created.
"""

__author__ = "Elliott Burdett"

import pandas as pd
import numpy as np
from astropy.table import Table

def quad_f(phi,c1,c2,c3):
    '''
    Quadratic function
    
    Parameters are all floats, returns a float
    '''
    x = phi/10
    return c1 + c2*x + c3*x**2

bhb_rrl = pd.read_csv("aau_bhb_rrl.csv")  #Created with select_bhb_rrl.py
coefficients = np.polyfit(bhb_rrl['phi1'], bhb_rrl['distance_modulus'], deg=2)
distance_fit = np.poly1d(coefficients)

pmdec_params = {'c1': -0.982, 'c2': -0.089, 'c3': 0.025} #Assumed from Andrew Li's S5 AAU Members
def pmdec_fit(phi1):
    return quad_f(phi1, pmdec_params['c1'], pmdec_params['c2'], pmdec_params['c3'])

pmra_params = {'c1': -0.164, 'c2': -0.349, 'c3': -0.057} #Assumed from Andrew Li's S5 AAU Members
def pmra_fit(phi1): 
    return quad_f(phi1, pmra_params['c1'], pmra_params['c2'], pmra_params['c3'])

s5_aau_members = Table.read('aau_members_full.fits', format='fits')
coefficients_left = np.polyfit(s5_aau_members[s5_aau_members['phi1'] < -11.5]['phi1'], s5_aau_members[s5_aau_members['phi1'] < -11.5]['phi2'], deg=2)
spatial_fit_function_left = np.poly1d(coefficients_left)
coefficients_right = np.polyfit(s5_aau_members[s5_aau_members['phi1'] > -11.5]['phi1'], s5_aau_members[s5_aau_members['phi1'] > -11.5]['phi2'], deg=2)
spatial_fit_function_right = np.poly1d(coefficients_right)
def spatial_fit_function(phi1):
    x = np.array(phi1)
    return np.where(x < -11.5, spatial_fit_function_left(phi1), spatial_fit_function_right(phi1))