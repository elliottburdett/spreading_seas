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
coefficients = np.polyfit(bhb_rrl['Phi1'], bhb_rrl['Distance_Modulus'], deg=2)
distance_fit = np.poly1d(coefficients)

pmdec_params = {'c1': -0.982, 'c2': -0.089, 'c3': 0.025} #Assumed from Andrew Li's S5 AAU Members
pmdec_fit = quad_f(phi1, pmdec_params['c1'], pmdec_params['c2'], pmdec_params['c3'])

pmra_params = {'c1': -0.164, 'c2': -0.349, 'c3': -0.057} #Assumed from Andrew Li's S5 AAU Members
pmra_fit = quad_f(phi1, pmra_params['c1'], pmra_params['c2'], pmra_params['c3'])

s5_aau_members = Table.read('aau_members_full.fits', format='fits')
coefficients = np.polyfit(s5_aau_members['phi1'], s5_aau_members['phi2'], deg=2) #Assumed from Andrew Li's S5 AAU Members
spatial_fit_function = np.poly1d(coefficients) #Coefficients = array([-0.00323187, -0.00125681,  0.80142462])





