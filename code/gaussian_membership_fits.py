"""
Matched-Filter Stream Membership Scoring and Visualization

This script provides a set of tools for:
- Generating color-magnitude matched filters using isochrone models.
- Evaluating stellar membership likelihoods in a stellar stream using spatial and proper motion Gaussian models.
- Applying a Gaussian-based scoring filter in photometric space.
- Visualizing PDF models of proper motion and spatial distributions.
- Optionally adjusting Gaussian widths with observational uncertainties and spatial widening.

The script assumes the existence of fitted stream models in phi1, phi2, PMRA, and PMDEC, represented by quadratic functions (see aau_fit_functions.py).
The Gaussian widths (sigmas) are scaled by a global lsigspatial, which acts as a tunable hyperparameter
to better capture the distribution of less confidently identified stream members (stragglers).
"""

__author__ = "Elliott Burdett"

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from ugali.analysis.isochrone import factory as isochrone_factory

def rotation_matrix(stream):
    if stream == 'AAU':
        return [[0.83697865, 0.29481904, -0.4610298], [0.51616778, -0.70514011, 0.4861566], [0.18176238, 0.64487142, 0.74236331]] # Atlas rotmat
    else:
        return 'Unknown Stream. This function supports AAU'

def spatial_fit_function(phi1, stream):
    if stream == 'AAU':
        coefficients_left = [-0.01779665, -0.37163805, -1.16975274] # AAU
        coefficients_right = [-0.00488336,  0.00988029,  0.83630395] # AAU
        aau_spatial_fit_function_left = np.poly1d(coefficients_left) # AAU
        aau_spatial_fit_function_right = np.poly1d(coefficients_right) # AAU
        x = np.array(phi1)
        return np.where(x < -11.5, aau_spatial_fit_function_left(phi1), aau_spatial_fit_function_right(phi1))
    else:
        return 'Unknown Stream. This function supports AAU'

def quad_f(phi,c1,c2,c3):
    '''
    Quadratic function
    
    Parameters are all floats, returns a float
    '''
    x = phi/10
    return c1 + c2*x + c3*x**2

def get_filter_splines(age, mu, z, abs_mag_min=2.9, app_mag_max = 23.5, color_min=0, color_max=1, dmu=0.5, C=[0.05, 0.1], E=2., err=None):
    """
    Generate near and far spline boundaries for color-magnitude filtering.

    Parameters
    ----------
    age : float
        Stellar population age (Gyr).
    mu : float
        Distance modulus.
    z : float
        Metallicity (e.g., 0.0001 for Z=0.0001).
    abs_mag_min : float, optional
        Minimum absolute magnitude cut (default: 2.9).
    app_mag_max : float, optional
        Maximum apparent magnitude cut (default: 23.5).
    color_min : float, optional
        Minimum color value for clipping (default: 0).
    color_max : float, optional
        Maximum color value for clipping (default: 1).
    dmu : float, optional
        Half-width of distance modulus bin (default: 0.5).
    C : list, optional
        Color error offsets [near, far] (default: [0.05, 0.1]).
    E : float, optional
        Error scaling factor (default: 2.0).
    err: function to give magerr as a function of magnitude

    Returns
    -------
    spline_near : scipy.interpolate.interp1d
        Spline for the lower color boundary.
    spline_far : scipy.interpolate.interp1d
        Spline for the upper color boundary.
    """
    if err is None:
        err = lambda x: 0.0010908679647672335 + np.exp((x - 27.091072029215375) / 1.0904624484538419)
    iso = isochrone_factory('marigo2017', survey='DES', age=age, distance_modulus=mu, z=z)
    
    gsel = (iso.mag > abs_mag_min) & (iso.mag + mu < app_mag_max)
    color =iso.color[gsel]
    mag = iso.mag[gsel]
    # Spread in magnitude     
    mnear = mag + mu - dmu / 2.
    mfar = mag + mu + dmu / 2.
    color_far = np.clip(color + E * err(mfar) + C[1], color_min, color_max)
    color_near = np.clip(color - E * err(mnear) - C[0], color_min,color_max)
    spline_far = interp1d(mag + mu, color_far , bounds_error=False, fill_value=np.nan)
    spline_near = interp1d(mag + mu, color_near, bounds_error=False, fill_value=np.nan)
    return spline_near, spline_far

def filter_data_score(color, mag, spl_near, spl_far, sigma=None):
    """
    Compute a Gaussian-based membership score for each point based on distance from the isochrone.
    Returns score=0 for points with magnitudes outside the spline range.

    Parameters
    ----------
    color : array-like
        Observed color (e.g., g - r).
    mag : array-like
        Observed magnitude (e.g., g).
    spl_near : spline
        Lower boundary spline from get_filter_splines().
    spl_far : spline
        Upper boundary spline from get_filter_splines().
    sigma : function or float, optional
        Standard deviation of Gaussian. If function, should take `mag` as input.
        If None, uses a default constant small value.

    Returns
    -------
    score_normalized : array-like
        Gaussian score between 0 and 1 for each point.
    """
    color = np.asarray(color)
    mag = np.asarray(mag)
    
    #Determine valid mag range from spline
    mag_min = spl_near.x.min()
    mag_max = spl_near.x.max()
    in_bounds = (mag >= mag_min) & (mag <= mag_max)

    #Initialize all scores to 0
    score_normalized = np.zeros_like(mag, dtype=float)

    #Only compute scores for in-bound magnitudes
    if np.any(in_bounds):
        near_vals = spl_near(mag[in_bounds])
        far_vals = spl_far(mag[in_bounds])

        color_iso = 0.5 * (near_vals + far_vals)
        band_width = 0.5 * (far_vals - near_vals)

        if sigma is None:
            sigma_vals = band_width
        elif callable(sigma):
            sigma_vals = sigma(mag[in_bounds])
        else:
            sigma_vals = np.full_like(mag[in_bounds], sigma)

        #Compute normalized Gaussian score
        score = norm.pdf(color[in_bounds], loc=color_iso, scale=sigma_vals)
        norm_factor = norm.pdf(color_iso, loc=color_iso, scale=sigma_vals)
        score_normalized[in_bounds] = score / norm_factor

    return score_normalized

def pmdec_gaussian(pmdec, phi1, pmdec_error=None, stream='AAU', widen=None, normalize_peak=True):
    """
    Evaluate the Gaussian PDF for pmdec at given phi1 values, optionally incorporating per-star pmdec error.
    
    Parameters:
    - pmdec : float or np.ndarray
    - phi1 : float or np.ndarray (same shape)
    - pmdec_error : float or np.ndarray (same shape), optional
        Observational errors on pmdec. If None, only intrinsic scatter is used.
    - widen : float or None
        If float, specifies the phi1 value where sigma begins to widen, reaching 2x sigma at phi1=30.
    
    Returns:
    - Gaussian PDF values
    """
    if stream == 'AAU':
        pmdec_params = {'c1': -0.982, 'c2': -0.089, 'c3': 0.025} #Assumed from Andrew Li's S5 AAU Members
        lsigpmdec = -1.510 #Assumed from Andrew Li's S5 AAU Members
        #sigma_pmdec = (10 ** lsigpmdec)
        sigma_pmdec = 0
    else:
        raise ValueError(f"Stream '{stream}' not implemented in pmdec_gaussian()")

    mu = quad_f(phi1, pmdec_params['c1'], pmdec_params['c2'], pmdec_params['c3'])

    if pmdec_error is None:
        total_sigma = sigma_pmdec
    else:
        pmdec_error = np.asarray(pmdec_error)
        safe_error = np.where(
            np.isfinite(pmdec_error) & np.isreal(pmdec_error),
            pmdec_error,
            0.0
        )
        total_sigma = np.sqrt(sigma_pmdec**2 + safe_error**2)

    if widen is not None:
        phi1 = np.asarray(phi1)
        scale_factor = np.ones_like(phi1)

        mask = phi1 > widen
        scale = (phi1[mask] - widen) / (30.0 - widen)
        scale = np.clip(scale, 0, 1)
        scale_factor[mask] = 1.0 + scale

        total_sigma *= scale_factor

    exponent = -0.5 * ((pmdec - mu) / total_sigma) ** 2

    pdf = np.exp(exponent)

    if not normalize_peak:
        norm = 1.0 / (np.sqrt(2 * np.pi) * total_sigma)
        pdf *= norm

    return pdf

def pmra_gaussian(pmra, phi1, pmra_error=None, stream='AAU', widen=None, normalize_peak=True):
    """
    Evaluate the Gaussian PDF for pmra at given phi1 values, optionally incorporating per-star pmra error.
    
    Parameters:
    - pmra : float or np.ndarray
    - phi1 : float or np.ndarray (same shape)
    - pmra_error : float or np.ndarray (same shape), optional
        Observational errors on pmra. If None, only intrinsic scatter is used.
    - widen : float or None
        If float, specifies the phi1 value where sigma begins to widen, reaching 2x sigma at phi1=30.
    
    Returns:
    - Gaussian PDF values
    """
    if stream == 'AAU':
        pmra_params = {'c1': -0.164, 'c2': -0.349, 'c3': -0.057} #Assumed from Andrew Li's S5 AAU Members
        lsigpmra = -1.342 #Assumed from Andrew Li's S5 AAU Members
        # sigma_pmra = (10 ** lsigpmra)
        sigma_pmra = 0
    else:
        raise ValueError(f"Stream '{stream}' not implemented in pmra_gaussian()")

    mu = quad_f(phi1, pmra_params['c1'], pmra_params['c2'], pmra_params['c3'])

    if pmra_error is None:
        total_sigma = sigma_pmra
    else:
        pmra_error = np.asarray(pmra_error)
        safe_error = np.where(
            np.isfinite(pmra_error) & np.isreal(pmra_error),
            pmra_error,
            0.0
        )
        total_sigma = np.sqrt(sigma_pmra**2 + safe_error**2)

    if widen is not None:
        phi1 = np.asarray(phi1)
        scale_factor = np.ones_like(phi1)

        mask = phi1 > widen
        scale = (phi1[mask] - widen) / (30.0 - widen)
        scale = np.clip(scale, 0, 1)
        scale_factor[mask] = 1.0 + scale

        total_sigma *= scale_factor

    exponent = -0.5 * ((pmra - mu) / total_sigma) ** 2

    pdf = np.exp(exponent)

    if not normalize_peak:
        norm = 1.0 / (np.sqrt(2 * np.pi) * total_sigma)
        pdf *= norm

    return pdf

def pm_gaussian_2d(pmra, pmdec, phi1, pmra_error=None, pmdec_error=None, pmra_pmdec_corr=None,
                   stream='AAU', widen=None, normalize_peak=True):
    """
    Evaluate the 2D Gaussian PDF for (pmra, pmdec) at given phi1 values,
    incorporating covariance between pmra and pmdec.

    Parameters:
    - pmra : float or np.ndarray
    - pmdec : float or np.ndarray (same shape as pmra)
    - phi1 : float or np.ndarray (same shape)
    - pmra_error : float or np.ndarray, optional
        Observational errors on pmra.
    - pmdec_error : float or np.ndarray, optional
        Observational errors on pmdec.
    - pmra_pmdec_corr : float or np.ndarray, optional
        Correlation coefficient between pmra and pmdec errors (rho), in [-1, 1].
        If None, assumed to be 0 (no covariance).
    - stream : str
        Stream name for parameter lookup.
    - widen : float or None
        If float, specifies the phi1 value where sigma begins to widen,
        reaching 2x sigma at phi1=30. Only use if searching for a stream
        extension, as it is arbitrary and not physically motivated.
    - normalize_peak : bool
        If True, peak is normalized to 1 (useful for membership likelihood).
        If False, the PDF is properly normalized (integrates to 1).

    Returns:
    - 2D Gaussian PDF values (same shape as inputs)
    """
    if stream == 'AAU':
        pmra_params  = {'c1': -0.164, 'c2': -0.349, 'c3': -0.057}
        pmdec_params = {'c1': -0.982, 'c2': -0.089, 'c3':  0.025}
        sigma_pmra_int  = 0 # intrinsic scatter (set to 10**(-1.342) to re-enable)
        sigma_pmdec_int = 0 # intrinsic scatter (set to 10**(-1.510) to re-enable)
    else:
        raise ValueError(f"Stream '{stream}' not implemented in pm_gaussian_2d()")

    phi1  = np.asarray(phi1)
    pmra  = np.asarray(pmra)
    pmdec = np.asarray(pmdec)

    # stream tracks
    mu_pmra  = quad_f(phi1, pmra_params['c1'],  pmra_params['c2'],  pmra_params['c3'])
    mu_pmdec = quad_f(phi1, pmdec_params['c1'], pmdec_params['c2'], pmdec_params['c3'])

    # total sigmas with intrinsic + observational added in quadrature. Apply to pmra and pmdec
    def _total_sigma(sigma_int, obs_error):
        if obs_error is None:
            return np.full_like(phi1, float(sigma_int), dtype=float)
        obs = np.asarray(obs_error, dtype=float)
        safe = np.where(np.isfinite(obs) & np.isreal(obs), obs, 0.0)
        return np.sqrt(sigma_int**2 + safe**2)

    sigma_1 = _total_sigma(sigma_pmra_int,  pmra_error) # total sigma for pmra
    sigma_2 = _total_sigma(sigma_pmdec_int, pmdec_error) # total sigma for pmdec

    # optional widening along phi1, implement only if searching for stream extension 
    if widen is not None:
        scale_factor = np.ones_like(phi1, dtype=float)
        mask = phi1 > widen
        scale = np.clip((phi1[mask] - widen) / (30.0 - widen), 0, 1)
        scale_factor[mask] = 1.0 + scale
        sigma_1 = sigma_1 * scale_factor
        sigma_2 = sigma_2 * scale_factor

    # correlation / covariance
    if pmra_pmdec_corr is None:
        rho = np.zeros_like(phi1, dtype=float)
    else:
        rho = np.clip(np.asarray(pmra_pmdec_corr, dtype=float), -1 + 1e-9, 1 - 1e-9)

    # residuals (z numerator)
    d1 = pmra  - mu_pmra # delta pmra
    d2 = pmdec - mu_pmdec # delta pmdec

    # 2D Gaussian exponent using the inverse of the 2x2 covariance matrix
    # z = 1/(1-rho^2) * [(d1/s1)^2 - 2*rho*(d1/s1)*(d2/s2) + (d2/s2)^2] for a bivariate Gaussian
    one_minus_rho2 = 1.0 - rho**2
    z = (1.0 / one_minus_rho2) * (
        (d1 / sigma_1)**2
        - 2.0 * rho * (d1 / sigma_1) * (d2 / sigma_2)
        + (d2 / sigma_2)**2
    )
    pdf = np.exp(-0.5 * z)

    if not normalize_peak:
        # Full normalization: 1 / (2*pi*s1*s2*sqrt(1-rho^2))
        norm = 1.0 / (2.0 * np.pi * sigma_1 * sigma_2 * np.sqrt(one_minus_rho2))
        pdf = pdf * norm

    return pdf
    
def phi2_gaussian(phi2, phi1, widen=None, normalize_peak=True,  stream='AAU'):
    if stream == 'AAU':
        # aau_members['phi2_model'] = spatial_fit_function(aau_members['phi1'])
        # aau_members['phi2_residual'] = aau_members['phi2'] - aau_members['phi2_model']
        # sigma_phi2 = aau_members['phi2_residual'].std()
        # print(f"Average sigma phi2: {sigma_phi2:.4f} degrees")
        sigma_spatial = 0.4352 #From Andrew Li's members
    else:
        raise ValueError(f"Stream '{stream}' not implemented in phi2_gaussian()")

    total_sigma = sigma_spatial
    if widen is not None:
        phi1 = np.asarray(phi1)
        scale_factor = np.ones_like(phi1)

        mask = phi1 > widen
        scale = (phi1[mask] - widen) / (30.0 - widen)
        scale = np.clip(scale, 0, 1)
        scale_factor[mask] = 1.0 + scale

        total_sigma *= scale_factor
    exponent = -0.5 * ((phi2 - spatial_fit_function(phi1, stream)) / total_sigma) ** 2

    pdf = np.exp(exponent)

    if not normalize_peak:
        norm = 1.0 / (np.sqrt(2 * np.pi) * total_sigma)
        pdf *= norm

    return pdf

def p_photometric(mag_g, mag_r, g_err, r_err, age, z, mu,
                  abs_mag_min=2.9, app_mag_max=23.5, sigma_i=0.1):
    """
    Compute an isochrone-based photometric membership weight for stars.

    This function evaluates how closely each star's observed (g-r) color matches
    the expected color of a theoretical isochrone at the star's inferred absolute
    g-band magnitude, assuming a stellar population of given age, metallicity,
    and distance modulus.

    The returned value is a Gaussian-like weight:

        p = exp[-0.5 * ((color_obs - color_iso) / sigma_color)^2]

    where:
        - color_obs is the observed (g-r) color,
        - color_iso is the isochrone-predicted color at the same absolute magnitude,
        - sigma_color is the propagated color uncertainty.

    A value near 1 indicates strong agreement with the isochrone, while values
    near 0 indicate poor agreement.

    Parameters
    ----------
    mag_g : array-like
        Observed apparent g-band magnitudes.
    mag_r : array-like
        Observed apparent r-band magnitudes.
    g_err : array-like
        Uncertainties on g-band magnitudes.
    r_err : array-like
        Uncertainties on r-band magnitudes.
    age : float
        Stellar population age passed to the isochrone model.
    z : float
        Metallicity passed to the isochrone model.
    mu : float or array-like
        Distance modulus(es) used to convert apparent magnitudes to absolute
        magnitudes. If scalar, applied to all stars.
    abs_mag_min : float, optional
        Minimum absolute magnitude cutoff applied to the isochrone. Only
        isochrone points with M_g > abs_mag_min are used. Default is 2.9.
    app_mag_max : float, optional
        Maximum allowed apparent g-band magnitude. Stars fainter than this are
        assigned zero weight. Default is 23.5.
    sigma_i: float
        Intrinsic dispersion of isochrone in the color dimension.

    Returns
    -------
    p : ndarray
        Array of photometric weights between 0 and 1 representing consistency
        with the specified isochrone.

    Notes
    -----
    - This is not a normalized probability density.
    - The likelihood is evaluated only in color-space, not full CMD-space.
    - Stars outside interpolation bounds or failing quality cuts receive p = 0.
    """
    mag_g = np.asarray(mag_g, dtype=float)
    mag_r = np.asarray(mag_r, dtype=float)
    mu    = np.asarray(mu,    dtype=float)
    if mu.ndim == 0:
        mu = np.full_like(mag_g, float(mu))

    color = mag_g - mag_r

    # Pick a reference mu safely within ugali's allowed bounds [10, 30]
    mu_ref = float(np.clip(np.nanmean(mu), 10.0, 30.0))

    iso = isochrone_factory('marigo2017', survey='DES',
                            age=age, distance_modulus=mu_ref, z=z)

    gsel = iso.mag > abs_mag_min
    iso_abs_mag = iso.mag[gsel]
    iso_color   = iso.color[gsel]

    c_iso_spline = interp1d(iso_abs_mag, iso_color,
                            bounds_error=False, fill_value=np.nan)

    abs_mag_g = mag_g - mu

    p = np.zeros_like(color)

    c_iso   = c_iso_spline(abs_mag_g)
    sigma_c = np.sqrt(g_err**2 + r_err**2 + sigma_i**2)

    in_bounds = (
        (mag_g < app_mag_max) &
        np.isfinite(c_iso) &
        np.isfinite(sigma_c) &
        (sigma_c > 0)
    )
    p[in_bounds] = np.exp(
        -0.5 * ((color[in_bounds] - c_iso[in_bounds]) / sigma_c[in_bounds])**2
    )

    return p