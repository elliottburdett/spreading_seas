from scipy.stats import norm
import numpy as np
from scipy.interpolate import interp1d

sigma_scale_factor = 10

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

def membership_score(pmra, pmdec, phi1):
    """
    Compute a membership score (0 to ~max_pdf^2) based on pmra and pmdec Gaussians at given phi1.
    Returns:
    - score: float or np.ndarray (same shape as params)
    """
    pmra_params = {'c1': -0.164, 'c2': -0.349, 'c3': -0.057}
    lsigpmra = -1.342
    sigma_pmra = (10 ** lsigpmra) * sigma_scale_factor
    pmdec_params = {'c1': -0.982, 'c2': -0.089, 'c3': 0.025}
    lsigpmdec = -1.510
    sigma_pmdec = (10 ** lsigpmdec) * sigma_scale_factor
    def quad_f(phi, c1, c2, c3):
        x = phi / 10.0
        return c1 + c2 * x + c3 * x**2

    mu_pmra = quad_f(phi1, pmra_params['c1'], pmra_params['c2'], pmra_params['c3'])
    norm_pmra = 1.0 / (np.sqrt(2 * np.pi) * sigma_pmra)
    exp_pmra = -0.5 * ((pmra - mu_pmra) / sigma_pmra) ** 2
    g_pmra = norm_pmra * np.exp(exp_pmra)

    mu_pmdec = quad_f(phi1, pmdec_params['c1'], pmdec_params['c2'], pmdec_params['c3'])
    norm_pmdec = 1.0 / (np.sqrt(2 * np.pi) * sigma_pmdec)
    exp_pmdec = -0.5 * ((pmdec - mu_pmdec) / sigma_pmdec) ** 2
    g_pmdec = norm_pmdec * np.exp(exp_pmdec)

    max_score = (1 / (np.sqrt(2 * np.pi) * sigma_pmra)) * (1 / (np.sqrt(2 * np.pi) * sigma_pmdec))
    score = (g_pmra * g_pmdec) / max_score
    return score

pmdec_params = {'c1': -0.982, 'c2': -0.089, 'c3': 0.025}
lsigpmdec = -1.510
sigma_pmdec = (10 ** lsigpmdec) * sigma_scale_factor

def pmdec_gaussian(pmdec, phi1, pmdec_error=None, widen=None):
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

    norm = 1.0 / (np.sqrt(2 * np.pi) * total_sigma)
    exponent = -0.5 * ((pmdec - mu) / total_sigma) ** 2

    return norm * np.exp(exponent)
    
phi1_vals = np.linspace(-30, 30, 300)
pmdec_vals = np.linspace(-3, 3, 300)

PHI1, PMDEC = np.meshgrid(phi1_vals, pmdec_vals)
Z = pmdec_gaussian(PMDEC, PHI1)

plt.figure(figsize=(8, 4))
plt.contourf(PHI1, PMDEC, Z, levels=50, cmap='plasma')
plt.colorbar(label='PDF')
plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\mu_\delta$')
plt.title('Gaussian PDF for PMDEC')
plt.tight_layout()
plt.show()

pmra_params = {'c1': -0.164, 'c2': -0.349, 'c3': -0.057}
lsigpmra = -1.342
sigma_pmra = (10 ** lsigpmra) * sigma_scale_factor

def pmra_gaussian(pmra, phi1, pmra_error=None, widen=None):
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

    norm = 1.0 / (np.sqrt(2 * np.pi) * total_sigma)
    exponent = -0.5 * ((pmra - mu) / total_sigma) ** 2

    return norm * np.exp(exponent)

phi1_vals = np.linspace(-30, 30, 300)
pmra_vals = np.linspace(-3, 3, 300)

PHI1, PMRA = np.meshgrid(phi1_vals, pmra_vals)
Z = pmra_gaussian(PMRA, PHI1)

plt.figure(figsize=(8, 4))
plt.contourf(PHI1, PMRA, Z, levels=50, cmap='viridis')
plt.colorbar(label='PDF')
plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\mu_\alpha cos\delta$')
plt.title('Gaussian PDE For PMRA')
plt.tight_layout()
plt.show()