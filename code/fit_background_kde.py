'''
Fits a KDE to the background (In the case of AAU, contamination from Sagittarius), and subtracts it.
Creates a movie of isochrone-based matched filters, assuming a flat distance gradient.
'''
__author__ = "Elliott Burdett"

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from matplotlib import gridspec
import matplotlib.colors as mcolors
from delve_dr3_map_partitions import get_filter_splines, filter_data
from aau_fit_functions import spatial_fit_function
from cmd_plots import plot_cmd_hist

bins_x = np.linspace(-30, 35, 200)
bins_y = np.linspace(-4, 4, 80)

mu_table = np.arange(16.5, 16.55, 0.1)
for mu in mu_table:
    age=12
    C=[0.02, 0.02]
    E=0.2
    spl_near, spl_far = get_filter_splines(age=age, mu=mu, z=0.0007, abs_mag_min=0.7, app_mag_max = 24, color_min=0, color_max=1.5, dmu=1, C=C, E=E, err=None)
    aau_sel = filter_data(color=(delve_on['PSF_MAG_APER_8_G_CORRECTED']-delve_on['PSF_MAG_APER_8_R_CORRECTED']), mag=delve_on['PSF_MAG_APER_8_G_CORRECTED'], spl_near=spl_near, spl_far=spl_far)
    
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(3, 2, width_ratios=[5, 3], height_ratios=[2, 1, 2], hspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[:, 1])
    
    H1, xedges1, yedges1 = np.histogram2d(delve_on[aau_sel]['phi1'], delve_on[aau_sel]['phi2'], bins=[bins_x, bins_y])
    smoothed_H1 = gaussian_filter(H1.T, sigma=0.8)
    phi1_grid, phi2_grid = np.meshgrid((bins_x[:-1] + bins_x[1:])/2, (bins_y[:-1] + bins_y[1:])/2)
    masked_H1 = np.ma.masked_where((smoothed_H1 > 15), smoothed_H1)

    bin_mask = smoothed_H1 > 15
    bin_indices_x = np.digitize(delve_on['phi1'], bins_x) - 1
    bin_indices_y = np.digitize(delve_on['phi2'], bins_y) - 1
    bin_indices_x = np.clip(bin_indices_x, 0, len(bins_x)-2)
    bin_indices_y = np.clip(bin_indices_y, 0, len(bins_y)-2)
    high_density_mask = bin_mask[bin_indices_y, bin_indices_x] #Mask for where counts are above N, where N is the same number above which I mask in the original plot.
    
    im1 = ax1.imshow(
        masked_H1,
        extent=[bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]],
        origin='lower',
        cmap='inferno',
        aspect='auto',
#        vmin=0,
#        vmax=10,
    )
    ax1.set_ylabel(r'$\phi_2$ [deg]', fontsize=10)
    ax1.set_title(rf'AAU Matched Filter, $\mu={mu:.2f}$', fontsize=12)
    phi1_fit = np.linspace(-30, 35, 100)
    phi2_fit = spatial_fit_function(phi1_fit)
    phi2_fit_outside = phi2_fit + 2
    ax1.plot(phi1_fit, phi2_fit, color='turquoise', linewidth=2, alpha=0.5, label=r'Stream track fit to $S^5$ AAU Members')
    ax1.fill_between(phi1_fit, phi2_fit-1, phi2_fit+1, color='turquoise', alpha=0.2)
    #ax1.plot(phi1_fit, phi2_fit_outside, color='green', linewidth=2, alpha=0.5, label=r'Stream track fit to $S^5$ AAU Members')
    ax1.fill_between(phi1_fit, phi2_fit_outside-1, phi2_fit_outside+1, color='green', alpha=0.6, label='±1° Background region')
    ax1.set_ylim(-4,4)
    ax1.axvline(x=10, color='red', linestyle='--', linewidth=1.5, label='DES Boundary')
    ax1.axvline(x=12, color='blue', linestyle='--', linewidth=1.5, label='Outside Broken Into Pieces (Li et al 2021)')
    ax1.legend(fontsize=4, loc='upper right')
    #cbar1 = fig.colorbar(im1, ax=ax1, label='Counts')
    
    phi2_deviation = np.abs(delve_on[aau_sel]['phi2'] - (spatial_fit_function(delve_on[aau_sel]['phi1']) + 2))
    outside_region = phi2_deviation < 1  # Points inside ±1° adjacent band
    
    phi1_selected = delve_on[aau_sel & outside_region & ~high_density_mask]['phi1']

    kde = gaussian_kde(phi1_selected) #Make KDE
    phi1_bin_centers = (bins_x[:-1] + bins_x[1:])/2
    kde_values = kde(phi1_bin_centers)
    #kde_values = (kde_values - kde_values.min()) / (kde_values.max() - kde_values.min())
    
    ax2.plot(phi1_bin_centers, kde_values, color='darkorange', lw=2)
    ax2.fill_between(phi1_bin_centers, 0, kde_values, color='darkorange', alpha=0.3)
    ax2.set_ylabel('Filter Pass KDE', fontsize=6)
    #ax2.set_ylim(0, 1)
    ax2.set_xlim(-30,35)
    ax2.axvline(x=10, color='red', linestyle='--', linewidth=1.5, label='DES Boundary')
    ax2.axvline(x=12, color='blue', linestyle='--', linewidth=1.5, label='Outside Li et al')
    
    kde_array = np.tile(kde_values, (len(bins_y)-1, 1))  # shape (79, 199)
    adjust = 65 * len(delve_on[aau_sel & ~high_density_mask])/(len(bins_x)*len(bins_y)) # KDE * phi1 domain * avg counts per bin (should)= background density -> KDE*65*14882/16000
    smoothed_H3 = gaussian_filter(H1.T - kde_array * adjust, sigma=0.8)
    masked_H3 = np.ma.masked_where((smoothed_H3 > 10), smoothed_H3)
    im3 = ax3.imshow(
        masked_H3,
        extent=[bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]],
        origin='lower',
        cmap='inferno',
        aspect='auto',
#        vmin=-3,
#        vmax=5,
    )
    ax3.set_ylabel(r'$\phi2$ (Corrected for Sgr Filter Passes)', fontsize=6)
    #cbar3 = fig.colorbar(im3, ax=ax3, label='Counts')

    ax3.axvline(x=10, color='red', linestyle='--', linewidth=1.5, label='DES Boundary')
    ax3.axvline(x=12, color='blue', linestyle='--', linewidth=1.5, label='Outside Li et al')
    ax3.set_xlabel(r'$\phi_1$ [deg]')
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    
    plot_cmd_hist(g=delve_on['PSF_MAG_APER_8_G_CORRECTED'],r=delve_on['PSF_MAG_APER_8_R_CORRECTED'],ax=ax4,sqrt_bin_count=100, cmap='inferno', title='CMD Selection: Apparent Magnitude',bright_bound=19,faint_bound=24,lower_color=0,higher_color=1)
    ax4.set_xlabel('g-r')
    ax4.set_ylabel(r'g')
    y_min, y_max = 18.67,23.67
    y_values = np.linspace(y_min, y_max, 100)
    x1_values = spl_near(y_values)
    x2_values = spl_far(y_values)
    ax4.plot(x1_values, y_values, 'r-', linewidth=2, color='blue', label=f'AAU Isochrone, age={age},mu={mu:.2f},C={C},E={E}')
    ax4.plot(x2_values, y_values, 'r-', linewidth=2, color='blue')
    ax4.set_ylim(24,19)
    ax4.legend(fontsize=6, loc='upper right')

    cbar_ax = fig.add_axes([0.07, 0.63, 0.01, 0.25])
    cbar1 = fig.colorbar(im1, cax=cbar_ax, label='')
    cbar_ax2 = fig.add_axes([0.07, 0.15, 0.01, 0.25])
    cbar2 = fig.colorbar(im3, cax=cbar_ax2, label='')
    
    plt.tight_layout()
    #plt.savefig(f'qwerty{mu}.png', bbox_inches='tight', dpi=300)