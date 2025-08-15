fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=[10, 8], 
                               gridspec_kw={'height_ratios': [2, 1, 2, 1, 1]},
                               sharex=True)

# Top panel: Scatter plot
ax1.scatter(sgr_aau_region['phi1'], sgr_aau_region['phi2'], s=3, alpha=0.5, c='Orange')
ax1.set_xlim(-30, 30)
ax1.set_ylabel(r'$\phi_2$ [deg]')
ax1.set_title('Confirmed Sgr Members in the AAU Box')

# Bottom panel: Prominence plot
phi1_grid, prominence, kde = sgr_prominence(sgr_aau_region['phi1'])
ax2.fill_between(phi1_grid, prominence, color='C0', alpha=0.3)
ax2.plot(phi1_grid, prominence, color='C0')
ax2.set_ylim(0, 1)
ax2.set_ylabel('Sgr Prominence')

spl_near, spl_far = get_filter_splines(age=12, mu=atlas_bhb_mu_fit(0), z=0.0007, abs_mag_min=0.7, app_mag_max = 24, color_min=0, color_max=1.5, dmu=1, C=[0.02, 0.02], E=0.2, err=None)
sel = filter_with_bhb_gradient(color=(delve_aau_region['PSF_MAG_APER_8_G_CORRECTED']-delve_aau_region['PSF_MAG_APER_8_R_CORRECTED']), mag=delve_aau_region['PSF_MAG_APER_8_G_CORRECTED'], phi1=delve_aau_region['phi1'], spl_near=spl_near, spl_far=spl_far)
H3, xedges3, yedges3 = np.histogram2d(delve_aau_region[sel]['phi1'], delve_aau_region[sel]['phi2'], bins=[bins_x, bins_y])
smoothed_H3 = gaussian_filter(H3.T, sigma=0.8)
im3 = ax3.imshow(
    smoothed_H3,
    extent=[bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]],
    origin='lower',
    cmap='inferno',
    aspect='auto',
#    vmin=0,
    vmax=6,
)

ax3.axvline(x=10, color='red', linestyle='--', linewidth=1.5, 
           label='DES Boundary')
ax3.set_ylabel(r'$\phi_2$ (deg)', fontsize=12)
#ax3.set_title('AAU & ~Sgr Matched Filter Based on Distance Gradients Alone', fontsize=14)
ax3.plot(phi1_fit, phi2_fit, color='turquoise', linewidth=2, alpha=0.2, label='Stream Track')
ax3.axvline(x=12, color='blue', linestyle='--', linewidth=1.5, 
           label='Outside Li et al')
ax3.legend(loc='upper right', framealpha=0.7)
ax3.set_ylim(-2,4)
ax3.set_xlim(-30,30)

counts, bin_edges = np.histogram(delve_aau_region[sel]['phi1'], bins=np.linspace(-30, 30, 61))
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

ax4.fill_between(bin_centers, counts, color='purple', alpha=0.3)
ax4.plot(bin_centers, counts, color='purple')
ax4.set_ylabel('Filter Passes')
ax4.set_ylim(0, counts.max()*1.1)  # Add 10% headroom

prominence_interp = np.interp(bin_centers, phi1_grid, prominence)

# Estimate non-Sgr component: when prominence=1 (pure Sgr), background=0
# when prominence=0 (no Sgr), background=full counts
background_estimate = counts * (1 - prominence_interp * 0.85)

ax5.fill_between(bin_centers, background_estimate, 
                color='green', alpha=0.3, label='Estimated non-Sgr')
ax5.plot(bin_centers, background_estimate, color='green')
ax5.plot(bin_centers, counts, color='purple', alpha=0.7, label='Total filtered')
ax5.set_ylabel('Estimated Sgr')
ax5.set_xlabel(r'$\phi_1$ (deg)')
ax5.set_ylim(0, counts.max()*1.1)
ax5.legend()

plt.tight_layout()
plt.show()