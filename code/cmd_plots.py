def plot_cmd_hist(g,r,ax=ax,sqrt_bin_count=100, cmap='inferno', title='CMD'):

    bins_x = np.linspace(0, 1, sqrt_bin_count)
    bins_y = np.linspace(16, 24.5, sqrt_bin_count)


    H1, xedges1, yedges1 = np.histogram2d(
        g-r, g, bins=[bins_x, bins_y])

    im1 = ax.imshow(
        H1.T,  # Transpose to match axis order
        extent=[bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]],
        origin='lower',
        cmap=cmap,
        aspect='auto')

    cbar = fig.colorbar(im1, ax=ax, label='Counts')

    ax.invert_yaxis()
    ax.set_xlabel('g-r', fontsize=12)
    ax.set_ylabel('g', fontsize=12)
    ax.set_title(title, fontsize=14)

def plot_cmd_scatter(g,r,ax,color='turquoise',s=20,marker='*',title='CMD'):

    ax.invert_yaxis()
    image = ax.scatter(g-r, g, c=color, s=s, marker=marker)
    ax.invert_yaxis()
    ax.set_xlabel('g-r', fontsize=12)
    ax.set_ylabel('g', fontsize=12)
    ax.set_title(title, fontsize=14)

def plot_hess(g_on,r_on,g_off,r_off,ax=ax,sqrt_bin_count=100, cmap='inferno', title='HESS'):

    bins_x = np.linspace(0, 1, sqrt_bin_count)
    bins_y = np.linspace(16, 24.5, sqrt_bin_count)


    H1, xedges1, yedges1 = np.histogram2d(
        g_on-r_on, g_on, bins=[bins_x, bins_y])

    H2, xedges2, yedges2 = np.histogram2d(
        g_off-r_off, g_off, bins=[bins_x, bins_y])

    im1 = ax.imshow(
        H1.T - H2.T,  # Transpose to match axis order
        extent=[bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]],
        origin='lower',
        cmap=cmap,
        aspect='auto')

    cbar = fig.colorbar(im1, ax=ax, label='Counts')

    ax.invert_yaxis()
    ax.set_xlabel('g-r', fontsize=12)
    ax.set_ylabel('g', fontsize=12)
    ax.set_title(title, fontsize=14)