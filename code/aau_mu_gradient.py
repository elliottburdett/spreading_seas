def filter_with_mu_gradient(color, mag, phi1, spl_near, spl_far):
    """
    Filter data points based on spline boundaries and phi1

    Necessary: Input mu=16.66 to get_filter_splines. This is the central mu value of the AAU stream.
    """
    x = phi1 / 10
    mu_adjust = -0.28*x + 0.045*(x ** 2)
    
    near_vals = spl_near(mag - mu_adjust)
    far_vals =  spl_far(mag - mu_adjust)
    sel = (color > near_vals) & (color < far_vals)
    return sel