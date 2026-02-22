def get_coupon(rm, freq=1, years=30):
    """Compute the coupon rate for a par bond.

    Given an annually-compounded yield ``rm``, returns the per-period coupon
    payment ``q_star`` such that the bond prices at par.

    Parameters
    ----------
    rm : float
        Annually-compounded bond yield (e.g. 0.05 for 5 %).
    freq : int, optional
        Number of coupon payments per year. Default is 1 (annual).
    years : int or float, optional
        Maturity of the bond in years. Default is 30.

    Returns
    -------
    q_star : float
        Coupon payment per period that sets the bond price equal to par.

    Notes
    -----
    The yield ``rm`` is first converted to a per-period rate ``rm_t`` via::

        rm_t = (1 + rm)^(1/freq) - 1

    The coupon is then the annuity factor applied to that per-period rate over
    ``years * freq`` periods.
    """
    rm_t = (1.0 + rm) ** (1.0 / float(freq)) - 1.0
    q_star = rm_t / (1 - (1.0 / ((1.0 + rm_t) ** (years * freq))))

    return q_star
