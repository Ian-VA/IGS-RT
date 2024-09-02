import numpy as np
from utility_functions.legendre_polynomial import calculate_legendre_polynomial

"""

Solve the RTE with the single scattering approximation

"""

def scatter_up(mu, mu0, azr, tau0, xk):
    """

    Calculates the scattering of upward-facing radiation, returning an intensity after scattering

    Args:
        mu: The cosine of the azimuth angle
        mu0: The incidence of radiation
        azr: An array of azimuthal angles in radians
        tau0: The optical depth of the atmosphere
        xk: Expansion moments

    """

    nk = len(xk)

    smu = np.sqrt(1.0 - mu*mu)
    smu0 = np.sqrt(1.0 - mu0*mu0)

    nu = mu*mu0 + smu*smu0*np.cos(azr)
    p = np.zeros_like(nu)

    for inu, nui in enumerate(nu):
        pk = calculate_legendre_polynomial(nui, nk-1)
        p[inu] = np.dot(xk, pk) ### expansion of the phase function in Legendre series using dot product

    mup = -mu
    I1up = p*mu0/(mu0 + mup)*(1.0 - np.exp(-tau0/mup -tau0/mu0))

    return I1up

def scatter_down(mu, mu0, azr, tau0, xk):
    """

    Calculates the scattering of downward-facing radiation, returning the intensity after scattering

    Args:
        mu: The cosine of the azimuth angle
        mu0: The incidence of radiation
        azr: An array of azimuthal angles
        tau0: The optical depth
        xk: Expansion moments

    """

    nk = len(xk)
    tiny = 1.0e-8 # for 1/(mu-mu0) singular point
    smu = np.sqrt(1.0 - mu*mu)
    smu0 = np.sqrt(1.0 - mu0*mu0)
    nu = mu*mu0 + smu*smu0*np.cos(azr)
    p = np.zeros_like(nu)

    for inu, nui in enumerate(nu):
        pk = calculate_legendre_polynomial(nui, nk-1)
        p[inu] = np.dot(xk, pk) ### expansion of the phase function in Legendre series using dot product
    if np.abs(mu - mu0) < tiny:
        I1dn = p*tau0*np.exp(-tau0/mu0)/mu0
    else:
        I1dn = p*mu0/(mu0 - mu)*(np.exp(-tau0/mu0) - np.exp(-tau0/mu))

    return I1dn
