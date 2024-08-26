import numpy as np
from math.gauss_calculation import gauss_node_calculate
from math.legendre_polynomial import calculate_legendre_polynomial, calculate__associated_legendre_polynomial
from numpy._typing import _80Bit

def gauss_seidel_iterations(m, mu0, nit, ng1, nlr, dtau, xk):
    """

    Computes the m-th Fourier moment of the diffuse light at Gauss nodes and at all optical depth levels in the atmosphere, including TOA and BOA

    Args:
        m: A list of fourier moments
        mu0: Cosine of the solar zenith angle 
        srfa: Lambertian surface albedo / reflection coefficient
        nit: Number of iterations 
        ng1: Number of gauss nodes per hemisphere
        nlr: Number of layer elements
        dtau: Optical thickness of each layer element
        xk: An array containing the phase function expansion scaled by single scattering albedo w_o/2 and (2k+1)

    """

    float_compare_tiny = 1.0e-8 # to compare floats
    nb = nlr+1 # number of boundaries
    nk = len(xk) # number of expansion moments
    ng2 = ng1*2 # number of gauss nodes per sphere
    tau0 = nlr*dtau # total optical depth
    tau = np.linspace(0.0, tau0, nb) # embedding of boundaries

    mup, w = gauss_node_calculate(0.0, 1.0, ng1) # compute gauss nodes first per hemisphere
    mug = np.concatenate((-mup, mup)) # extend gauss X for whole hemisphere
    wg = np.concatenate((w, w)) # extend gauss weights for whole hemisphere

    pk = np.zeros((ng2, nk)) # legendre polynomials
    p = np.zeros(ng2) # m-th moment of the phase function

    if m == 0:
        ### if m == 0, we can just use regular legendre polynomials, since m > 0 means azimuthal dependence 
        pk0 = calculate_legendre_polynomial(mu0, nk-1)
        for ig in range(ng2):
            pk[ig, :] = calculate_legendre_polynomial(mug[ig], nk-1)
            p[ig] = np.dot(xk, pk[ig, :]*pk0)

    else:
        pk0 = calculate_associated_legendre_polynomial(m, mu0, nk-1)
        for ig in range(ng2):
            pk[ig, :] = calculate_associated_legendre_polynomial(m, mug[ig], nk-1)
            p[ig] = np.dot(xk, pk[ig, :]*pk0)


    """

    Solution to the Radiative Transfer equation in the single scattering approximation at positive Gauss nodes and all layer boundaries (so, the first dimension)

    """


    ### _dn means computed with down scattering

    I1dn = np.zeros(ng1)

    for ig in range(ng1):
        mu = mup[ig]
        if (np.abs(mu0 - mu) < float_compare_tiny): # uh oh!! singularity
            I1dn[ig] = p[ng1+ig]*dtau*np.exp(-dtau/mu0)/mu0
        else:
            I1dn[ig] = p[ng1+ig]*mu0/(mu0 - mu) * (np.exp(-dtau/mu0) - np.exp(-dtau/mu))

    Idn = np.zeros((nb, ng1)) # single scattering at all levels
    Idn[1, :] = I1dn

    for ib in range(2, nb):
        Idn[ib, :] = \
                Idn[ib-1, :]*np.exp(-dtau/mup) + \
                    I1dn*np.exp(-tau[ib-1]/mu0)

    ### _up means computed with up scattering

    I1up = p[0:ng1]*mu0/(mu0 + mup) * (1.0 - np.exp(-dtau/mup -dtau/mu0)) # single scattering from one layer

    Iup = np.zeros_like(I1up) # single scattering at all boundaries

    if m == 0 and srfa > float_compare_tiny:
        Iup[nb-1, :] = 2.0 * srfa * mu0 * np.exp(-tau0/mu0)
        Iup[nb-2, :] = Iup[nb-1, :]*np.exp(-dtau/mup) + \
                            I1up*np.exp(-tau[nb-2]/mu0)

    else:
        Iup[nb-2, :] = I1up*np.exp(-tau[nb-2]/mu0)

    for ib in range(nb-3, -1, -1):
        Iup[ib, :] = Iup[ib+1, :]*np.exp(-dtau/mup) + \
                            I1up*np.exp(-tau[ib]/mu0)

gauss_seidel_iterations(0, 0.5, 0.11, 1, 4, 4, [0, 5, 5, 3])




