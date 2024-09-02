from gauss_seidel_iterations import gauss_seidel_iterations
from utility_functions.legendre_polynomial import calculate_associated_legendre_polynomial, calculate_legendre_polynomial
import numpy as np

def source_function_integrate_down(m, mu, mu0, nlr, dtau, xk, mug, wg, Ig05):
    tiny = 1.0e-8
    ng2 = len(wg)
    nk = len(xk)
    nb = nlr+1
    tau0 = nlr*dtau
    tau = np.linspace(0.0, tau0, nb)

    pk = np.zeros((ng2, nk))

    if m == 0:
        pk0 = calculate_legendre_polynomial(mu0, nk-1)
        pku = calculate_legendre_polynomial(mu, nk-1)

        for ig in range(ng2):
            pk[ig, :] = calculate_legendre_polynomial(mug[ig], nk-1)
    else:
        pk0 = calculate_associated_legendre_polynomial(m, mu0, nk-1)
        pku = calculate_associated_legendre_polynomial(m, mu, nk-1)

        for ig in range(ng2):
            pk[ig, :] = calculate_associated_legendre_polynomial(m, mug[ig], nk-1)

    p = np.dot(xk, pku*pk0)

    if np.abs(mu - mu0) < tiny:
        I11dn = p*dtau*np.exp(-dtau/mu0)/mu0
    else:
        I11dn = p*mu0/(mu0 - mu) * (np.exp(-dtau/mu0) - np.exp(-dtau/mu))

    I1dn = np.zeros(nb)
    I1dn[1] = I11dn

    for ib in range(2, nb):
        I1dn[ib] = I1dn[ib-1]*np.exp(-dtau/mu) + I11dn*np.exp(-tau[ib-1]/mu0)

    wpij = np.zeros(ng2)
    for jg in range(ng2):
        wpij[jg] = wg[jg]*np.dot(xk, pku[:]*pk[jg, :])

    Idn = np.copy(I1dn)
    J = np.dot(wpij, Ig05[0, :])
    Idn[1] = I11dn + (1.0 - np.exp(-dtau/mu)) * J

    for ib in range(2, nb):
        J = np.dot(wpij, Ig05[ib-1, :])
        Idn[ib] = Idn[ib-1]*np.exp(-dtau/mu) + I11dn*np.exp(-tau[ib-1]/mu0) + (1.0 - np.exp(-dtau/mu)) * J

    return Idn[nb-1] - I1dn[nb-1]

def source_function_integrate_up(m, mu, mu0, srfa, nlr, dtau, xk, mug, wg, Ig05, Igboa):
    tiny = 1.0e-8
    ng2 = len(wg)
    nk = len(xk)
    nb = nlr+1
    tau0 = nlr*dtau
    tau = np.linspace(0.0, tau0, nb)

    pk = np.zeros((ng2, nk))

    if m == 0:
        pk0 = calculate_legendre_polynomial(mu0, nk-1)
        pku = calculate_legendre_polynomial(mu, nk-1)

        for ig in range(ng2):
            pk[ig, :] = calculate_legendre_polynomial(mug[ig], nk-1)
    else:
        pk0 = calculate_associated_legendre_polynomial(m, mu0, nk-1)
        pku = calculate_associated_legendre_polynomial(m, mu, nk-1)

        for ig in range(ng2):
            pk[ig, :] = calculate_associated_legendre_polynomial(m, mug[ig], nk-1)

    p = np.dot(xk, pku*pk0)

    if np.abs(mu - mu0) < tiny:
        I11up = p*dtau*np.exp(-dtau/mu0)/mu0
    else:
        I11up = p*mu0/(mu0 - mu) * (np.exp(-dtau/mu0) - np.exp(-dtau/mu))

    I1up = np.zeros(nb)
    I1up[1] = I11up

    for ib in range(2, nb):
        I1up[ib] = I1up[ib-1]*np.exp(-dtau/mu) + I11up*np.exp(-tau[ib-1]/mu0)

    wpij = np.zeros(ng2)

    for jg in range(ng2):
        wpij[jg] = wg[jg]*np.dot(xk, pku[:]*pk[jg, :])

    Iup = np.copy(I1up)

    if m == 0 and srfa > 0.0:
        Iup[nb-1] = 2.0*srfa*np.dot(Igboa, mug[ng1:ng2]*wg[ng1:ng2]) + 2.0*srfa*mu0*np.exp(-tau0/mu0)

    J = np.dot(wpij, Ig05[nb-2, :])
    Iup[nb-2] = Iup[nb-1]*np.exp(-dtau/mup)


