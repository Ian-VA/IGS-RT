import numpy as np

def calculate_legendre_polynomial(x, kmax):
    """

    Calculates an ordinary Legendre polynomial P(x) given scalar x up to order kmax

    args:
        x: Cosine of a zenith angle where x=[-1:1]
        kmax: The highest order of the legendre polynomial

    """

    nk = kmax+1
    pk = np.zeros(nk) # a list of P(x), where a corresponding index K is P_k(x) (the Legendre polynomial with order K given x)

    if kmax == 0: # for k=0, the legendre polynomial is obviously 1
        pk[0] = 1.0 
    elif kmax == 1: # for k=1, the legendre polynomial is known to be x
        pk[0] = 1.0
        pk[1] = x
    else: # if highest order is greater than 1
        pk[0] = 1.0
        pk[1] = x

        for ik in range(2, nk): # if k > 1, use recurrence relation (k+1)*P_(k+1)(x) = (2k+1) * P_k(x) - k*P_(k-1)(x) with indices redefined as k -> k-1 and both sides scaled by 1/k for convenience

            pk[ik] = (2.0 - 1.0/ik) * x * pk[ik-1] - (1.0- 1.0/ik) * pk[ik-2]

        return pk

def calculate_associated_legendre_polynomial(x, kmax, m):
    """

    Calculates an associated Legendre polynomial P(x) given scalar x up to order kmax and Fourier degree m

    args:
        x: Cosine of a zenith angle where x=[-1:1]
        kmax: The highest order of the legendre polynomial
        m: The Fourier degree (>0)

    """

    nk = kmax+1
    qk = np.zeros(nk)


    c0 = 1.0
    for ik in range(2, 2*m+1, 2):
        c0 = c0 - c0/ik

    qk[m] = np.sqrt(c0) * np.power(np.sqrt(1.0 - x*x), m)

    m1= m*m - 1.0
    m4 = m*m - 4.0

    for ik in range(m+1, nk):
        c1 = 2.0*ik - 1.0
        c2 = np.sqrt((ik + 1.0) * (ik - 3.0) - m4)
        c3 = 1.0/np.sqrt((ik+1.0)* (ik-1.0) - m1)
        qk[ik] = (c1*x*qk[ik-1] - c2*qk[ik-2])*c3

    return qk
