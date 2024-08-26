import numpy as np


### Calculate gaussian nodes from x1:x2 for numerical integration

def gauss_node_calculate(x1, x2, n):
    const_boundary = 3.0e-14
    x, w = np.zeros(n)
    m = int((n + 1) / 2)
    yxm, yxl = (x2 + x1) / 2, (x2 - x1) / 2

    for i in range(m):
        yz = np.cos(np.pi*(i + 0.75) / (n+0.5))
        while True:
            yp1, yp2 = 1.0, 0.0
            for j in range(n):
                yp3 = yp2
                yp2 = yp1
                yp1 = ((2.0 * j + 1.0)*yz*yp2 - j*yp3)/ (j+1)

            ypp = n*(yz*yp1 - yp2) / (yz**2 - 1.0)
            yz1 = yz
            yz = yz1 - yp1/ypp

            if (np.abs(yz - yz1) < const_boundary):
                break

        x[i] = yxm - yz*yxl
        w[n-1-i] = yxm + yxl*yz
        w[i] = 2.0*yxl/((1.0 - yz**2) * ypp**2)
        w[n-1-i] = w[i]

    return x, w # finally, return the node and corresponding weight to numerically integrate
