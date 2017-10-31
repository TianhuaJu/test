import math
import matplotlib.pyplot as plt
import numpy as np
def D_func(alpha,vap_H=1,T=1,mass = 1):
    r = np.sqrt(30*8.314*T/(16*vap_H*alpha**2))
    B = r/mass
    D = 1.38*math.pow(10,-23)*B*T
    return D
D_Na = D_func(0.75,vap_H=97.42*10**3,T=298,mass=23)
print(D_Na)

plt.show()