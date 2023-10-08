import numpy as np 
import matplotlib.pyplot as plt

def f1(x):
    # x = [-512, 512]
    return -x*np.sin(np.sqrt(np.abs(x)))

def f2(val):
    x, y = val[0], val[1]
    # x, y = [-100, 100]
    square = (x*x + y*y)
    res = square**0.25 * (np.sin(50 * square**0.1)**2 + 1)
    return res


# x = np.linspace(-512, 512, 100)
# plt.plot(f1(x))
# plt.show()

#* i) f(x) = -x*sin(sqrt(abs(x))) 
#  con x = [-512, 512], necesito 10 bits para representar el rango 
#  de x, con esto alcanza para representar un individuo

cantIndividuo = 100
cromosoma = [10]
probMutacion = 0.1
probCruza = 0.8













