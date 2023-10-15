import numpy as np
from alg_enjambre import *
from graficas import *

#* Funciones del ejercicio 1 guia 6
def f1(x):      # x = [-512, ..., 512]
    return -x*np.sin((np.abs(x))**0.5)    

def f2(val):    # f(x, y) con x, y = [-100, 100] 
    x, y = val[:, 0], val[:, 1]

    square = (x*x + y*y)
    res = square**0.25 * (np.sin(50 * square**0.1)**2 + 1)
    return res

#* Utilizar algoritmo de enjambre

# (func, cantIdv, maxIter, c1, c2, xmin, xmax)

#* Funcion f1(x)
func = f1
cantIdv = 20
maxIter = 100
c1 = 0.8
c2 = 0.2
xmin = -512
xmax = 512

print("\nFuncion f1(x):")
mejorPosEnjambre = enjambre_mejor_global(func, cantIdv, maxIter, c1, c2, xmin, xmax)

#* Funcion f2(x,y)
func = f2
cantIdv = 20
maxIter = 100
c1 = 0.8
c2 = 0.2
xmin = [-100, -100]
xmax = [100, 100]

print("\nFuncion f2(x, y):")
mejorPosEnjambre = enjambre_mejor_global(func, cantIdv, maxIter, c1, c2, xmin, xmax)
plt.show()

# ? Test f1
# a = np.arange(3).reshape(3, 1)
# print(f1(a))

# ? Test f2
# a = np.arange(10).reshape(5, 2)
# print(f2(a))