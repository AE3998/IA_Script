import numpy as np
from alg_enjambre import *

#* Funciones del ejercicio 1 guia 6
def f1(x):      # x = [-512, ..., 512]
    return -x*np.sin(np.sqrt(np.abs(x)))    

def f2(val):    # f(x, y) con x, y = [-100, 100]
    x, y = val[0], val[1]   
    square = (x*x + y*y)
    res = square**0.25 * (np.sin(50 * square**0.1)**2 + 1)
    return res

#* Utilizar algoritmo de enjambre
# mejorPosEnjambre = enjambre_mejor_global(func, cantParticulas, dim, maxIter, c1_ini, c1_fin, c2_ini, c2_fin, rango)

#* Funcion f2(x, y)
mejorPosEnjambre = enjambre_mejor_global(f1, 30, 1, 100, [1.2, 0.5], [0.5, 1.2], [[-512, 512]])

#* Funcion f2(x, y)
#! Falta corregir enjambre_mejor_global para que funcion con funciones de dos variables como f2
# mejorPosEnjambre = enjambre_mejor_global(f2, 30, 2, 100, [0.7, 0.3], [0.3, 0.7], [[-100, 100], [-100, 100]])