import numpy as np
import matplotlib.pyplot as plt
from herramientas.percMulticapa import *

nombreArchivo = "datos/XOR_trn.csv"
capas = [3, 1]
alpha = 5
tasaAp = 5
maxErr = 0.3
maxEpoc = 80
umbral = 1e-1
graf = True

Wji = entrenar(nombreArchivo, capas, alpha, tasaAp, maxErr, 
                maxEpoc,umbral ,graf)

nombreArchivo = "datos/XOR_tst.csv"
# probar(nombreArchivo, Wji, alpha)
plt.show()



# a = np.arange(9).reshape(3, 3) - 5
# b = np.ones(3)
# # Y = sigmoidea(a, b, 0.1)

# Vi = a@b
# print(Vi)

# alpha = 0.5
# Y = 2/(1 + np.exp(-alpha * Vi)) - 1
# print(Y)
# x = np.linspace(-20, 20, 100)
# y = 2/(1 + np.exp(-alpha * x)) - 1

# plt.plot(x, y)
# plt.plot(Vi, Y, 'ro')
# plt.show()

