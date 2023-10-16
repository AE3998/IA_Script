import numpy as np
from alg_hormiga import *

# camino_optimo(nombArch, cantHorm, itMax, tasaEvap, metodoActFermona, nodoInit)

nombArch = "gr17.csv"
cantHorm = 30
itMax = 2000
tasaEvap = 0.08
# (0 = Global, 1 = Uniforme, 2 = Local)
metodoActFermona = 0
# [0 ... 17]
nodoInit = 15

mejorCamino, dist = camino_optimo(nombArch, cantHorm, itMax, tasaEvap, metodoActFermona, nodoInit)

print(f"Mejor camino encontrado: {mejorCamino}")
print(f"Distancia recorrida: {dist}")
plt.show()