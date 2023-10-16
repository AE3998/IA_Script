import numpy as np
from alg_hormiga import *

# camino_optimo(nombArch, cantHorm, itMax, tasaEvap, metodoActFeromona, nodoInit)

# Dijimos que eligiendo un valor para la tasa de evaporacion "p" (0.1, 0.90) hay que probar con 
# distintos valores para la cantidad de feromonas "τ" (0.1, 1 y 10) y armar una tabla con esos
# valores y con los metodos (uniforme, local, global) para comparar.
# Por ejemplo, con p = 0.1, τ = 0.1 y el metodo de deposito de feromonas uniforme, repetir N = 10 
# veces la corrida y con esas 10 soluciones calcular el tiempo promedio que llevo encontrar 
# la solucion, la distancia promedio de la solucion encontrada y el número de iteraciones promedio 
# utilizado. Y lo mismo para las demas combinaciones de parametros.
# Luego empleando otra tasa de evaporacion “p” realizamos lo mismo y comparamos cual es el efecto 
# de esos parametros sobre la calidad de las soluciones encontradas.
# Esta explicado y con una tabla de ejemplo al final del word de anotaciones

nombArch = "gr17.csv"
cantHorm = 30
itMax = 1000
tasaEvap = 0.1     
# (0 = Global, 1 = Uniforme, 2 = Local)
metodoActFeromona = 0       # el metodo local es el que mas iteraciones usa
# [0 ... 17]
nodoInit = 15   # nodo inicial donde se ubican las hormigas

mejorCamino, dist = camino_optimo(nombArch, cantHorm, itMax, tasaEvap, metodoActFeromona, nodoInit)

print(f"Mejor camino encontrado: {mejorCamino}")
print(f"Distancia recorrida: {dist}")
plt.show()