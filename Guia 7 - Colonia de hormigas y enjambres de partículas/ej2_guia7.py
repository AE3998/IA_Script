import numpy as np
from alg_hormiga import *
from rich.table import Table
from rich.console import Console
from rich.live import Live
import time

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

#* =======[Parametro de entrada]=======
nombArch = "gr17.csv"
cantHorm = 30
itMax = 1500
tasaEvap = 0.9     
# (0 = Global, 1 = Uniforme, 2 = Local)
metodoActFeromona = [0, 1, 2]       
nombreMetodo = ['Global', 'Uniforme', 'Local']
nodoInit = 0   # [0 ... 17] nodo inicial donde se ubican las hormigas
cantFermona = [0.1, 1, 10]      # [0.1, 1, 10]  le ponemos nombre "Q" en la tabla
graf = False

#* =======[Inicializacion de los parametros del ciclo]=======
tiempo = 0
table = Table()
console = Console()
datos = ["metodo", "tasaEvap", "Q", "seg.", "dist", "iter"]
# metodo, rho, tau, tiempo, distancia, iteracion
# rho = \u03C1", tau = "\u03C4"
for dato in datos:
    table.add_column(dato, style='bold')

# Manipular los resultados en forma numerica
np.set_printoptions(precision=2, suppress=True)
totalDatos = len(metodoActFeromona) * len(cantFermona)
datosNum = np.zeros(shape=(totalDatos, len(datos)), dtype=float)
idxDatos = 0

#* =======[Ciclo de entrenamiento]=======
minDistActual = 9999999
mejorCaminoActual = 0
with console.capture() as capture:
    # Aplicamos el algoritmo para cada metodo (global, uniforme y local) y para cada cant de feromonas
    for idxMet, metodo in enumerate(metodoActFeromona):
        for Q in cantFermona:
            timeInit = time.time()

            #* Entrenamiento
            mejorCamino, dist, it = camino_optimo(nombArch, cantHorm, itMax, tasaEvap, metodo, nodoInit, Q, graf)
            tiempo = round(time.time() - timeInit, 2)
            
            if(dist < minDistActual):
                minDistActual = dist
                mejorCaminoActual = mejorCamino

            #? Agregar los resultados en las tablas 
            table.add_row(nombreMetodo[idxMet], 
                        str(tasaEvap),
                        str(Q),
                        str(tiempo),
                        str(dist), 
                        str(it))  

            datosNum[idxDatos] = np.round(np.array([idxMet, tasaEvap, Q, tiempo, dist, it]), 2)
            idxDatos += 1
            
    console.print(table)

# Extraer la tabla en string
table_str = capture.get()

print("\nMejor camino encontrado:", mejorCaminoActual)
print("Distancia recorrida:", minDistActual)
# print(f"Tiempo de ejecucion: {round(tiempo, 2)} seg.")

#* Mostrar los datos
print("\nTabla comparativa:\n", table_str)
print(f"\nMi ndarray!!\n", datosNum)

#* =======[Guardar los datos en un archivo]=======
#! Escribir en el archivo txt las tablas
with open('tablas.txt', 'a') as file:
    file.write(table_str)

#! Escribir los datos numericos en un csv
with open('tablas.csv', 'a') as file:
    np.savetxt(file, datosNum, delimiter=',')

if(graf):
    plt.show()