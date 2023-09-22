import numpy as np
import matplotlib.pyplot as plt
from SOM_entrenamiento import *
# from SOM import *
"""
    El entrenamiento del SOM se realiza en tres etapas:
    Etapa 1: ordenamiento global (vecindad grande y tasa de aprendizaje grande)
    Etapa 2: transicion (reduce de forma lineal la vecindad y tasa de aprendizaje)
    Etapa 3: ajuste fino (no hay vecindad y tasa de aprendizaje chica)

    Entonces al algoritmo de entrenamiento le pasamos vectores con valores para cada etapa, 
    como las epocas, radio de vecindad y tasa de aprendizaje.
"""
nombreArchivo = "circulo.csv"
epocas = [80, 200, 80]
dimSom = [4, 4] # en (i, j)
tasaAp = [0.45, 0.08]
radio = [2, 0.1]

# Cargar datos
data = cargarDatos(nombreArchivo)

neurSom, clusters = SOM_entrenamiento(data, epocas, dimSom, tasaAp, radio)
colorearClustersSOM(data, neurSom, clusters)
plt.show()
#! FALTA CORREGIR EL ENTRENAMIENTO DEL SOM PARA QUE FUNCIONA CON UN SOM UNIDIMENSIONAL
#! COMO PIDE AL FINAL DEL EJERCICIO 1
#* Repetir para un SOM unidimensional con la misma cantidad de neuronas
# nombreArchivo = "te.csv"
# epocas = [50, 100, 50]
# dimSom = [25] # en (i, j)
# tasaAp = [0.25, 0.1]
# radio = [2, 0.1]

# SOM_entrenamiento(nombreArchivo, epocas, dimSom, tasaAp, radio)
# plt.show()
