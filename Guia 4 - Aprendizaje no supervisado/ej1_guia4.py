import numpy as np
import matplotlib.pyplot as plt
from SOM_entrenamiento import *

"""
    El entrenamiento del SOM se realiza en tres etapas:
    Etapa 1: ordenamiento global (vecindad grande y tasa de aprendizaje grande)
    Etapa 2: transicion (reduce de forma lineal la vecindad y tasa de aprendizaje)
    Etapa 3: ajuste fino (no hay vecindad y tasa de aprendizaje chica)

    Entonces al algoritmo de entrenamiento le pasamos vectores con valores para cada etapa, 
    como las epocas, radio de vecindad y tasa de aprendizaje.
"""
nombreArchivo = "te.csv"
epocas = [80, 200, 80]
dimSom = [10, 10] # en (i, j)
tasaAp = [0.5, 0.01]
radio = [2, 0.1]

# Cargar datos
data = cargarDatos(nombreArchivo)

neuronasSOM, clusters = SOM_entrenamiento(data, epocas, dimSom, tasaAp, radio)
colorearClustersSOM(data, neuronasSOM, clusters)

# plt.show()

#* Repetir para un SOM unidimensional con la misma cantidad de neuronas

nombreArchivo = "te.csv"
epocas = [50, 100, 50]
dimSom = [1, 100] # en (i, j)
tasaAp = [0.5, 0.1]
radio = [2, 0.1]

# Cargar datos
data = cargarDatos(nombreArchivo)

neuronasSOM, clusters = SOM_entrenamiento(data, epocas, dimSom, tasaAp, radio)
colorearClustersSOM(data, neuronasSOM, clusters)
plt.show()

#* Detalles observados:
# - Donde hay mas densidad de puntos habra mas densidad de neuronas, es decir, el SOM no solo
# copia la forma de los datos sino tambien la densidad. Por eso en espacios que hay pocos datos
# habra menos neuronas y donde hay mas datos habra mas neuronas.
# - Otra cosa que podemos ver usando te.csv y un SOM bidimensional, es que en algunos casos quedan 
# una o dos neuronas fuera de los datos, eso es porque estan conectadas con una vecina de cada lado
# y durante el ordenamiento esas dos vecinas la fueron llevando hacia su lugar hasta que en un 
# momento quedaron ahi fuera y nunca se actualizaron solas porque quedaron mas lejos de los datos
# que las demas neuronas.