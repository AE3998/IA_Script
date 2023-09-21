import numpy as np
import matplotlib.pyplot as plt
from SOM_entrenamiento import *
# from SOM import *

nombreArchivo = "circulo.csv"
epocas = [150, 300, 150]
dimSom = [6, 6] # en (i, j)
tasaAp = [0.25, 0.1]
radio = [2, 0.1]

# Cargar datos
data = cargarDatos(nombreArchivo)

SOM_entrenamiento(data, epocas, dimSom, tasaAp, radio)
plt.show()