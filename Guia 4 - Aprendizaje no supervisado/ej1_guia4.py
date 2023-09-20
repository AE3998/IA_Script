import numpy as np
import matplotlib.pyplot as plt
from SOM_entrenamiento import *

nombreArchivo = "circulo.csv"
epocas = [150, 300, 150]
dimSom = [5, 5] # en (i, j)
tasaAp = [0.25, 0.1]
radio = [2, 0.1]

SOM_entrenamiento(nombreArchivo, epocas, dimSom, tasaAp, radio)
plt.show()