import numpy as np
from algGenet_Ej2 import *

# En el word con anotaciones esta explicado como se piensa y resuelve el ejercicio

#? Test 
archivo_train = "leukemia_test.csv"
archivo_test = "leukemia_test.csv"
cantIndividuos = 50
cantPadres = 0.5
cantMaxGeneracion = 500
probMutacion = 0.8
probCruza = 0.1
alpha = 1
beta = 0.15
fitnessBuscado = 0.9

mejorIndv, mejorFit = algGenetico(archivo_train, archivo_test, cantIndividuos, cantPadres,cantMaxGeneracion, 
                                probMutacion, probCruza, alpha, beta, fitnessBuscado)

cantFeature = np.sum(mejorIndv)
porcentaje = np.round(cantFeature/7129 * 100, 2)
print(f"Mejor individuo que usa {cantFeature} caracteristicas")
print(f"que es {porcentaje} % del total.")
print(f"Logra un fitness de: {mejorFit}")