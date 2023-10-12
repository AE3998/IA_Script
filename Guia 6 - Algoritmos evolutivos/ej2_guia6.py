import numpy as np
from algGenet_Ej2 import *

# En el word con anotaciones esta explicado como se piensa y resuelve el ejercicio

#? Test 
archivo_train = "leukemia_train.csv"
archivo_test = "leukemia_test.csv"
cantIndividuos = 100
cantPadres = 0.5
cantMaxGeneracion = 200
probMutacion = 0.1
probCruza = 0.8
alpha = 0.9
beta = 0.2
fitnessBuscado = 0.85

mejorIndv, mejorFit = algGenetico(archivo_train, archivo_test, cantIndividuos, cantPadres,cantMaxGeneracion, 
                                probMutacion, probCruza, alpha, beta, fitnessBuscado)

cantFeature = np.sum(mejorIndv)
porcentaje = np.round(cantFeature/7129 * 100, 2)
print(f"El mejor individuo usa {cantFeature} caracteristicas")
print(f"que es {porcentaje} % del total.")
print(f"Logra un fitness de: {mejorFit}")

# Guardamos el mejor individuo en un csv para comprobar el funcionamiento del
# clasificador con los datos filtrados (en comprobarEj2.py)
np.savetxt("mejorIndv.csv", mejorIndv, delimiter=",")