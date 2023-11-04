from entrenamiento import *
from prueba import *
import matplotlib.pyplot as plt

# Ejercicio 1 y 2

# --- Resolucion OR ---
tasaErrorAceptable = 0   # debe tener 100% de acierto porque el OR es un problema facil de resolver
numMaxEpocas = 10
gamma = 0.01
graficar = True
W = entrenamiento("icgtp1datos/OR_trn.csv", tasaErrorAceptable, numMaxEpocas, gamma, graficar)
prueba("icgtp1datos/OR_tst.csv", W)

# --- Resolucion XOR ---
# tasaErrorAceptable = 0.26   
# numMaxEpocas = 100   # Epoca = pasar todos los patrones
# gamma = 0.1  
# graficar = False
# W = entrenamiento("icgtp1datos/XOR_trn.csv", tasaErrorAceptable, numMaxEpocas, gamma, graficar)
# prueba("icgtp1datos/XOR_tst.csv", W)

plt.show()  # mostramos la grafica aca para que no se la deba cerrar para que
            # muestre los resultados de las pruebas

# Como los pesos se inicializan al azar, cada vez que se ejecuta varía el resultado obtenido
# aunque se utilicen los mismos valores para gamma, tasa de error y datos.