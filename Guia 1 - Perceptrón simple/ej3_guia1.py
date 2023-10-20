from entrenamiento import *
from prueba import *

# --- Entrenamiento y prueba con particiones utilizando desviaciones del 50% ---
nombreArchivo = "icgtp1datos/OR_50_trn.csv"
gamma = 0.1
tasaErrorAceptable = 0
numMaxEpocas = 10
graficar = True
print("--- OR con desviaciones del 50% ---")
W = entrenamiento(nombreArchivo, tasaErrorAceptable, numMaxEpocas, gamma, graficar)
nombreArchivo = "icgtp1datos/OR_50_tst.csv"
prueba(nombreArchivo, W)

# --- Entrenamiento y prueba con particiones utilizando desviaciones del 90% ---
# nombreArchivo = "icgtp1datos/OR_90_trn.csv"
# gamma = 0.01
# tasaErrorAceptable = 0.06
# numMaxEpocas = 100
# graficar = True
# print("\n\n--- OR con desviaciones del 90% ---")
# W = entrenamiento(nombreArchivo, tasaErrorAceptable, numMaxEpocas, gamma, graficar)
# nombreArchivo = "icgtp1datos/OR_90_tst.csv"
# prueba(nombreArchivo, W)

# para mantener la grafica
plt.show()