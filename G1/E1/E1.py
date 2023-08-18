from entrenar import *
from probar import probar
import numpy as np


# ==========[Entrenar y probar los datos de OR]==============
nombreArchivo = "datos/OR_trn.csv"
velApren= 0.02
tasaErr = 0.15
maxEpoca = 10
graficar = True
print("Comienza el entrenamiento de caso OR...")
Wi = entrenarPesos(nombreArchivo, velApren, tasaErr, maxEpoca, graficar)

# Prueba 
nombreArchivo = "datos/OR_tst.csv"
print("\nComprobar el funcionamiento de la neurona entrenada...")
probar(nombreArchivo, Wi)


# ==========[Entrenar y probar los datos de XOR]==============
nombreArchivo = "datos/XOR_trn.csv"
velApren= 0.02
tasaErr = 0.15
maxEpoca = 10
graficar = False
print("\n" * 2)
print("Comienza el entrenamiento de caso XOR...")
Wi = entrenarPesos(nombreArchivo, velApren, tasaErr, maxEpoca, graficar)

# Prueba 
nombreArchivo = "datos/XOR_tst.csv"
print("\nComprobar el funcionamiento de la neurona entrenada...")
probar(nombreArchivo, Wi)