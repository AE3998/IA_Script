from entrenar import *
from probar import probar
import numpy as np


# ==========[Entrenar y probar los datos de OR_50]==============
nombreArchivo = "datos/OR_50_trn.csv"
velApren= 0.02
tasaErr = 0.15
maxEpoca = 10
graficar = True
print("Comenzando el entrenamiento de caso OR 50...")
Wi = entrenarPesos(nombreArchivo, velApren, tasaErr, maxEpoca, graficar)

# Prueba 
nombreArchivo = "datos/OR_50_tst.csv"
print("\nComprobando la neurona entrenada...")
probar(nombreArchivo, Wi)

# ==========[Entrenar y probar los datos de OR_90]==============
nombreArchivo = "datos/OR_90_trn.csv"
velApren= 0.02
tasaErr = 0.15
maxEpoca = 10
graficar = True
print("\n" * 2)
print("Comenzando el entrenamiento de caso OR 90...")
Wi = entrenarPesos(nombreArchivo, velApren, tasaErr, maxEpoca, graficar)

# Prueba 
nombreArchivo = "datos/OR_90_tst.csv"
print("\nComprobando la neurona entrenada...")
probar(nombreArchivo, Wi)