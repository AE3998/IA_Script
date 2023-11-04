import numpy as np
import matplotlib.pyplot as plt
from herramientas.percMulticapa import *

nombreArchivo = "datos/concent_trn.csv"
capas = [3, 1]
maxErr = 0.1
maxEpoc = 201
tasaAp = 0.005
alpha = 5
umbral = 0.08
graf = True
XOR = False

Wji = entrenar(nombreArchivo, capas, alpha, tasaAp, maxErr, 
                maxEpoc,umbral ,graf, XOR)

nombreArchivo = "datos/concent_tst.csv"
probar(nombreArchivo, Wji, alpha, umbral,graf, XOR)