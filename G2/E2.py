import numpy as np
import matplotlib.pyplot as plt
from herramientas.percMulticapa import *

nombreArchivo = "datos/concent_trn.csv"
capas = [4, 1]
maxErr = 0.1
maxEpoc = 500
tasaAp = 0.2
alpha = 5
umbral = 0.08
graf = True
XOR = False

Wji = entrenar(nombreArchivo, capas, alpha, tasaAp, maxErr, 
                maxEpoc,umbral ,graf, XOR)
