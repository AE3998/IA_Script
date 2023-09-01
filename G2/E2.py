import numpy as np
import matplotlib.pyplot as plt
from herramientas.percMulticapa import *

nombreArchivo = "datos/concent_trn.csv"
capas = [5, 1]
maxErr = 0.1
maxEpoc = 500
tasaAp = 0.1
alpha = 1
umbral = 0.08
graf = True

Wji = entrenar(nombreArchivo, capas, alpha, tasaAp, maxErr, 
                maxEpoc,umbral ,graf)
