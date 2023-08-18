import numpy as np
from entrenar import cargarDatos, perceptron

def probar(nombreArchivo, Wi):
    Xi, Yd = cargarDatos(nombreArchivo)

    nPatron = np.size(Xi, 0)
    contErr = 0
    for i in range(nPatron):
        Y = perceptron(Wi, Xi[i, :])
        contErr += Y != Yd[i]
    
    print("Tasa de error de comprobacion ", contErr/nPatron * 100, "%")
