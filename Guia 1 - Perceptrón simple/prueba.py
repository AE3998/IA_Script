import numpy as np
from cargarDatos import *
from perceptronSimple import *

def prueba(nombreArchivo, W):
    """
    Funcion para realizar la prueba del perceptron simple.
    Entradas: nombre del archivo con datos y los pesos entrenados.
    Salida: tasa de error obtenida al realizar la prueba con datos nunca vistos antes.
    """
    
    print("\n--- Comienzo de las pruebas ---")

    # Cargar datos:
    (X, yd) = cargarDatos(nombreArchivo)

    # Inicializar cantidad de patrones (filas) y un contador de errores:
    cantPatrones = np.size(X, 0)
    contErrores = 0

    # Recorrer cada patron y calcular los errores
    for i in range(cantPatrones):
        Y = perceptronSimple(W, X[i, :])
        if(Y != yd[i]):
            contErrores += 1
    
    print("Tasa de error en pruebas: ", contErrores/cantPatrones * 100, "%")