import numpy as np
from cargarDatos import *
from perceptronSimple import *
import matplotlib.pyplot as plt
from graficar import *

def entrenamiento(nombreArchivo, tasaErrorAceptable, numMaxEpocas, gamma, graficar = False):
    """
    Funcion de entrenamiento del perceptron simple. 
    Entradas: nombre del archivo, tasa de error aceptable, cantidad de epocas maxima y velocidad de aprendizaje (gamma).
    Salida: pesos actualizados una vez hecho el entrenamiento y validacion.
    """

    print("--- Comienzo del entrenamiento ---")

    # Cargar datos:
    (X, yd) = cargarDatos(nombreArchivo)

    # Inicializamos un vector de pesos con valores entre -0.5 y 0.5
    W = np.random.rand(np.size(X, 1)) - 0.5
    # ya que random.rand() genera numeros aleatorios con distribucion uniforme entre 0 y 1, y les resto 0.5
    
    # Inicializar epocas realizadas:
    contEpocas = 0
    # Inicializar tasa de error (100% para comenzar):
    tasaErrorActual = 1
    # Cantidad de patrones (cantidad de filas):
    cantPatrones = np.size(X, 0)
    # Vector para ir guardando las tasas de error obtenidas (solo para ver como varian a lo largo de las epocas):
    errores = []

    if(graficar):
        (fig, ax) = iniciarGrafica(X)
        recta = generarRecta(W, ax)

    while(contEpocas < numMaxEpocas and tasaErrorActual > tasaErrorAceptable):
        contEpocas += 1

        # --- Etapa de entrenamiento ---
        # Recorremos cada patron
        for i in range(cantPatrones):
            Y = perceptronSimple(X[i,:], W)
            if(Y != yd[i]):    # actualizar errores
                W = W + gamma/2*(yd[i] - Y) * X[i, :]
                # Actualizar grafica
                if(graficar):
                    actualizarRecta(W, recta)

        # --- Etapa de validacion ---
        contErrores = 0  
        for i in range(cantPatrones):
            Y = perceptronSimple(X[i,:], W)
            if(Y != yd[i]):      # Forma corta: contErrores += Y != yd[i]
                contErrores += 1
        
        # Calculo de tasa de error en validacion
        tasaErrorActual = contErrores/cantPatrones
        errores.append(tasaErrorActual)     # actualizo el vector de tasas de error obtenidas

    if(contEpocas == numMaxEpocas): 
        print("Corto por cantidad de epocas (",contEpocas,")")
    else:
        print("Corto por tasa de error aceptable")
        print("Epocas: ", contEpocas)

    # print("\nTasas de error obtenidas a lo largo del entrenamiento:")
    # print(errores)
    # Si veo que los errores varian muy poco durante mucho tiempo, puedo probar aumentar 
    # la velocidad de aprendizaje

    # Mostrar grafica
    # plt.show(block = False)   # mostramos la grafica en ej1_guia1.py para que no se la deba
                                # cerrar para que termine la etapa de pruebas
    
    return W    # retorno los pesos entrenados