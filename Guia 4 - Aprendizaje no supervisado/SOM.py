import numpy as np
from graficas import *
from cargarDatos import *

#! Era una implementacion no matricial que comenzamos a hacer pero la cambiamos por la otra
#! implementacion matricial para que funcione mucho mas rapido.

def obtenerNeurona(nSom, patron):
    minIdx = (0, 0)
    minDist = 1000

    for fila in range(nSom.shape[0]):
        for col in range(nSom.shape[1]):
            dist = np.linalg.norm(nSom[fila, col, :] - patron) 
            if(dist < minDist):
                minDist = dist
                minIdx = (fila, col)

    return minIdx

def obtenerVecinos(nSom, indNeuGanadora, radio):
    listVec = []

    for fila in range(nSom.shape[0]):
        for col in range(nSom.shape[1]):
            # Seria la norma 1
            dist = np.abs(fila - indNeuGanadora[0]) + np.abs(col - indNeuGanadora[1])
            if(dist <= radio):
                listVec.append((fila, col))

    return listVec

def SOM_entrenamiento(nombreArchivo, epocas, dimSom, tasaAp, radio, iris=False):
    """
        Funcion para entrenamiento de un SOM.

        El entrenamiento del SOM se realiza en tres etapas:
        Etapa 1: ordenamiento global (vecindad grande y tasa de aprendizaje grande)
        Etapa 2: transicion (reduce de forma lineal la vecindad y tasa de aprendizaje)
        Etapa 3: ajuste fino (no hay vecindad y tasa de aprendizaje chica)

        Entonces al algoritmo de entrenamiento le pasamos vectores con valores para cada etapa, 
        como las epocas, radio de vecindad y tasa de aprendizaje.
    """

    # Cargar datos
    data = cargarDatos(nombreArchivo)

    # Para ejercicio 3 donde se pide comparar entre k-medias y un SOM
    if iris:
        X, yd = data[:, :-3], data[:, -3:]
        # Reutilizar los codigos
        data = X

    # Inicializar pesos al azar [-0.5, 0.5]
    neuronasSOM = np.random.rand(dimSom[0], dimSom[1], data.shape[1]) - 0.5

    fig, ax, rectHoriz, rectVert = iniciarGraficaSOM(data, neuronasSOM, iris)

    #todo 1ra etapa: ordenamiento global
    for epoca in range(epocas[0]):
        for i in range(data.shape[0]):
            patron = data[i, :]
            neurGanadora = obtenerNeurona(neuronasSOM, patron)
            listVec = obtenerVecinos(neuronasSOM, neurGanadora, radio[0])
            
            # Actualizar cada uno de los vecinos, incluyendo la neurona misma
            for idx in listVec:
                neuronasSOM[idx] += tasaAp[0] * (patron - neuronasSOM[idx])
        
        # epoca % 4 == 0 es para que grafique cada 4 epocas nomas
        if(epoca % 4 == 0 or epoca == epocas[0]-1):
            title = "Ordenamiento global ep " + str(epoca)
            actualizarGraficaSOM(fig, ax, title, neuronasSOM, rectHoriz, rectVert)

    # Graficar la ultima epoca
    title = "Ordenamiento global ep " + str(epocas[0])
    actualizarGraficaSOM(fig, ax, title, neuronasSOM, rectHoriz, rectVert)

    #todo 2da etapa: transicion
    # el radio de la vecindad y la tasa de aprendizaje se reducen de forma lineal
    radioEtapa2 = np.linspace(radio[0], radio[1], epocas[1])
    tasaApEtapa2 = np.linspace(tasaAp[0], tasaAp[1], epocas[1])

    for epoca in range(epocas[1]):
        for i in range(data.shape[0]):
            patron = data[i, :]
            neurGanadora = obtenerNeurona(neuronasSOM, patron)
            listVec = obtenerVecinos(neuronasSOM, neurGanadora, radioEtapa2[epoca])
            
            # Actualizar cada uno de los vecinos, incluyendo la neurona misma
            for idx in listVec:
                neuronasSOM[idx] += tasaApEtapa2[epoca] * (patron - neuronasSOM[idx])
        
        if(epoca % 4 == 0 or epoca == epocas[1]-1):
            title = "Etapa transicion ep " + str(epoca)
            actualizarGraficaSOM(fig, ax, title, neuronasSOM, rectHoriz, rectVert)

    # Graficar la ultima epoca
    title = "Etapa transicion ep " + str(epocas[1])
    actualizarGraficaSOM(fig, ax, title, neuronasSOM, rectHoriz, rectVert)

    #todo 3er etapa: ajuste fino
    for epoca in range(epocas[2]):
        for i in range(data.shape[0]):
            patron = data[i, :]
            neurGanadora = obtenerNeurona(neuronasSOM, patron)
            listVec = obtenerVecinos(neuronasSOM, neurGanadora, radio[1])
            
            # Actualizar cada uno de los vecinos, incluyendo la neurona misma
            for idx in listVec:
                neuronasSOM[idx] += tasaAp[1] * (patron - neuronasSOM[idx])
        
        if(epoca % 4 == 0 or epoca == epocas[2]-1):
            title = "Ajuste fino ep " + str(epoca)
            actualizarGraficaSOM(fig, ax, title, neuronasSOM, rectHoriz, rectVert)

    # Graficar la ultima epoca
    title = "Gráfica final ep " + str(epocas[0])
    actualizarGraficaSOM(fig, ax, title, neuronasSOM, rectHoriz, rectVert)

    return