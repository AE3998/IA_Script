import numpy as np
from graficas import *

def cargarDatos(nombreArchivo):
    data = np.genfromtxt(nombreArchivo, delimiter=",", max_rows=None)
    return data

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

def obtenerVecinos(nSom, winIdx, radio):
    listVec = []

    for fila in range(nSom.shape[0]):
        for col in range(nSom.shape[1]):
            # Seria la norma 1
            dist = np.abs(fila - winIdx[0]) + np.abs(col - winIdx[1])
            if(dist <= radio):
                listVec.append((fila, col))

    return listVec

def SOM_entrenamiento(nombreArchivo, epocas, dimSom, tasaAp, radio, iris=False):
    """
        Funcion para entrenamiento de un Som.
        Entradas: nombre del archivo csv de datos, vector epocas donde se tenga las epocas para
        las tres etapas de entrenamiento,... 
    """

    # Cargar datos
    data = cargarDatos(nombreArchivo)
    if iris:
        X, yd = data[:, :-3], data[:, -3:]
        # Reutilizar los codigos
        data = X

    # Inicializar pesos al azar [-0.5, 0.5]
    neurSom = np.random.rand(dimSom[0], dimSom[1], data.shape[1]) - 0.5

    fig, ax, rectHoriz, rectVert = iniciarGrafica(data, neurSom)

    #todo 1ra etapa: ordenamiento global
    for epoca in range(epocas[0]):
        for i in range(data.shape[0]):
            patron = data[i, :]
            neurGanadora = obtenerNeurona(neurSom, patron)
            listVec = obtenerVecinos(neurSom, neurGanadora, radio[0])
            
            # Actualizar cada uno de los vecinos, incluyendo la neurona misma
            for idx in listVec:
                neurSom[idx] += tasaAp[0] * (patron - neurSom[idx])
        
        if(epoca % 4 == 0):
            title = "Ordenamiento global ep " + str(epoca)
            actualizarGrafica(fig, ax, title, neurSom, rectHoriz, rectVert)

    # Graficar la ultima epoca
    title = "Ordenamiento global ep " + str(epocas[0])
    actualizarGrafica(fig, ax, title, neurSom, rectHoriz, rectVert)


    #todo 2da etapa: transicion
    radioEtapa2 = np.linspace(radio[0], radio[1], epocas[1])
    tasaApEtapa2 = np.linspace(tasaAp[0], tasaAp[1], epocas[1])

    for epoca in range(epocas[1]):
        for i in range(data.shape[0]):
            patron = data[i, :]
            neurGanadora = obtenerNeurona(neurSom, patron)
            listVec = obtenerVecinos(neurSom, neurGanadora, radioEtapa2[epoca])
            
            # Actualizar cada uno de los vecinos, incluyendo la neurona misma
            for idx in listVec:
                neurSom[idx] += tasaApEtapa2[epoca] * (patron - neurSom[idx])
        
        if(epoca % 4 == 0):
            title = "Etapa transicion ep " + str(epoca)
            actualizarGrafica(fig, ax, title, neurSom, rectHoriz, rectVert)

    # Graficar la ultima epoca
    title = "Etapa transicion ep " + str(epocas[1])
    actualizarGrafica(fig, ax, title, neurSom, rectHoriz, rectVert)

    #todo 3er etapa: ajuste fino
    for epoca in range(epocas[2]):
        for i in range(data.shape[0]):
            patron = data[i, :]
            neurGanadora = obtenerNeurona(neurSom, patron)
            listVec = obtenerVecinos(neurSom, neurGanadora, radio[1])
            
            # Actualizar cada uno de los vecinos, incluyendo la neurona misma
            for idx in listVec:
                neurSom[idx] += tasaAp[1] * (patron - neurSom[idx])
        
        if(epoca % 4 == 0):
            title = "Ordenamiento global ep " + str(epoca)
            actualizarGrafica(fig, ax, title, neurSom, rectHoriz, rectVert)

    # Graficar la ultima epoca
    title = "Ordenamiento global ep " + str(epocas[0])
    actualizarGrafica(fig, ax, title, neurSom, rectHoriz, rectVert)


    return