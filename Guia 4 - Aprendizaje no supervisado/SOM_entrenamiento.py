import numpy as np
from graficas import *
# from cargarDatos import *

def cargarDatos(nombreArchivo):
    #? Hacer un test de tamano reducido, por es o max_rows = 10 
    data = np.genfromtxt(nombreArchivo, delimiter=",", max_rows=None)
    return data

def obtenerNeurona(nSOM, patron):
    # Aumentar la dimension
    p_aux = patron[np.newaxis, np.newaxis, :]

    # Despejar la distancia del patron con todas las neuronas
    dist = np.linalg.norm(nSOM - p_aux, axis=2) 

    #? Print
    # print(f"Distancias a las neuronas = \n {dist}")

    # Despejar el indice de la neurona con menor distancia
    winNeur = np.unravel_index(np.argmin(dist), shape=dist.shape)

    return winNeur

def obtenerVecinos(Z, winIdx, radio):

    # Aumentar dimension al index de la neurona ganadora 
    # idx_aux = np.array((winIdx[1], winIdx[0]))[:, np.newaxis, np.newaxis]
    idx_aux = np.array(winIdx)[:, np.newaxis, np.newaxis]

    # Determinar las distancias de cada vecino de manera vectorial
    dist = np.linalg.norm(Z - idx_aux, axis=0, ord=1)

    # ? mostrar Z
    # print(f"Z = \n{Z}") 
    # print(f"idx_aux = \n{idx_aux}")
    # print(f"dist = \n{dist}")

    # Retornar la matriz booleano, sera True aquellos que estan 
    # dentro del radio
    return (dist <= radio)

def SOM_entrenamiento(data, epocas, dimSom, tasaAp, radio, iris=False):
    """
        Funcion para entrenamiento de un SOM.
        Entradas: nombre del archivo csv de datos, vector epocas donde se tenga las epocas para
        las tres etapas de entrenamiento, tamanio del SOM, tasa de aprendizaje para la epoca 1 y 3,
        y radio de vecindad para epoca 1 y 3.
    """
    # # ! Seed para test
    # np.random.seed(10000)

    # Definir los indices de la matriz (i, j)
    ii = np.arange(dimSom[0]) # quiero 3 fila, 3 en y
    jj = np.arange(dimSom[1]) # quiero 4 columna, 4 en x

    Z = np.array(np.meshgrid(jj, ii)) # 3 fila y 4 columna, (x=4, y=3)
    Z = np.array((Z[1], Z[0]))#! Por como esta definido meshgrid, hay que invertir el orden

    # Inicializar pesos al azar
    neurSom = np.random.rand(dimSom[0], dimSom[1], data.shape[1]) - 0.5

    fig, ax, rectHoriz, rectVert = iniciarGraficaSOM(data, neurSom, iris)

    #todo 1ra etapa: ordenamiento global
    tasaApEtapa1 = tasaAp[0]

    for epoca in range(epocas[0]):

        #? Por legibilidad no mas
        # print(f"\n\nepoca = {epoca}")
        # neurSom = np.round(neurSom, 2)

        for i in range(data.shape[0]):
            #? otro print
            # print(f"\n\n index = {i}")
            patron = data[i, :]
            neurGanadora = obtenerNeurona(neurSom, patron)
            idxVecBool = obtenerVecinos(Z, neurGanadora, radio[0])

            #? Print para verificar las cosas
            # print(f"neurSom = \n{neurSom}")
            # print(f"patron = \n{patron}")
            # print(f"neurGanadora = \n{neurGanadora}")
            # print(f"idxVecBool = \n{idxVecBool}")

            # Actualiza los pesos de las neuronas incluyendo uno mismo
            neurSom[idxVecBool] += tasaApEtapa1 * (patron - neurSom[idxVecBool])
        
        if(epoca % 4 == 0 or epoca == epocas[0]-1):
            title = "Ordenamiento global ep " + str(epoca)
            actualizarGraficaSOM(fig, ax, title, neurSom, rectHoriz, rectVert)
            
            #? Otro print porque me parece raro algo que no cambia el patron
            # print(epoca)
            # print(neurSom[idxVecBool])
            # print(patron)

    #todo 2da etapa: transicion
    radioEtapa2 = np.linspace(radio[0], radio[1], epocas[1])
    tasaApEtapa2 = np.linspace(tasaAp[0], tasaAp[1], epocas[1])

    for epoca in range(epocas[1]):
        for i in range(data.shape[0]):

            patron = data[i, :]
            neurGanadora = obtenerNeurona(neurSom, patron)
            idxVecBool = obtenerVecinos(Z, neurGanadora, radioEtapa2[epoca])
            neurSom[idxVecBool] += tasaApEtapa2[epoca] * (patron - neurSom[idxVecBool])
        
        if(epoca % 4 == 0 or epoca == epocas[1]-1):
            title = "Etapa transicion ep " + str(epoca)
            actualizarGraficaSOM(fig, ax, title, neurSom, rectHoriz, rectVert)

    #todo 3er etapa: ajuste fino
    tasaApEtapa3 = tasaAp[1]

    for epoca in range(epocas[2]):
        for i in range(data.shape[0]):

            patron = data[i, :]
            neurGanadora = obtenerNeurona(neurSom, patron)
            idxVecBool = obtenerVecinos(Z, neurGanadora, radio[1])

            neurSom[idxVecBool] += tasaApEtapa3 * (patron - neurSom[idxVecBool])

        if(epoca % 4 == 0 or epoca == epocas[2]-1):
            title = "Ajuste fino ep " + str(epoca)
            actualizarGraficaSOM(fig, ax, title, neurSom, rectHoriz, rectVert)
            
            #? Otro print
            # print(f"neurSom = \n{neurSom}")
            # print(f"patron = \n{patron}")
            # print(f"neurGanadora = \n{neurGanadora}")
            # print(f"idxVecBool = \n{idxVecBool}")

    #todo Obtener el grupo de clusters
    # Iniciar la lista de clusters
    clusters = []
    for _ in range(dimSom[0] * dimSom[1]):
        clusters.append([])

    # Dimension de la neurona
    dim = (dimSom[0], dimSom[1])

    # Recorrer cada patron y asignarle a cluster mas cercano
    for i in range(data.shape[0]):
        patron = data[i, :]
        neurGanadora = obtenerNeurona(neurSom, patron)
        clustIdx = np.ravel_multi_index(neurGanadora, dims=dim)
        clusters[clustIdx].append(i)

    return neurSom, clusters


# nombreArchivo = "circulo.csv"
# epocas = [200, 200, 150]
# dimSom = [5, 5] # en (i, j)
# tasaAp = [0.25, 0.1]
# radio = [2, 0.1]

# SOM_entrenamiento(nombreArchivo, epocas, dimSom, tasaAp, radio)
# plt.show()