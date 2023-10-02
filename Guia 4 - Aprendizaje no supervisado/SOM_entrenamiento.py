import numpy as np
from graficas import *

def cargarDatos(nombreArchivo):
    # Hacer un test de tamano reducido, por es o max_rows = 10 
    data = np.genfromtxt(nombreArchivo, delimiter=",", max_rows=None)
    return data

def obtenerNeuronaGanadora(nSOM, patron):
    """
        Funcion para obtener la neurona ganadora en el SOM.
        Entrada: pesos de las neuronas que forman el SOM y un patron de entrada.
        Salida: neurona ganadora.
    """
    # Aumentar la dimension
    p_aux = patron[np.newaxis, np.newaxis, :]

    # Despejar la distancia entre el patron que entra y todas las neuronas
    dist = np.linalg.norm(nSOM - p_aux, axis=2) 
    # print(f"Distancias a las neuronas = \n {dist}")

    # Despejar el indice de la neurona con menor distancia
    neuronaGanadora = np.unravel_index(np.argmin(dist), shape=dist.shape)

    return neuronaGanadora

def obtenerVecinos(Z, indNeuGanadora, radio):
    """
        Funcion para obtener las neuronas vecinas de la ganadora en el SOM.
        Recibe la grilla (Z) del SOM, el indice de la neurona ganadora y el radio de vecindad.
    """
    # Aumentar dimension al index de la neurona ganadora 
    # idx_aux = np.array((indNeuGanadora[1], indNeuGanadora[0]))[:, np.newaxis, np.newaxis]
    idx_aux = np.array(indNeuGanadora)[:, np.newaxis, np.newaxis]

    # Determinar las distancias de cada vecino de manera vectorial
    dist = np.linalg.norm(Z - idx_aux, axis=0, ord=1)

    # print(f"Z = \n{Z}") 
    # print(f"idx_aux = \n{idx_aux}")
    # print(f"dist = \n{dist}")

    # Retornar la matriz booleano, sera True para aquellos que estan dentro del radio de vecindad
    return (dist <= radio)

def SOM_entrenamiento(data, epocas, dimSom, tasaAp, radio, iris=False):
    """
        Funcion para entrenamiento de un SOM.
        Entradas: nombre del archivo csv de datos, vector epocas donde se tenga las epocas para
        las tres etapas de entrenamiento, tamanio del SOM, tasa de aprendizaje para la epoca 1 y 3,
        y radio de vecindad para epoca 1 y 3.
    """

    # definir los indices de la matriz (i, j)
    ii = np.arange(dimSom[0]) # quiero 3 fila, 3 en y
    jj = np.arange(dimSom[1]) # quiero 4 columna, 4 en x

    # armamos la grilla
    Z = np.array(np.meshgrid(jj, ii)) # 3 fila y 4 columna, (x=4, y=3)
    Z = np.array((Z[1], Z[0]))  # por como esta definido meshgrid, hay que invertir el orden

    # Inicializar pesos al azar
    neuronasSOM = np.random.rand(dimSom[0], dimSom[1], data.shape[1]) - 0.5

    fig, ax, rectHoriz, rectVert = iniciarGraficaSOM(data, neuronasSOM, iris)

    #* 1ra etapa: ordenamiento global
    tasaApEtapa1 = tasaAp[0]

    for epoca in range(epocas[0]):
        # print(f"\n\nepoca = {epoca}")
        # neuronasSOM = np.round(neuronasSOM, 2)

        for i in range(data.shape[0]):
            # print(f"\n\n index = {i}")
            patron = data[i, :]
            neurGanadora = obtenerNeuronaGanadora(neuronasSOM, patron)
            idxVecBool = obtenerVecinos(Z, neurGanadora, radio[0])

            # print(f"neuronasSOM = \n{neuronasSOM}")
            # print(f"patron = \n{patron}")
            # print(f"neurGanadora = \n{neurGanadora}")
            # print(f"idxVecBool = \n{idxVecBool}")

            # Actualiza los pesos de la neurona ganadora y sus neuronas vecinas
            neuronasSOM[idxVecBool] += tasaApEtapa1 * (patron - neuronasSOM[idxVecBool])
        
        # graficamos cada cuatro epocas y al final
        if(epoca % 4 == 0 or epoca == epocas[0]-1):
            title = "Ordenamiento global ep " + str(epoca)
            actualizarGraficaSOM(fig, ax, title, neuronasSOM, rectHoriz, rectVert)

    #* 2da etapa: transicion
    radioEtapa2 = np.linspace(radio[0], radio[1], epocas[1])
    tasaApEtapa2 = np.linspace(tasaAp[0], tasaAp[1], epocas[1])

    for epoca in range(epocas[1]):
        for i in range(data.shape[0]):

            patron = data[i, :]
            neurGanadora = obtenerNeuronaGanadora(neuronasSOM, patron)
            idxVecBool = obtenerVecinos(Z, neurGanadora, radioEtapa2[epoca])
            # Actualiza los pesos de la neurona ganadora y sus neuronas vecinas
            neuronasSOM[idxVecBool] += tasaApEtapa2[epoca] * (patron - neuronasSOM[idxVecBool])
        
        if(epoca % 4 == 0 or epoca == epocas[1]-1):
            title = "Etapa transicion ep " + str(epoca)
            actualizarGraficaSOM(fig, ax, title, neuronasSOM, rectHoriz, rectVert)

    #* 3er etapa: ajuste fino
    tasaApEtapa3 = tasaAp[1]

    for epoca in range(epocas[2]):
        for i in range(data.shape[0]):

            patron = data[i, :]
            neurGanadora = obtenerNeuronaGanadora(neuronasSOM, patron)
            idxVecBool = obtenerVecinos(Z, neurGanadora, radio[1])
            # Actualiza los pesos de la neurona ganadora y sus neuronas vecinas
            neuronasSOM[idxVecBool] += tasaApEtapa3 * (patron - neuronasSOM[idxVecBool])

        if(epoca % 4 == 0 or epoca == epocas[2]-1):
            title = "Ajuste fino ep " + str(epoca)
            actualizarGraficaSOM(fig, ax, title, neuronasSOM, rectHoriz, rectVert)

    #* Obtener el grupo de clusters
    # Iniciar la lista de clusters
    clusters = []
    for _ in range(dimSom[0] * dimSom[1]):
        clusters.append([])

    # Dimension de la neurona
    dim = (dimSom[0], dimSom[1])

    # Recorrer cada patron y asignarlo al cluster mas cercano
    for i in range(data.shape[0]):
        patron = data[i, :]
        neurGanadora = obtenerNeuronaGanadora(neuronasSOM, patron)
        clustIdx = np.ravel_multi_index(neurGanadora, dims=dim)
        clusters[clustIdx].append(i)

    return neuronasSOM, clusters