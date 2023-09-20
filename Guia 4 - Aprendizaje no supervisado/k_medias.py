# En este video lo hace con sklearn (nosotros lo debemos hacer desde cero), pero 
# al final lo grafica usando matplotlib y esta muy buena la forma en que lo hace:
# https://www.youtube.com/watch?v=mICySHB0fh4

import numpy as np
from graficas import iniciarGraficaKM3D, actualizarGraficaKM3D

def k_medias(data, k, numMaxIteraciones=100):
    """
        Funcion que implementa el algoritmo k-means o k-medias.
        Entradas: datos, parametro "k" a utilizar y un numero maximo de iteraciones para
        utilizar en caso de ser necesario.
        Salida: arreglo de centroides y clusteres formados.
    """

    # Inicializo los centroides de manera aleatoria seleccionando k datos aleatorios
    # data.shape[0] -> cantidad de filas
    # ordeno de forma aleatoria los indices con permutation y luego todo los primeros "k"
    # indices [:k] y se toman esos "k" datos de "data" como centroides iniciales
    centroides = data[np.random.permutation(data.shape[0])[:k]]

    #* Graficar 3D
    fig, ax, dataPlot, centPlot = iniciarGraficaKM3D(data, centroides)
    iteracion = 0

    # El _ en los bucles significa que el indice no es relevante, ya que no se utiliza dentro 
    # del bucle, solo queremos iterar una cierta cantidad de veces
    for _ in range(numMaxIteraciones):
        # Inicializo vectores vacios para almacenar los indices de los puntos en cada cluster
        clusters = []
        for _ in range(k):
            clusters.append([])
        
        # Asigno cada punto al cluster mas cercano
        for i in range(len(data)):
            punto = data[i]
            distancias = []

            # Calculo la distancia euclidiana entre el punto y todos los centroides
            for centroide in centroides:
                distancia = np.linalg.norm(punto - centroide)
                distancias.append(distancia)
            
            # Busco el indice del centroide mas cercano al punto
            # np.argmin devuelve el indice del valor minimo en el vector, es decir, el indice
            # del centroide mas cercano al punto
            cluster_ind = np.argmin(distancias)
            
            # Agrego el indice del punto al cluster correspondiente
            clusters[cluster_ind].append(i)
        
        # Calculo los nuevos centroides para cada cluster, como el promedio de los puntos que 
        # tiene cada cluster. Con data[cluster] selecciono los puntos que pertenecen al cluster 
        # actual utilizando sus indices (que estan en "clusters"), y luego calculo la media a 
        # lo largo del eje 0 (columnas)
        new_centroides = []
        for cluster in clusters:
            new_centroide = np.mean(data[cluster], axis=0)
            new_centroides.append(new_centroide)
        
        # Camparo si todos (all) los centroides son iguales a los nuevos centroides calculados
        # Si no se realizan mas reasignacioens cortamos el bucle
        # if np.all(centroides == new_centroides):
        if np.array_equal(centroides, new_centroides):
            # print("No se realizan mas reasignaciones")
            break
        
        # Actualizo los centroides
        centroides = np.array(new_centroides)

        # Actualizar la grafica
        iteracion += 1
        title = "K = " + str(k) + " iteracion " + str(iteracion)
        actualizarGraficaKM3D(ax, title, dataPlot, centPlot, centroides, clusters)
    
    # Devuelvo los centroides y el vector de vectores con los indices de los puntos de datos en cada cluster.
    # return centroides, clusters
    ax.set_title("K = " + str(k) + " finalizado en iteracion " + str(iteracion))
    return np.array(centroides), clusters   # convierte la lista a un array de numpy por si me sirve mas