import numpy as np
# from cargarDatos import *


def cargarDatos(nombreArchivo):
    #! Hacer un test de tamano reducido, por es o max_rows = 10 
    data = np.genfromtxt(nombreArchivo, delimiter=",", max_rows=2)
    return data

def obtenerNeurona(nSOM, patron):
    # Aumentar la dimension
    p_aux = patron[np.newaxis, np.newaxis, :]

    # Despejar la distancia del patron con todas las neuronas
    dist = np.linalg.norm(nSOM - p_aux, axis=2) #?.T Ver

# Todo
    #? Por disminuir la dimension, todo se ve trasnpuesta, por eso esta el T para visualizar mejor
    print(f"Distancias a las neuronas = \n {dist.T}")

    # Despejar el indice de la neurona con menor distancia
    winNeur = np.unravel_index(np.argmin(dist), shape=dist.shape)

    return winNeur

def obtenerVecinos(Z, winIdx, radio):

    # Aumentar dimension al index de la neurona ganadora 
    idx_aux = np.array(winIdx)[:, np.newaxis, np.newaxis]
    
    # Determinar las distancias de cada vecino de manera vectorial
    dist = np.linalg.norm(Z - idx_aux, axis=0, ord=1)

    # Retornar la matriz booleano, sera True aquellos que estan 
    # dentro del radio
    # ? Hay un transpose por el problema de disminuir la dimension
    return np.transpose(dist <= radio)

def SOM_entrenamiento(nombreArchivo, epocas, dimSom, tasaAp, radio):
    """
        Funcion para entrenamiento de un Som.
        Entradas: nombre del archivo csv de datos, vector epocas donde se tenga las epocas para
        las tres etapas de entrenamiento,... 
    """

    #! Dejar fijo para un test posterior, son para test
    np.random.seed(10000)

    #! Tener suma cuidado que dimSom[0] hace referencia a la dimension i, es vertical
    #! mientras que dimSom[1] es horizontal que es j Hay que ponernos de acuerdo
    x = np.arange(dimSom[0])
    y = np.arange(dimSom[1])

    Z = np.array(np.meshgrid(x, y))

    # Cargar datos
    data = cargarDatos(nombreArchivo)

    # Inicializar pesos al azar
    neurSom = np.random.rand(dimSom[0], dimSom[1], data.shape[1])


    # 1ra etapa: ordenamiento global
    for epoca in range(epocas[0]):

        #? Por legibilidad no mas
        print(f"\n\nepoca = {epoca}")

        neurSom = np.round(neurSom, 2)

        for i in range(data.shape[0]):
            patron = data[i, :]
            neurGanadora = obtenerNeurona(neurSom, patron)
            idxVecBool = obtenerVecinos(Z, neurGanadora, radio[0])

            #? Pongo los print para verificar las cosas
            print(f"\n\n index = {i}")
            print(f"neurSom = \n{neurSom}")
            print(f"patron = \n{patron}")
            print(f"neurGanadora = !Guarda por el problema de dimension, esta transpuesta! \n{neurGanadora}")
            print(f"idxVecBool = \n{idxVecBool.T}")

            # Actualiza los pesos de las neuronas incluyendo uno mismo
            neurSom[idxVecBool] += tasaAp[0] * (patron - neurSom[idxVecBool])



    # 2da etapa: transicion
    
    # 3er etapa: ajuste fino

    return


nombreArchivo = "circulo.csv"
epocas = [2, 2, 2]
dimSom = [3, 4] # en (i, j)
tasaAp = [0.5, 0.01]
radio = [2, 1]

SOM_entrenamiento(nombreArchivo, epocas, dimSom, tasaAp, radio)