import numpy as np

def desordenarDatos(datos):
    """
    Funcion para desordenar las filas del archivo por si vienen ordenadas por categoria.
    Es para que la red no se aprenda todos los de una categoria y luego los de otra.
    Entrada: datos obtenidos del archivo.
    Salida: datos con filas desordenadas de forma aleatoria.
    """
    # Obtener un indice aleatorio para permutar las filas
    indices_aleatorios = np.random.permutation(datos.shape[0])  # matriz.shape[0] = cantidad de filas
    # Ordenar las filas de la matriz de forma aleatoria
    return datos[indices_aleatorios]