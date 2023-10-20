import numpy as np

def cargarDatos(nombreArchivo):
    """
    Funcion para cargar los datos a partir de un archivo csv.
    Se agrega una entrada -1 al comienzo y se separan las entradas de la salida deseada.
    Se usa como (X, yd) = cargarDatos(datos)
    Entrada: nombre del archivo de datos.
    Salida: entradas y salidas deseadas, por separado.
    """

    # Se usa la funcion genfromtxt de Numpy para cargar archivos csv
    datos = np.genfromtxt(nombreArchivo, delimiter=',')

    # --- Desordeno las filas del archivo por si venian ordenadas por categoria ---
    # Es para que la red (el perceptron) no se aprenda todos los de una categoria y luego los de otra.
    # Obtener un indice aleatorio para permutar las filas
    indices_aleatorios = np.random.permutation(datos.shape[0])  # matriz.shape[0] = cantidad de filas
    # Ordenar las filas de la matriz de forma aleatoria
    data = datos[indices_aleatorios]

    # --- Separo las entradas y las salidas deseadas ---
    X = data[:, :-1]    # todas las filas, todas las columnas excluyendo la ultima
    n, m = X.shape      # dimension de la matriz
    X0 = -1 * np.ones((n, 1))   # vector de -1 para agregar al comienzo
    Xnew = np.hstack((X0, X))   # concatena de forma horizontal los arreglos
    yd = data[:, -1]   # salidas deseadas (todas las filas de la ultima columna)     

    # devuelve un vector de entradas X[x0, x1, ..., xn] con x0 = -1 y las salidas deseadas yd por separado
    return (Xnew, yd)    