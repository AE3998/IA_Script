import numpy as np

def cargarDatos(nombreArchivo):
    data = np.genfromtxt(nombreArchivo, delimiter=",", max_rows=None)
    return data