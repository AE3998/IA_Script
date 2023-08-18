import numpy as np
import matplotlib.pyplot as plt
from graficar import *



def cargarDatos(archivo):
    """
    Rutina para cargar los datos de entrenamiento. 
    Entrada: nombre de path del archivo 
    Salida: 
        xn = patron de entrada 
        yd = salida esperada 
    """
    # Cargar el archivo. 
    # delimiter = separadro
    # max_rows = fila maxima a extraer, conviene para test pequeno 
    datos = np.genfromtxt(archivo, delimiter=",", max_rows=None)
    
    # Cargar los patrones de entrada, de todas las filas
    # excluir la ultima columna y agregar una columna 
    # al inicio que corresponde a la entrada x0 = -1.  

    # [0.9, -1.1, 1] => xn = [[-1], 0.9, -1.1] yd = [1]
    x0 = -1*np.ones(shape=(np.size(datos, 0), 1))
    xn = np.hstack((x0, datos[:, :-1]))
    
    # yd.shape = (10,) = 1D
    yd = datos[:, -1]

    return xn, yd


def perceptron(Wi, Xi):
    # Calcular la sumatoria de WiXi
    # Wi = [0.1, 0.2, 0.3], Xi = [-1, 0.9, -1.1]
    # Vi = sum([0.1 * -1, 0.2 * 0.9, 0.3 * -1.1]) = -0.25
    # Vi = np.sum(Wi * Xi)

    # Aplicar la funcion de activacion sgn
    # Y = sgn(Vi)
    # Y da 1 cuando Vi >= 0 y -1 cuando Vi < 0
    
    # Todo se puede reducir en una sola linea 
    return -1 if np.sum(Wi * Xi) < 0 else 1

#
# Main del entrenamiento 
def entrenarPesos(nombreArchivo, velApren, tasaErr, maxEpoca, graficar = False):

    """
    Rutina que entrena los pesos sinapticos Wi
    Entrada:
        nombreArchivo: Ruta hacia donde se alojan los datos.
        velApren: Tasa de aprendizaje
        tasaErr: Tasa de error entre [0 - 1] (0% ~ 100%)
        maxEpoca: Numero maximo de epoca

    Salida:
        Wi = Pesos sinapticos
    """
    # Cargar los datos 
    Xi, Yd = cargarDatos(nombreArchivo)

    # Inicializar los pesos Wi en el rango [-0.5, 0.5]
    # rand(a) retorna una vector de tamano (a)
    # con los componentes aleatorios en el rango [0, 1)
    Wi = np.random.rand(np.size(Xi, 1)) - 0.5
    
    # Inicializar los contadores. Tasa de error = 100% y epoca = 0.
    tasaErrActual = 1
    maxEpocaCont = 0
    nPatron = np.size(Yd, 0)

    if(graficar):
        fig, ax = iniciarGrafica(Xi)
        recta = iniciarRecta(Wi, ax)

    while(maxEpocaCont < maxEpoca and tasaErrActual > tasaErr):
        maxEpocaCont += 1

        # Recorrer cada patron e ir ajustando los pesos
        for i in range(nPatron):
            Y = perceptron(Wi, Xi[i, :])
            if (Y != Yd[i]):
                # Ajustar los pesos
                Wi = Wi + velApren/2 * (Yd[i] - Y) * Xi[i, :]
                if(graficar):
                    actualizarRecta(Wi, recta)
        
        # Medir la tasa de error, recorrer de nuevo 
        # cada patron de entrada
        contError = 0
        for i in range(nPatron):
            Y = perceptron(Wi, Xi[i, :])
            contError += Y != Yd[i]

        # print(contError, "/", nPatron)
        # Determinar la tasa de error
        tasaErrActual = contError/nPatron
    
    if(tasaErrActual < tasaErr):
        print("La rutina ha logrado minimizar la tasa de error:", tasaErrActual * 100, "%")
    else:
        print("La rutina llego la cantidad maxima de Epoca definida.")

    plt.show()

    return Wi


    
