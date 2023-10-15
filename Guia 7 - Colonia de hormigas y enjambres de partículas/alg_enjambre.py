import numpy as np
from graficas import *

#* Algoritmo de optimizacion por enjambre de particulas
#* Utilizamos el algoritmo de enjambre del mejor global (gEP)

def enjambre_mejor_global(func, cantIdv, maxIter, c1, c2, xmin, xmax):
    
    # Chequear si viene un numero, lo pasa a un arreglo
    if(not(isinstance(xmin, list))):
        xmin = np.array([xmin])
    else:
        xmin = np.array(xmin)
    if(not(isinstance(xmax, list))):
        xmax = np.array([xmax])
    else:
        xmax = np.array(xmax)

    # Variables a utilizar
    dim = xmin.shape[0]     # dimension
    posActualIdv = np.random.uniform(low=xmin, high=xmax, size=(cantIdv, dim))
    velActualIdv = np.zeros(shape=(cantIdv, dim))

    mejorPosIdv = posActualIdv.copy()    # inicialmente la mejor posicion es la actual
    fitness = func(mejorPosIdv)

    #* queremos encontrar el minimo global, asi que usamos min o argmin
    idxMaxFit = np.argmin(fitness)
    actualMaxFit = fitness[idxMaxFit]
    mejorPosEnjambre = mejorPosIdv[idxMaxFit]

    # Bucle "repetir" general
    contIter = 0
    idxMejor = 0
    n = 0

    delta_c1 = (c2 - c1)/maxIter
    delta_c2 = -delta_c1

    ax, puntos = iniciarGrafica(func, xmin, xmax, mejorPosIdv)

    while (contIter <= maxIter):
        contIter += 1

        # Recorrer todas las particulas y actualizar aquellos 
        # que logran obtener mejor posicion
        idxBool = func(posActualIdv) < fitness
        mejorPosIdv[idxBool] = posActualIdv[idxBool]

        # Actualizar la lista de fitness
        fitness = func(mejorPosIdv)
        idxMejor = np.argmin(fitness)

        # Actualizar el mejor individuo
        if(actualMaxFit > fitness[idxMejor]):
            mejorPosEnjambre = mejorPosIdv[idxMejor]
            actualMaxFit = fitness[idxMejor]
            n = 0
        else:
            n += 1

        # Segundo criterio de corte: 10 generaciones seguidas sin mejora
        if(n >= 18):
            print("Corto por", n, "iteraciones sin mejoras")
            break

        r1 = np.random.rand(cantIdv, dim)
        r2 = np.random.rand(cantIdv, dim)

        # Recorremos las particulas para actualizar la velocidad actual
        aux1 = c1 * r1 * (mejorPosIdv - posActualIdv)
        aux2 = c2 * r2 * (mejorPosEnjambre - posActualIdv)
        velActualIdv += aux1 + aux2
        
        c1 += delta_c1
        c2 += delta_c2

        # Actualizar posicion actual
        posActualIdv += velActualIdv

        # Nos aseguramos de que la posicion este en el rango utilizando la funcion "clip" de Numpy.
        # Si la pos esta fuera del rango, le asigna el valor del limite inferior o superior 
        # del rango, segun el mas cercano
        posActualIdv = np.clip(posActualIdv, a_min=xmin, a_max=xmax)

        cambios = np.sum(idxBool)
        # print(f"Cambios: {cambios}")
        title = f"Iteracion {str(contIter)}, {cambios} mejora. "
        puntos = actualizarGrafica(func, dim, posActualIdv, ax, puntos, title )

    print("Cantidad de iteraciones:", contIter)
    print("Minimo encontrado:", mejorPosEnjambre)
    print(f"fitness: {actualMaxFit}")

    return mejorPosEnjambre