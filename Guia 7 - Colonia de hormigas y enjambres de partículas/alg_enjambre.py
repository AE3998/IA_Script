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

    # Inicializamos dim, vectores de posicion (de forma aleatoria para permitir que explore
    # diferentes regiones del espacio, y con distr. uniforme), y la velocidades (nulas). 
    dim = xmin.shape[0]     # dimension
    posActualIdv = np.random.uniform(low=xmin, high=xmax, size=(cantIdv, dim))   
    velActualIdv = np.zeros(shape=(cantIdv, dim))

    mejorPosIdv = posActualIdv.copy()    # inicialmente la mejor posicion es la actual
    fitness = -func(mejorPosIdv)        # el signo (-) es para convertir a un problema de maximizacion

    # Queremos encontrar el minimo global, pero como pusimos el signo (-) usamos max o argmax
    # ya que lo transformamos a un problema de maximizacion de fitness
    idxMaxFit = np.argmax(fitness)
    actualMaxFit = fitness[idxMaxFit]
    mejorPosEnjambre = mejorPosIdv[idxMaxFit]

    # Bucle "repetir" general
    contIter = 0
    idxMejor = 0
    n = 0

    # c1 y c2 son dos constantes que nos permiten controlar cuanta importancia le damos 
    # a la experiencia personal de la particula y cuanta a la experiencia del enjambre, 
    # para el calculo de la velocidad. 
    # Hacemos que varien a lo largo de las iteraciones.
    delta_c1 = (c2 - c1)/maxIter
    delta_c2 = -delta_c1

    ax, puntos = iniciarGrafica(func, xmin, xmax, mejorPosIdv)

    while (contIter <= maxIter):
        contIter += 1

        # Recorrer todas las particulas y actualizar aquellos que obtienen mejor posicion
        # func(posActualIdv) es calcular la funcion objetivo en la pos actual de la particula,
        # y la comparamos con la func objetivo en la mejor posicion de la particula (fitness)
        idxBool = -func(posActualIdv) > fitness
        mejorPosIdv[idxBool] = posActualIdv[idxBool]    # actualiza la mejor pos usando indexado booleano

        # Actualizar la lista de fitness
        fitness = -func(mejorPosIdv)
        idxMejor = np.argmax(fitness)

        # Actualizar la mejor posicion global o del enjambre si se cumple.
        # Seria comparar la func objetivo en la mejor pos historica de la particula 
        # contra la func objetivo en la mejor pos global (fitness[idxMejor])
        if(actualMaxFit < fitness[idxMejor]):   
            mejorPosEnjambre = mejorPosIdv[idxMejor]
            actualMaxFit = fitness[idxMejor]
            n = 0
        else:
            n += 1

        # Segundo criterio de corte: 10 generaciones seguidas sin mejora
        if(n >= 15):
            print("Corto por", n, "iteraciones sin mejoras")
            break

        # Vectores que agregan una componente aleatoria al algoritmo para permitir
        # hacer una exploracion mas completa del espacio de busqueda, con mas libertad.
        r1 = np.random.rand(cantIdv, dim)
        r2 = np.random.rand(cantIdv, dim)

        # Actualizar la velocidad actual de las particulas
        # aux1 trata de llevar la particula hacia la direccion donde tuvo su mejor 
        # desempenio, y aux2 trata de llevarla hacia la mejor posicion del enjambre
        aux1 = c1 * r1 * (mejorPosIdv - posActualIdv)   # experiencia personal de la particula
        aux2 = c2 * r2 * (mejorPosEnjambre - posActualIdv)  # experiencia del enjambre
        velActualIdv += aux1 + aux2
        
        # Actualizar posicion actual
        posActualIdv += velActualIdv

        c1 += delta_c1
        c2 += delta_c2

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
    print(f"fitness: {-actualMaxFit}")

    return mejorPosEnjambre