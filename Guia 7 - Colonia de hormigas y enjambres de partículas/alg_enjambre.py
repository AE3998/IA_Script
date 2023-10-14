import numpy as np

#* Algoritmo de optimizacion por enjambre de particulas
#* Utilizamos el algoritmo de enjambre del mejor global (gEP)

def enjambre_mejor_global(func, cantParticulas, dim, maxIter, c1, c2, rango):
    
    # Variables a utilizar
    posActualParticula = np.zeros((cantParticulas, dim))
    velActualParticula = np.zeros((cantParticulas, dim))
    rango = np.array(rango)     # convierto la lista "rango" en un array de numpy

    # Inicializacion de particulas en el espacio
    for i in range(cantParticulas):
        for j in range(len(rango)):
            posActualParticula[i][j] = rango[j][0] + np.random.rand()*(rango[j][1]-rango[j][0])

    mejorPosInd = posActualParticula    # inicialmente la mejor posicion es la actual
    mejorPosEnjambre = mejorPosInd[0].copy()    # inicialmente se define como la pos de la primera
    actualMaxFitness = 0

    if(dim == 1):
        fitness = func(mejorPosEnjambre)
    else:
        fitness = func(mejorPosEnjambre[0], mejorPosEnjambre[1])

    # Estos "delta" son para hacer que las constantes c1 y c2 varien a lo largo de las iteraciones,
    # como explicamos en las anotaciones de practica.
    # Se hace que c1 sea grande al comienzo y vaya disminuyendo (tiene su explicacion), y que c2
    # sea chica al comienzo y vaya creciendo (tiene su explicacion en las anotaciones)
    delta_c1 = np.linspace(c1[0], c1[1], maxIter)
    delta_c2 = np.linspace(c2[0], c2[1], maxIter)

    # Bucle "repetir" general
    contIter = 0
    n = 0
    while (contIter <= maxIter):
        contIter += 1

        # Recorremos todas las particulas
        for i in range(cantParticulas):
        
            if(dim == 1):
                # Comprobar si la posicion actual es la mejor historica
                if (func(posActualParticula[i]) < func(mejorPosInd[i])):
                    mejorPosInd[i] = posActualParticula[i]

                # Comprobar si la posicion actual es la mejor historica de la bandada
                if (func(mejorPosInd[i]) < func(mejorPosEnjambre)):
                    mejorPosEnjambre = mejorPosInd[i]
                    fitness = func(mejorPosEnjambre)
            else:
                # Comprobar si la posicion actual es la mejor historica
                if (func(posActualParticula[i][0], posActualParticula[i][1]) < func(mejorPosInd[i][0], mejorPosInd[i][1])):
                    mejorPosInd[i] = posActualParticula[i]

                # Comprobar si la posicion actual es la mejor historica de la bandada
                if (func(mejorPosInd[i][0], mejorPosInd[i][1]) < func(mejorPosEnjambre[0], mejorPosEnjambre[1])):
                    mejorPosEnjambre = mejorPosInd[i]
                    fitness = func(mejorPosEnjambre[0], mejorPosEnjambre[1])

        # Si el mejor fitness no mejora durante "n" iteraciones seguidas, se corta
        # el algoritmo porque suponemos que convergio
        if(actualMaxFitness >= fitness):
            n += 1
        else:
            actualMaxFitness = fitness
            n = 0

        # Segundo criterio de corte: 10 generaciones seguidas sin mejora
        #! Hay algo raro porque siempre me dice que corta este criterio y la cant de iteraciones
        #! solo hace 10, como que nunca mejora el fitness
        if(n >= 10):
            print("Cortó por", n, "iteraciones sin mejoras")
            break
        
        # Generar vectores aleatorios r1 y r2
        # Estos vectores "r" toman valores aleatorios con distribucion uniforme entre 0 y 1, si es
        # cercano a 1 mantiene la direccion original y si es ser cercano a 0 la va a modificar mas 
        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)

        # Recorremos las particulas para actualizar la vel y pos actual
        for i in range(cantParticulas):
            for j in range(dim):
                # Actualizar velocidad actual
                aux1 = delta_c1[contIter]*(r1[j])*(mejorPosInd[i][j]-posActualParticula[i][j])
                aux2 = delta_c2[contIter]*(r2[j])*(mejorPosEnjambre[j]-posActualParticula[i][j])
                velActualParticula[i][j] = velActualParticula[i][j] + aux1 + aux2
                # Actualizar posicion actual
                posActualParticula[i][j] = posActualParticula[i][j] + velActualParticula[i][j]

                # Nos aseguramos de que la posicion este en el rango utilizando la funcion "clip" de Numpy.
                # Si la pos esta fuera del rango, le asigna el valor del limite inferior o superior 
                # del rango, segun el mas cercano
                posActualParticula = np.clip(posActualParticula, rango[j, 0], rango[j, 1])

    #! HAY QUE COMPARAR LA VELOCIDAD CON EL DEL EJERCICIO 1 GUIA 6 PORQUE SE PIDE EN EL ENUNCIADO
    print("Cantidad de iteraciones:", contIter)
    print("Minimo encontrado:", mejorPosEnjambre)
    return mejorPosEnjambre