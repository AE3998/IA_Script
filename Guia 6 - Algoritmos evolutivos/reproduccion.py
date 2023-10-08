import numpy as np

# Algoritmos para realizar la reproduccion (operadores de variacion) y generar descendencia (hijos)

#* Algoritmo para la cruza
def cruza(poblacion, idxPadres, codCrom, probCruza):

    cantPadres = idxPadres.shape[0]
    cantIdv = poblacion.shape[0]
    lenCrom = np.sum(codCrom)

    cantHijos = cantIdv - cantPadres

    # Chequeo de por si tengo numero impar de hijos, le paso a 
    # ser par y despues lo elimino el que esta demas
    if(cantHijos % 2):
        cantHijos += 1

    hijos = np.empty(shape=(cantHijos, lenCrom), dtype=bool)

    i = 0
    while(i < cantHijos):
        # Elijo 2 padres 
        idxs = np.random.choice(idxPadres, size=2, replace=False)
        padre1 = poblacion[idxs[0], :]
        padre2 = poblacion[idxs[1], :]

        if(np.random.rand() < probCruza):
            # Elijo un punto de corte y concateno la cruza de esos padres
            corte = np.random.choice(lenCrom)
            hijos[i] = np.concatenate((padre1[:corte], padre2[corte:]))
            i += 1
            hijos[i] = np.concatenate((padre2[:corte], padre1[corte:]))
            i += 1

        else:
            hijos[i] = padre1
            hijos[i+1] = padre2
            i += 2

    # Actualizo la poblacion
    poblacion = np.concatenate((poblacion[idxPadres], hijos))

    return poblacion[:cantIdv]

#* Algoritmo para la mutacion
def mutacion(poblacion, probMutacion, codCrom):
    lenCrom = np.sum(codCrom)

    for i in range(poblacion.shape[0]):
        if(np.random.rand() < probMutacion):
            idxMut = np.random.choice(lenCrom)
            poblacion[i, idxMut] = np.logical_not(poblacion[i, idxMut])

    return poblacion