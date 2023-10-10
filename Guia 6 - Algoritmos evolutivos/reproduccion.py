import numpy as np

# Algoritmos para realizar la reproduccion (operadores de variacion) y generar descendencia (hijos)

#* Algoritmo para la cruza
def repCruza(poblacion, idxPadres, codCrom, probCruza):

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

        # tiro un numero al azar y si es menor a la probabilidad hace la cruza
        if(np.random.rand() < (1 - probCruza)):    
            # Elijo un punto de corte entre [1, end-1] y concateno la cruza de esos padres
            corte = np.random.choice(lenCrom-2) + 1
            hijos[i] = np.concatenate((padre1[:corte], padre2[corte:]))
            i += 1
            hijos[i] = np.concatenate((padre2[:corte], padre1[corte:]))
            i += 1

        # si no es mayor a la probabilidad, pasan como estaban
        else:
            hijos[i] = padre1
            hijos[i+1] = padre2
            i += 2

    # Actualizo la poblacion
    poblacion = np.concatenate((poblacion[idxPadres], hijos))

    return poblacion[:cantIdv]

#* Algoritmo para la mutacion de cada hijo
def repMutacion(poblacion, probMutacion, codCrom):
    lenCrom = np.sum(codCrom)

    #? Ver el tema de np.copy, porque como trabajan de referencia los datos
    #? no se va a notar la diferencia si quiero comparar los datos luego de 
    #? aplicar esta funcion. 
    #? No es necesario esta linea, principalmente para notar que realmente 
    #? genera cambios.
    poblacion = np.copy(poblacion)

    for i in range(poblacion.shape[0]):
        if(np.random.rand() < probMutacion):
            idxMut = np.random.choice(lenCrom)
            poblacion[i, idxMut] = np.logical_not(poblacion[i, idxMut])

    return poblacion



#? Test de los metodos
np.random.seed(0)

poblacion = np.random.randint(0, 2, size=(10, 6))
poblacion = poblacion.astype(bool)

idxPadres = np.array([2, 3])
codCrom = np.array([3, 3])

probCruza = 0.8
probMutacion = 0.1

cruza = repCruza(poblacion, idxPadres, codCrom, probCruza)
mutacion = repMutacion(cruza, probMutacion, codCrom)

print(f"Poblacion: \n{poblacion.astype(int)}")
print(f"\nPadres: \n{idxPadres}")
print(f"\nCruza: \n{cruza.astype(int)}")
print(f"\nMutacion: \n{mutacion.astype(int)}")