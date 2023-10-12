import numpy as np

# Algoritmos para realizar la reproduccion (operadores de variacion) y generar descendencia (hijos)

#* Algoritmo para la cruza
def repCruza(poblacion, idxPadres, codCrom, probCruza):

    cantPadres = idxPadres.shape[0]
    cantIdv = poblacion.shape[0]
    lenCrom = np.sum(codCrom)   

    cantHijos = cantIdv - cantPadres

    # Chequeo de por si tengo numero impar de hijos, sumo uno para que sea par (porque 
    # la cruza se hace entre pares de padres), y despues lo elimino el que esta demas
    if(cantHijos % 2):
        cantHijos += 1

    # matriz de hijos vacia (cada fila es un hijo o individuo y cada columna un gen o bit)
    hijos = np.empty(shape=(cantHijos, lenCrom), dtype=bool)

    i = 0
    while(i < cantHijos):   
        # elijo 2 padres al azar para usarlos en la cruza
        idxs = np.random.choice(idxPadres, size=2, replace=False)
        padre1 = poblacion[idxs[0], :]
        padre2 = poblacion[idxs[1], :]

        # tiro un numero al azar y si es menor a la probabilidad de cruza aplico la cruza
        if(np.random.rand() < (1 - probCruza)):    
            # elijo un punto de corte al azar entre [1, end-1] y concateno la cruza de esos padres
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

    # Actualizo la poblacion combinando los padres originales con los hijos creados
    poblacion = np.concatenate((poblacion[idxPadres], hijos))

    return poblacion[:cantIdv]

#* Algoritmo para la mutacion para cada hijo
def repMutacion(poblacion, probMutacion, codCrom):
    """
        Algoritmo de mutacion para cada hijo.
        Entradas: matriz de poblacion de cromosomas (cada fila es un individuo y cada columna un 
        gen o bit del cromosoma), la probabilidad de mutacion y un arreglo con el numero de bits 
        para codificar cada gen del cromosoma (para determinar la longitud del cromosoma).
        Salida: poblacion luego de haber aplicado la mutacion.
    """

    lenCrom = np.sum(codCrom)   # longitud total del cromosoma

    # Ver el tema de np.copy, porque como trabajan de referencia los datos
    # no se va a notar la diferencia si quiero comparar los datos luego de 
    # aplicar esta funcion. 
    # No es necesario esta linea, principalmente para notar que realmente 
    # genera cambios.
    # poblacion = np.copy(poblacion)

    # Iteramos sobre cada fila de la matriz de poblacion, es decir, sobre cada individuo o cromosoma.
    # Luego tiramos un numero al azar y si es menor a la probabilidad de mutacion se elige un gen (bit)
    # al azar y se invierte su valor (como lo estamos trabajando con valores bool usamos "not")
    for i in range(poblacion.shape[0]):    
        if(np.random.rand() < probMutacion):
            idxMut = np.random.choice(lenCrom)      # 
            poblacion[i, idxMut] = np.logical_not(poblacion[i, idxMut])

    return poblacion

#? Test de los metodos
# np.random.seed(0)

# poblacion = np.random.randint(0, 2, size=(10, 6))
# poblacion = poblacion.astype(bool)

# idxPadres = np.array([2, 3])
# codCrom = np.array([3, 3])

# probCruza = 0.8
# probMutacion = 0.1

# cruza = repCruza(poblacion, idxPadres, codCrom, probCruza)
# mutacion = repMutacion(cruza, probMutacion, codCrom)

# print(f"Poblacion: \n{poblacion.astype(int)}")
# print(f"\nPadres: \n{idxPadres}")
# print(f"\nCruza: \n{cruza.astype(int)}")
# print(f"\nMutacion: \n{mutacion.astype(int)}")