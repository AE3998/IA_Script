import numpy as np

# Para implementar operadores de seleccion que utilicemos, ya sea ruleta, ventana, competencia.
# Siempre recibimos un arreglo de fitness con los valores de aptitud de cada individuo de la
# poblacion, y la cantidad de individuos que se deben seleccionar como padres para la siguiente
# generacion.
# Y retornamos los indices de los individuos seleccionados como padres.

def selectRuleta(fitness, cantPadres):
    # Se debe normalizar los fitness para la probabilidad sumen 1
    sum = np.sum(fitness)
    normFit = fitness/sum
    
    # arreglo de indices con longitud del numero de individuos de la poblacion (fitness.shape[0])
    idxFit = np.arange(fitness.shape[0])

    # p es el parametro de probabilidad, debe ser un vector 1D que sus componentes sumen 1.
    # Con np.random.choice hacemos la seleccion por ruleta.
    # Elige cantPadres indices de manera aleatoria del arreglo idxFit con probabilidades dadas
    # por el arreglo normalizado normFit. Los individuos con una aptitud mas alta tendran
    # una mayor probabilidad de ser seleccionados, pero no esta garantizado que sean seleccionados.
    idxPadres = np.random.choice(idxFit, size=cantPadres, p=normFit)

    return idxPadres

def selectVentana(fitness, cantPadres):
    # La funcion admite repeticion porque al achicar la ventana van quedando los mejores
    
    ordenFit = np.argsort(-fitness)     # indices de invididuos ordenados de mayor a menor fitness
    idxPadres = np.empty(shape=(cantPadres), dtype=int) # arreglo vacio para guardar los indices de los padres seleccionados

    cantInd = ordenFit.shape[0]     # cant de individuos de la poblacion
    paso =  cantInd // cantPadres   # con el paso va reduciendo o desplazando la ventana

    for i in range(cantPadres):     # iteramos sobre los padres que vamos a seleccionar
        idxPadres[i] = np.random.choice(ordenFit[:cantInd - (paso*i)])
        # con np.random.choice elegimos un indice aleatorio dentro de la ventana, donde la 
        # ventana es una porcion del arreglo ordenFit que se va reduciendo de tamanio en 
        # cada iteracion del bucle for
        # print(ordenFit[:cantInd - (paso*i)])

    return idxPadres
 
def selectCompetencia(fitness, cantPadres):
    # Se eligen “n” individuos al azar y luego de esos “n” nos quedamos con el de mejor fitness.
    # Mientras mas grande sea ese parametro, mas chances de elegir a los mejores individuos.

    # arreglo de indices con longitud del numero de individuos de la poblacion (fitness.shape[0])
    idxFit = np.arange(fitness.shape[0])
    # matriz (arreglo) con la forma dada en shape y valores True porque inicialmente todos 
    # los individuos se consideran como "competidores"
    idxBool = np.full(shape=(fitness.shape[0]), fill_value=True)    

    # calculamos la cantidad de competidores para cada iteracion
    cantCompetencia = fitness.shape[0] // cantPadres

    idxPadres = []  # lista vacia para ir guardando los indices de los individuos seleccionados como padres

    for _ in range(cantPadres):
        # se eligen cantCompetencia indices al azar de la poblacion de competidores (idxFit con True)
        # y se asume que no admiete repeticion (replace=False)
        idxSelec = np.random.choice(idxFit[idxBool], size=cantCompetencia, replace=False)
        idxBool[idxSelec] = False   # se marcan como False los ind seleccionados para que no se vuelvan a elegir
        
        # indice del individuo con mejor fitness entre los seleccionados para competir
        idxMaxFit = np.argmax(fitness[idxSelec])    
        # agregamos ese indice a la lista de padres
        idxPadres.append(idxSelec[idxMaxFit])   
    return np.array(idxPadres)

# -------------------------------------------------------

#? test seleccion
# fitness = np.random.randint(0, 7, size=(25))
# cantPadres = 8
# print(fitness)

# idxPadres = selectRuleta(fitness, cantPadres)
# print("\nRuleta: ")
# print(f"Los indices seleccionados son: {idxPadres}")
# print(f"Los valores de fitness son: {fitness[idxPadres]}")

# idxPadres = selectVentana(fitness, cantPadres)
# print("\nVentana: ")
# print(f"Los indices seleccionados son: {idxPadres}")
# print(f"Los valores de fitness son: {fitness[idxPadres]}")

# idxPadres = selectCompetencia(fitness, cantPadres)
# print("\nCompentencia: ")
# print(f"Los indices seleccionados son: {idxPadres}")
# print(f"Los valores de fitness son: {fitness[idxPadres]}")
