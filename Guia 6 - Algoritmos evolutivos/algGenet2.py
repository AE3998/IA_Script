import numpy as np 
import matplotlib.pyplot as plt
from seleccion import *
from reproduccion import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def evaluar(poblacion, x_train, y_train, x_test, y_test, alpha, beta):

    # Inicializar los parametros de entrenamiento
    cantInd = poblacion.shape[0]
    totalFeature = poblacion.shape[1]
    fitness = np.empty(shape=(cantInd))


    # Recorremos los individuos (cromosomas) y evaluar su fitness
    for i in range(cantInd):

        # Inicializar el clasificador
        clf = KNeighborsClassifier(n_neighbors=5) 
        # clf = SVC(kernel='poly')

        cromo = poblacion[i, :]
        selectedFeature = np.sum(cromo)

        # Usar clasificador
        aux_x_train = x_train[:, cromo]
        aux_y_train = y_train

        clf.fit(aux_x_train, aux_y_train)
        accuracy = accuracy_score(y_test, clf.predict(x_test[:, cromo]))

        fitness[i] = alpha * accuracy - beta * selectedFeature/totalFeature

    return fitness

def cargarDatos(nomArchivo):
    data = np.genfromtxt(nomArchivo, delimiter=',', max_rows=None)
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def algGenetico(archivo_train, archivo_test, cantIndividuos, cantPadres,cantMaxGeneracion, 
                probMutacion, probCruza, alpha, beta, fitnessBuscado):

    x_train, y_train = cargarDatos(archivo_train)
    x_test, y_test = cargarDatos(archivo_test)

    # Inicializar de forma aleatoria la poblacion 
    lenCromosoma = x_train.shape[1]

    # Inicializar la poblacion de cromosomas con True y False
    poblacion = np.random.choice([True, False], size=(cantIndividuos, lenCromosoma))

    # Evaluar la poblacion con la funcion de fitness dada por el problema
    fitness = evaluar(poblacion, x_train, y_train, x_test, y_test, alpha, beta)
    maxFit = np.max(fitness)

    # Cuando la entrada es entre [0, 1] (porcentaje de la poblacion total)
    if(isinstance(cantPadres, float)):
        cantPadres = int(cantIndividuos * cantPadres)

    #* Bucle hasta cumplir criterio de corte
    codCrom = np.array([lenCromosoma])
    actualMaxFitness = maxFit
    cantGeneraciones = 0
    idxElite = 0 
    n = 0

    while(maxFit < fitnessBuscado):

        #* Aplicar metodo de seleccion
        idxPadres = selectVentana(fitness, cantPadres)
        # idxPadres = selectCompetencia(fitness, cantPadres)
        # idxPadres = selectRuleta(fitness, cantPadres)

        #* Aplicar operadores de cruza y mutacion
        poblacionCruza = repCruza(poblacion, idxPadres, codCrom, probCruza)
        newPoblacion = repMutacion(poblacionCruza, probMutacion, codCrom)

        #* Elitismo 
        # (pisar el ultimo hijo)
        idxElite = np.argsort(-fitness)[0]
        newPoblacion[-1] = poblacion[idxElite]

        # Volver a evaluar la nueva poblacion
        fitness = evaluar(newPoblacion, x_train, y_train, x_test, y_test, alpha, beta)
        maxFit = np.max(fitness)
        print(f"maxFit: {maxFit}")

        # Actualizar la poblacion
        poblacion = newPoblacion

        # Si el mejor fitness no mejora durante "n" generaciones seguidas, se corta
        # el algoritmo porque suponemos que convergio
        if(actualMaxFitness >= maxFit):
            n += 1
        else:
            actualMaxFitness = maxFit
            n = 0

        cantGeneraciones += 1
        # Criterio de corte, maxima generacion o 10 generaciones seguidas sin mejora
        if(cantGeneraciones >= cantMaxGeneracion or n >= 10):
            print(f"Finalizado por criterio de corte.")
            if(n >= 10):
                print(f"n >= {n}")
            else:
                print(f"Cantidad de generacion >= {cantGeneraciones}")
            break

    # Retornar el individuo con mejor fitness

    print(f"Seleccion finalizado en {cantGeneraciones} generaciones.")
    idxMejor = np.argmax(fitness)

    return poblacion[idxMejor], fitness[idxMejor]