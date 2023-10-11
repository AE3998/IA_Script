import numpy as np
from algGenet2 import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#? Test 
archivo_train = "leukemia_test.csv"
archivo_test = "leukemia_test.csv"
cantIndividuos = 50
cantPadres = 0.5
cantMaxGeneracion = 500
probMutacion = 0.8
probCruza = 0.1
alpha = 1
beta = 0.15
fitnessBuscado = 0.9

mejorIndv, mejorFit = algGenetico(archivo_train, archivo_test, cantIndividuos, cantPadres,cantMaxGeneracion, 
                                probMutacion, probCruza, alpha, beta, fitnessBuscado)

cantFeature = np.sum(mejorIndv)
porcentaje = np.round(cantFeature/7129 * 100, 2)
print(f"Mejor individuo que usa {cantFeature} caracteristicas")
print(f"que es {porcentaje} % del total.")
print(f"Logra un fitness de: {mejorFit}")



# #* Cada cromosoma sera un patron del dataset, es decir, sera de (1x7129), pero cambiando
# #* los valores para que quede todo con 1s y 0s. 
# #* Vamos a tener cromosomas de 7129 genes (las 7129 caracteristicas que se mencionan en el enunciado).
# #* Pero queremos elegir un conjunto reducido de caracteristicas o entradas, y para eso se usa 
# #* un algoritmo genetico.

# #* Cada individuo del algoritmo genetico sera una posible solucion, que sera una cadena de bits
# #* cuya longitud es la cantidad de caracteristicas (entradas) del dataset, y cada bit indica si
# #* queremos (1) o no (0) usar esa caracteristica en el entrenamiento.

# #* Lo otro que se modifica es la funcion de fitness...

# #* -------------------------------------------------------------------------

# #* Cargar datos
# #! Habria que cargar los datos originales y recortar el dataset para quedarnos con un grupo 
# #! reducido segun las caracteristicas que nos interesan. 
# #! ¿Antes de hacer eso se usa el algoritmo genetico para determinar que caracteristicas usamos?
# #! ¿Y luego de calcular el accuracy y el fitness con la formula dada, se vuelve a usar el 
# #! algoritmo genetico?

# #! EN MIS ANOTACIONES DE WORD O EN EL PDF DE LOS PROFES DE PRACTICA HAY UN ALGORITMO Y FORMULAS.

# #! Aca tendriamos algo como:
# # X_train = ...
# # yd_train = ...
# # X_test = ...
# # yd_test = ...

# # alfa y beta son parametros que controlan que tanta importancia se le da a estos dos objetivos 
# # Si alfa es mas grande, el algoritmo tratara de maximizar el accuracy sin importarle tanto usar 
# # mas cantidad de caracteristicas, mientras que si beta es mas alto tratara de reducir la cantidad  
# # de caracteristicas sacrificando un poco de accuracy.
# alpha = 1
# beta = 0.1

# #? fitness = alpha * accuracy - beta * (select_features/total_features)

# #* Inicializar un clasificador (de sklearn) 
# # (Puse los codigos para K vecinos mas cercanos y SVC pero despues vamos probando cual nos 
# # da mejores resultados)

# # K vecinos mas cercanos
# clf_K_neigh = KNeighborsClassifier(n_neighbors=5) 

# # Maquina de soporte vectorial
# # clf_SVC = SVC(kernel='poly')

# scores_K_neigh = []
# # scores_SVC = []

# #* Entrenamiento del clasificador
# # K vecinos mas cercanos
# clf_K_neigh.fit(X_train, yd_train) 
# scores_K_neigh.append(clf_K_neigh.score(X_test, yd_test))

# # Maquina de soporte vectorial
# # clf_SVC.fit(X_train, yd_train) 
# # scores_SVC.append(clf_SVC.score(X_test, yd_test))

# print("Accuracy K vecinos mas cercanos:", round(scores_K_neigh, 2))
# # print("Accuracy SVM:", round(scores_SVC, 2))