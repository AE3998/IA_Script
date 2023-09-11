# Ejercicio 2 - Guia 3
# Usando el esquema de 5 particiones generadas con KFold, comparar
# el desempeño del perceptron multicapa con los siguientes clasificadores:
# Naive Bayes, Analisis discriminante lineal, K vecinos mas cercanos, Arbol de decision y
# Maquina de soporte vectorial.

#* Imports de todos los metodos
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#* --- Cargar datos ---

X, yd = load_digits(return_X_y=True)   

# Cuando se usa "clf" es de "clasificador", porque tambien hay otros algoritmos para regresion

#* MLPClassifier 
# Perceptron multicapa para usar como clasificador.
clf_MLP = MLPClassifier(hidden_layer_sizes=(20, 10), learning_rate_init=0.005, max_iter=300, activation='logistic',
                    early_stopping=True, validation_fraction=0.3, shuffle=True, random_state=0)

#* Naive Bayes
clf_GNB = GaussianNB()
# solo tiene dos posibles parametros que por lo general no se utilizan.

#* Análisis discriminante lineal
clf_LDA = LinearDiscriminantAnalysis()

#* K vecinos mas cercanos
clf_K_neigh = KNeighborsClassifier(n_neighbors=5)   
# se puede cambiar el numero de vecinos. Con 5 obtuve el mejor resultado.
# Tambien se puede cambiar el tipo de "metrica" o distancia que se quiere usar 
# (minkowski, sino euclidea o Manhattan por ejemplo). Para eso se usan los parametros
# "p" y "metric" (p=1 -> manhattan, p=2 -> euclidean, p=arbitrario -> Minkowski)

#* Arbol de decision
clf_DTree = DecisionTreeClassifier(criterion='entropy')
# este puede recibir varios parametros (ver desde la documentacion)
# uno es el criterio o medida de seleccion (“gini” por defecto, “entropy”, “log_loss”)

#* Maquina de soporte vectorial
clf_SVC = SVC(kernel='poly')
# algunos de los parametros (todos opcionales) que se pueden modificar son:
#-kernel: linear, poly, rbf (radial, por defecto), sigmoid o precomputed. 
# Dependiendo de la forma de los datos puede ser mejor uno u otro nucleo o kernel, por
# ejemplo para separar clases que son curvas o no lineales.
# Es este caso, el mejor resultado se obtuvo con un kernel='poly' y luego con 'rbf'.
#-C: es un parametro de regularizacion para la penalizacion, entiendo que es cuanto error
# es soportable, un valor pequeño de C crea un hiperplano de menor margen y un valor de 
# C grande crea un hiperplano de mayor margen.

#* ---  ---

scores_MLP = []
scores_GNB = []
scores_LDA = []
scores_K_neigh = []
scores_DTree = []
scores_SVC = []

kf_5 = KFold(n_splits=5, random_state=None, shuffle=False)

for train, test in (kf_5.split(X)):
    X_train, X_test, yd_train, yd_test = X[train], X[test], yd[train], yd[test]

    #* Entreno y pruebo cada modelo, guardando en un vector los resultados de cada uno en las 5 particiones
    #* en vez de hacer la prueba con "predict" y luego obtener el accuracy, use "score" (explicado en el ej1_G3_item1)

    # MLP
    clf_MLP.fit(X_train, yd_train) 
    scores_MLP.append(clf_MLP.score(X_test, yd_test))

    # Naive Bayes
    clf_GNB.fit(X_train, yd_train) 
    scores_GNB.append(clf_GNB.score(X_test, yd_test))

    # Análisis discriminante lineal
    clf_LDA.fit(X_train, yd_train) 
    scores_LDA.append(clf_LDA.score(X_test, yd_test))

    # K vecinos mas cercanos
    clf_K_neigh.fit(X_train, yd_train) 
    scores_K_neigh.append(clf_K_neigh.score(X_test, yd_test))

    # Arbol de decision
    clf_DTree.fit(X_train, yd_train) 
    scores_DTree.append(clf_DTree.score(X_test, yd_test))

    # Maquina de soporte vectorial
    clf_SVC.fit(X_train, yd_train) 
    scores_SVC.append(clf_SVC.score(X_test, yd_test))

#* --- Muestro los resultados obtenidos con todos los clasificadores ---

print("Media del MLP:", round(np.mean(scores_MLP)*100, 2), "%")
print("Varianza del MLP:", round(np.var(scores_MLP)*100, 2) , "%\n")

print("Media del Naive Bayes:", round(np.mean(scores_GNB)*100, 2), "%")
print("Varianza del Naive Bayes:", round(np.var(scores_GNB)*100, 2) , "%\n")

print("Media del Análisis discriminante lineal:", round(np.mean(scores_LDA)*100, 2), "%")
print("Varianza del Análisis discriminante lineal:", round(np.var(scores_LDA)*100, 2) , "%\n")

print("Media del K vecinos mas cercanos:", round(np.mean(scores_K_neigh)*100, 2), "%")
print("Varianza del K vecinos mas cercanos:", round(np.var(scores_K_neigh)*100, 2) , "%\n")

print("Media del árbol de decisión:", round(np.mean(scores_DTree)*100, 2), "%")
print("Varianza del árbol de decisión:", round(np.var(scores_DTree)*100, 2) , "%\n")

print("Media del Support Vector Machine:", round(np.mean(scores_SVC)*100, 2), "%")
print("Varianza del Support Vector Machine:", round(np.var(scores_SVC)*100, 2) , "%\n")

# En algun momento que hablaba de normalizar las entradas, lo que hacia era normalizar cada dimension
# es decir, en el caso del Iris por ejemplo, todos los anchos de petalo los normalizo sacando la media y 
# dividiendo..., luego la columna de largo de sepalo normalizado sacando su media... y asi con cada dimension