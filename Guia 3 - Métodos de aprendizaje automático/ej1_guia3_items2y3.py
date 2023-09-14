# Ejercicio 1 - Guia 3 - Items 2 y 3

import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

#* --- Cargar datos ---

X, yd = load_digits(return_X_y=True)   

#* --- MLPClassifier ---
# Es un perceptron multicapa para usar como clasificador.
# En la documentación se puede ver todos los parámetros que se pueden usar y que significa cada uno.
clf = MLPClassifier(hidden_layer_sizes=(30, 10), learning_rate_init=0.01, max_iter=300, activation='logistic',
                    early_stopping=True, validation_fraction=0.3, shuffle=True, random_state=0)

# este dio mejores resultados con learning_rate_init entre 0.01 y 0.005, fuera de eso empeora mucho
# probe las mismas arquitecturas (hidden_layer_sizes) que en el ej1

#* --- Particionar ---
# KFold:
    # permite utilizar la validacion cruzada con k-fold que vimos en teoria  

#? --- Item 2: 5 particiones generadas mediante KFold ---
# Recordar no confundir k-fold con leave-k-out (repasar de la teoria)

print("\nItem 2: 5 particiones generadas mediante KFold")

scores_kf5 = []
kf_5 = KFold(n_splits=5, shuffle=True)
# shuffle=True si queremos que se mezclen los datos antes de generar las particiones 
# (no se mezclas las muestras en cada particion)
# usando shuffle=True da mejores resultados que son False y da igual con k=5 que con k=10, 
# mientras que con shuffle=False con k=10 daba mejor que con k=5

for train, test in (kf_5.split(X)):
    # Cada pliegue está constituido por dos matrices: la primera está relacionada con el conjunto de 
    # entrenamiento y la segunda con el conjunto de prueba. Así, se pueden crear los conjuntos de 
    # entrenamiento/prueba utilizando la indexación numpy:

    X_train, X_test, yd_train, yd_test = X[train], X[test], yd[train], yd[test]
    clf.fit(X_train, yd_train) 
    scores_kf5.append(clf.score(X_test, yd_test))   
    # en vez de usar predic y luego calcular el accuracy, use directo score

print("Media con 5 particiones:", round(np.mean(scores_kf5), 2))
print("Varianza con 5 particiones:", round(np.var(scores_kf5), 4))

#? --- Item 3: 10 particiones generadas mediante KFold ---

print("\nItem 3: 10 particiones generadas mediante KFold")

scores_kf10 = []
kf_10 = KFold(n_splits=10, shuffle=True)

for train, test in (kf_10.split(X)):
    X_train, X_test, yd_train, yd_test = X[train], X[test], yd[train], yd[test]
    clf.fit(X_train, yd_train) 
    scores_kf10.append(clf.score(X_test, yd_test))

print("Media con 10 particiones:", round(np.mean(scores_kf10), 2))
print("Varianza con 10 particiones:", round(np.var(scores_kf10), 4))