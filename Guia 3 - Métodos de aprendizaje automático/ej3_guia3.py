# Ejercicio 3 - Guia 3
#* Metodos de ensambles de clasificadores Bagging y AdaBoost

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

#* Cargar datos
X, yd = load_wine(return_X_y=True)  # devuelve (data, target) directamente

# Como vimos en el ej1, otra forma obteniendo toda la info y luego separando data y target es:
# data_wine = load_wine()
# X = data_wine.data
# yd = data_wine.target
# labels = data_wine.target_names 

# print(yd)   # se ve que las categorias vienen ordenadas

# Estos metodos se basan en el ensamblado de varios clasificadores debiles base, 
# por defecto utilizan arboles de decision, pero se puede usar otros (VSM por ejemplo).
# buscando en internet y probando llegue a que este estimador base usando un arbol de decision
# funciona bastante bien.
estimator_base = DecisionTreeClassifier(max_depth=5)
# estimator_base_VSM = SVC(kernel='poly')

#* Bagging
clf_Bagging = BaggingClassifier(estimator=estimator_base, n_estimators=50, max_samples=0.8, max_features=0.8)
# n_estimators: numero de estimadores (arboles) en el ensamblado, al aumentar mejor el rendimiento y aumenta 
# el tiempo de entrenamiento.
# max_samples: fracción de muestras a utilizar para entrenar cada estimador. 
# max_features: fracción de características a utilizar para entrenar cada estimador (puede ser útil si tienes un gran número de características).

# otra opcion, funciona peor:
# clf_Bagging = BaggingClassifier(estimator=estimator_base_VSM, n_estimators=10, random_state=0)     

#* AdaBoost
clf_AdaBoost = AdaBoostClassifier(estimator=estimator_base, n_estimators=50)
# en AdaBoost los pesos de los clasificadores posteriores se centra mas en los casos dificiles de clasificar

# accuracy_bagging = []
score_Bagging = []
score_AdaBoost = []

# Generamos 5 particiones con KFold
kf_5 = KFold(n_splits=5, random_state=None, shuffle=False)

for train, test in (kf_5.split(X)):
    X_train, X_test, yd_train, yd_test = X[train], X[test], yd[train], yd[test]

    #* Entreno y pruebo cada modelo, guardando en un vector los resultados de cada uno en las 5 particiones
    #* en vez de hacer la prueba con "predict" y luego obtener el accuracy, use "score" (explicado en el ej1_G3_item1)

    # Entrenar y probar clasificador bagging
    clf_Bagging.fit(X_train, yd_train) 
    score_Bagging.append(clf_Bagging.score(X_test, yd_test))

    # para probar que da lo mismo usar el metodo "score" directamente en lugar de predict y luego accuracy_score
    # y_pred = clf_Bagging.predict(X_test)    
    # accuracy_bagging.append(accuracy_score(yd_test, y_pred))  

    # Entrenar y probar clasificador AdaBoost
    clf_AdaBoost.fit(X_train, yd_train) 
    score_AdaBoost.append(clf_AdaBoost.score(X_test, yd_test))

#* --- Muestro los resultados obtenidos con los clasificadores ---

# print("Accuracy media del Bagging:", round(np.mean(accuracy_bagging)*100, 2), "%")    
print("Media del Bagging:", round(np.mean(score_Bagging), 2))
print("Varianza del Bagging:", round(np.var(score_Bagging), 4), "\n")

print("Media del AdaBoost:", round(np.mean(score_AdaBoost), 2))
print("Varianza del AdaBoost:", round(np.var(score_AdaBoost), 4), "\n")
        