import numpy as np
from algGenet_Ej2 import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Funcion para comprobar que con las caracteristicas filtradas den un buen resultado al
# entrenar y probar el clasificador.

#* Habria que hacer esto mismo con un archivo que tenga todos 1 para verificar que nuestra 
#* solucion obtenida en verdad mantiene un buen accuracy reduciendo la cantidad de caracteristicas

archivo_train = "leukemia_train.csv"
archivo_test = "leukemia_test.csv"

x_train, y_train = cargarDatos(archivo_train)
x_test, y_test = cargarDatos(archivo_test)

# Guardamos el mejor individuo, que seria una arreglo de 0s y 1s solo con 1s en las
# caracteristicas filtradas, las que se desean utilizar
data = np.genfromtxt("mejorIndv.csv")

# Pasamos los indices de caracteristicas filtradas de 1s y 0s a True y False
idxBool = data.astype(bool)
print("Cantidad de caracteristicas utilizadas:", np.sum(idxBool))

# Filtramos los archivos de entrenamiento y test dejando solo las caracteristicas deseadas
x_train = x_train[:, idxBool]
x_test = x_test[:, idxBool]

# Inicializar el clasificador
clf = KNeighborsClassifier(n_neighbors=5) 
# clf = SVC(kernel='poly')

# Entrenamos con los datos ya filtrados
clf.fit(x_train, y_train)

# Calculamos el accuracy del clasificador
accuracy = accuracy_score(y_test, clf.predict(x_test))
print("Accuracy:", accuracy)