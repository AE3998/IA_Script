# Ejercicio 1 - Guia 3 - Item 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

#* --- Cargar datos ---
# Este dataset "digits" tiene digitos (numeros) manuscritos, entonces se usa para entrenar modelos 
# que puedan identificar los numeros.
X, yd = load_digits(return_X_y=True)   
# return_X_y=True es un parametro de load_digits para que devuelva (data, target) en lugar de todo,  
# que es lo que necesitamos para entranar. Sino se puede hacer digits = load_digits() y devuelve 
# como todo un diccionario (tipo un JSON) y debemos acceder a "digits.data" (serian los datos "X") 
# y "digits.target" serian las etiquetas o salidas "y" que indican que numero seria.
# Eso se ve en la documentacion. 

# print(X)
# print(yd)   # la salida seran los digitos que estarian escritos [0 1 2 3 8...]
# print(X.shape)

# Otra forma de acceder (como explique arriba):
digits = load_digits()      # lo use solo para una version de la matriz de confusion
# X = digits.data
# yd = digits.target
# print(digits)
# print(X)
# print(yd)

#* --- MLPClassifier ---
# Es un perceptron multicapa para usar como clasificador.
# En la documentación se puede ver todos los parámetros que se pueden usar y que significa cada uno.
clf = MLPClassifier(hidden_layer_sizes=(20, 10), learning_rate_init=0.01, max_iter=300, activation='logistic',
                    early_stopping=True, validation_fraction=0.3, shuffle=True, random_state=0)
# hidden_layer_sizes es para definir lel numero de nueronas en cada capa. En la capa final deberian 
# ser 10 porque son 10 numeros los que se clasifican (del 0 al 9)
# early_stopping es para que el criterio de parada lo haga con datos que no vio en el entrenamiento, 
# seria la validacion o monitoreo que nosotros haciamos con los mismos datos de entrenamiento pero
# aca se hace con un grupo aparte, que ahi se le dio un 30% (0.3) de los datos de entrenamiento
# que los subdivide para validar y no los usa para entrenar.
# shuffle=True mezcla los parametros en cada epoca.
#! randam_state no entendi que es en la documentacion

#! Hay que probar que resultados da cambiando los parametros de MLPClassifier

#* --- Particionar ---
# train_test_split:
    # genera una particion simple de entrenamiento y prueba, es decir, de todos los
    # datos que tenemos separa una parte para entrenamiento y otra parte para pruebas.
# KFold:
    # permite utilizar la validacion cruzada con k-fold que vimos en teoria  

#? --- Item 1: Una unica particion de datos genererada mediante train_test_split ---

#* --- train_test_split ---
X_train, X_test, y_train, y_test = train_test_split(X, yd, test_size=0.3)  
# test_size es para indicar el porcentaje que queremos que deje para el test (0.3 = 30%)
# shuffe=True es para que ademas de particionar los datos, los mezcle de forma aleatoria. Por defecto
# viene como True, si se quiere que los datos queden como vienen, se debe usar shuffed=False
# random_state=42 es una semilla opcional, yo no la use

# print(len(X_train))
# print(len(X_test))

#* --- Entrenamiento ---
clf.fit(X_train, y_train)   # "fit" es para entrenar el modelo (no hace falta devolver algo, ya lo entrena)

#* --- Prueba ---
y_pred = clf.predict(X_test)    # "predict" es para predecir o probar el modelo 
                                # Devuelve las salidas obtenidas por la red

#* --- Calculo de medidas de desempeño ---
# Se podrian mostrar todas las medidas que vimos en la teoria "Medidas de desempeño y cap. de generalizacion"

#? Accuracy
ACC = accuracy_score(y_test, y_pred)    
print("Accuracy:", ACC)  # si da 0.8 es un accuracy del 80% por ejemplo

#? Score
# El metodo "score" devuelve el accuracy o exactitud media en los datos de prueba y etiquetas dadas,
# entiendo que utilizando el clasificador entrenado, hace como la prueba de "predict" y ya compara 
# con las salidas deseadas en las pruebas, mientras que sino tengo que usar predict para tener las 
# salidas que da la red y luego compararlas con las salidas deseadas usando accuary_score por 
# ejemplo como hice antes.
# "score" de clf no es lo mismo que Score F1, para ese se usa f1_score
score = clf.score(X_test, y_test)  
print("\nScore:", score)    

#? Matriz de confusion
matriz_confusion = confusion_matrix(y_test, y_pred, labels=digits.target_names) 
#? preguntar si labels ordena la matriz segun las etiquetas, porque no lo entendi bien
#? ahi labels seria [0 1 2 3 4 5 6 7 8 9]
# matriz_confusion = confusion_matrix(y_test, y_pred)   
print("\nMatriz de confusión:")
print(matriz_confusion)

#? Mostrar la matriz de confusion de forma grafica
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion, display_labels=clf.classes_)
disp.plot()
plt.show()

#? Classification report
print("\nClassification report:")
report = classification_report(y_test, y_pred, labels=digits.target_names)
# Devuelve un informe que muestre las principales métricas de clasificación
# En este caso en realidad "labels" no hace falta porque te muestra las categorias con numeros,
# pero lo dejo para recordar en otro caso que nos sirva. Sino se puede armar un vector 
# target_names = ['class 0', 'class 1', 'class 2'] y usar classification_report(y_test, y_pred, target_names=target_names))

# la columna "support" que muestra se refiere a la cantidad de elementos que se usaron en el test 
# para cada categoria y en accuracy el total de elementos
print(report)