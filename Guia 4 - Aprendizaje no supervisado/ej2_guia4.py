from sklearn.datasets import load_iris
from k_medias import *
from desordenarDatos import *

#! ----- FALTA TERMINAR -----
#! Habria que pensar como se puede mostrar de forma grafica tambien.

data, yd = load_iris(return_X_y = True)

# datos = load_iris()     # en target las etiquetas de salidas deseadas (vienen ordenadas por categorias)
# etiquetas = desordenarDatos(datos.target)
# print(etiquetas)

centroides, clusters = k_medias(data, 3, 100)

# en este problema son "k" centroides que tendran 4 dimensiones porque los datos de iris vienen asi,
# y en "clusters" tendre un vector con vectores donde estan los indices de los puntos de datos
# que van dentro de cada cluster.
 
print(centroides)   
print("")
print(clusters)     