from sklearn.datasets import load_iris
from k_medias import *
import matplotlib.pyplot as plt
# from desordenarDatos import *

#! ----- FALTA TERMINAR -----
#! Habria que pensar como se puede mostrar de forma grafica tambien.

data, yd = load_iris(return_X_y = True)
desordenarDatos = True

# Desordenar datos
if (desordenarDatos):
    idx_perm = np.random.permutation(data.shape[0])
    data = data[idx_perm]
    yd = yd[idx_perm]

# datos = load_iris()     # en target las etiquetas de salidas deseadas (vienen ordenadas por categorias)
# etiquetas = desordenarDatos(datos.target)
# print(etiquetas)

centroides, clusters = k_medias(data, 4, 100, grafDim=2)
plt.show()

# en este problema son "k" centroides que tendran 4 dimensiones porque los datos de iris vienen asi,
# y en "clusters" tendre un vector con vectores donde estan los indices de los puntos de datos
# que van dentro de cada cluster.
 
# print(centroides)   
# print("")
# print(clusters)     