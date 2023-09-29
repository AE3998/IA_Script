from sklearn.datasets import load_iris
from k_medias import *
from SOM_entrenamiento import *
# from SOM import *
import matplotlib.pyplot as plt

#! ----- FALTA TERMINAR -----
#! Hacer la comparacion con el SOM que pide el ejercicio y hacer la matriz de contingencia
#! Tomamos cada neurona del SOM como columnas de la matriz de contingencia y cada cluster
#! del k-medias como filas de la matriz de contingencia.

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

centroides, clusters = k_medias(data, 4, grafDim=2)
#* Comparacion con SOM 

# nombreArchivo = "irisbin_trn.csv"
epocas = [100, 350, 100]
dimSom = [2, 2]
tasaAp = [0.6, 0.1]
radio = [2, 0.1]
iris = True

neurSom, clusters = SOM_entrenamiento(data, epocas, dimSom, tasaAp, radio, iris)
colorearClustersSOM(data, neurSom, clusters,iris=True)
plt.show()

# en este problema son "k" centroides que tendran 4 dimensiones porque los datos de iris vienen asi,
# y en "clusters" tendre un vector con vectores donde estan los indices de los puntos de datos
# que van dentro de cada cluster.
 
# print(centroides)   
# print("")
# print(clusters)     