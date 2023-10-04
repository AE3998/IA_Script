from sklearn.datasets import load_iris
from k_medias import *
from SOM_entrenamiento import *
import matplotlib.pyplot as plt
from graficas import ContingencyMatrixDisplay

# Cargar datos
data, yd = load_iris(return_X_y = True)
desordenarDatos = True

# Desordenar datos
if (desordenarDatos):
    idx_perm = np.random.permutation(data.shape[0])
    data = data[idx_perm]
    yd = yd[idx_perm]

centroides, clusters_KM = k_medias(data, 4, grafDim=2)

#* Comparacion con SOM 
# nombreArchivo = "irisbin_trn.csv"
epocas = [100, 350, 100]
dimSom = [2, 2]
tasaAp = [0.6, 0.1]
radio = [2, 0.1]
iris = True

neuronasSOM, clusters_SOM = SOM_entrenamiento(data, epocas, dimSom, tasaAp, radio, iris)
colorearClustersSOM(data, neuronasSOM, clusters_SOM, iris=True)

# Graficamos la matriz de contingencia para comparar las soluciones de clustering obtenidas
# con el k-medias y el SOM
ContingencyMatrixDisplay(data, clusters_KM, clusters_SOM)
plt.show()

# En "clusters_KM" y "clusters_SOM" tendremos un vector con vectores donde estan los indices de 
# los puntos de datos que van dentro de cada cluster obtenido con k-medias y con el SOM. 

# Sobre la matriz de contingencia:
# Mientras nos de valores mas altos significa que encuentra mas coincidencias en esos clusters,
# y no necesariamente tienen que ser los valores de la diagonal como en la matriz de confusion, 
# sino que puede variar porque se arman distintos clusters.
# Y si uso la misma cantidad de clusters en cada metodo, por ejemplo 4, si las soluciones 
# de clustering obtenidas son similares, tendria que haber 4 valores altos y el resto bajos 