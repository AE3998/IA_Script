import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import davies_bouldin_score, fowlkes_mallows_score, adjusted_rand_score
from k_medias import *

# Cargo los datos de Iris
iris = load_iris()
X = iris.data

# Lei que se puede escalar los datos (X) para que tengan media cero y varianza unitaria,
# asi en algunos casos puede mejorar la convergencia y facilitar la interpretacion.
# En sklearn se usa StandardScaler

# Inicializo vectores para guardar los resultados de las metricas
db_scores = []
fm_scores = []
rand_scores = []

# Itero sobre diferentes valores de k (desde 2 hasta 10)
for k in range(2, 11):
    # Ejecuto el algoritmo k-medias
    centroides, clusters = k_medias(X, k)
    
    # Calculo las etiquetas de cluster a partir de los indices
    etiquetas_clusters = np.zeros(X.shape[0])
    for i in range(len(clusters)):  # clusters -> vector de vectores
        cluster = clusters[i]      # cluster -> vector con indices de puntos dentro de ese cluster
        etiquetas_clusters[cluster] = i

    #* Métricas que utilizamos:
    # - Indice Davies-Bouldin (metrica interna):
    # Es útil para evaluar la compacidad y la separación de los clusters. Un valor mas cercano a
    # 0 de DB indica una mejor calidad del clustering, indicando que son mas compactos y separados.
    # - Indice Rand e Indice Fowlkes-Mallows (metricas externas):
    # Evaluan la similitud entre los clusters generados y los clusters de referencia 
    # (etiquetas verdaderas), que en el problema de Iris las tenemos. Entonces nos dan  
    # informacion sobre la calidad del clustering en funcion de una etiquetas verdaderas.
    # Valores mas altos de estos indices FM y RI indicarin un mejor resultado.

    # Calculo las metricas
    DB = davies_bouldin_score(X, etiquetas_clusters)
    # las metricas FM y RI reciben las etiquetas verdaderas (iris.target)
    FM = fowlkes_mallows_score(iris.target, etiquetas_clusters)
    RI = adjusted_rand_score(iris.target, etiquetas_clusters)

    db_scores.append(DB)
    fm_scores.append(FM)
    rand_scores.append(RI)

    #! graficar esos tres vectores en una misma grafica para mostrar como varian con el valor de "k"

    # Muestro los resultados
    print("\nk = ", k)
    print("\n", "Davies-Bouldin:", DB,
          "\n", "Fowlkes-Mallows:", FM, 
          "\n", "Rand Index:", RI)


plt.show()

#* Conclusiones obtenidas:
# k=2 nos da el valor de indice Davies-Doublin mas bajo, lo cual es bueno, pero con k=3 se tiene
# mejores valores para los indices FM y RI, ya que son valores mas altos, y sigue teniendo un valor
# de DB razonable.
# Otra cosa que podriamos pensar es que, en el problema de Iris, como sabemos que tenemos tres 
# categorias, puede ser coherente que k=3 sea el valor optimo, ya que queremos separar en cada 
# cluster un tipo de flor.
