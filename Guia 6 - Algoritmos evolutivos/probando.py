import numpy as np 
import matplotlib.pyplot as plt
from seleccion import *

# -----------------------------------------------

#* Convertir binario (en formato array) a int
res = 0
cromosoma = np.array([0,0,1,0])

for i in range(len(cromosoma)):
    if(cromosoma[i] == 1):
        exp = len(cromosoma)-1-i 
        res += np.sum(2**exp) 
print(res)

# -----------------------------------------------

# Inicializar de forma aleatoria la poblacion
cantIndividuos = 10
codCrom = [5, 5]
lenCromosoma = np.sum(codCrom)
poblacion = np.random.randint(0, 2, size=(cantIndividuos, lenCromosoma))
print(poblacion)
# Pasar a boolean para agilizar los calculos posteriores (arreglo con booleanos)
poblacion = poblacion.astype(bool)
print(poblacion)

# -----------------------------------------------

fitness = np.array([3, 2, 5, 1, 4, 6, 0, 2])
ordenFit = np.argsort(-fitness)
print("OrdenFit:", ordenFit)

cantPadres = 4
cantInd = ordenFit.shape[0]
paso =  cantInd // cantPadres
print("Paso:", paso)

idxBool = np.full(shape=(fitness.shape[0]), fill_value=True)
print(idxBool)

cantCompetencia = fitness.shape[0] // cantPadres
print(cantCompetencia)

# ----------------------------------------------------------------

cantIndividuos = 2
lenCromosoma = 4
poblacion = np.random.choice([True, False], size=(cantIndividuos, lenCromosoma))
print(poblacion)