import numpy as np 
import matplotlib.pyplot as plt

#? Pseudo codigo de algoritmos geneticos:
#? Inicializar poblacion
#? Evaluar poblacion
#? Mientras(mejor_fitness < fitness_buscado) hacer
#?      Seleccion de progenitores (padres)
#?      Reproduccion (cruza y mutacion)
#?      Reemplazo poblacional
#?      Evaluar poblacion
#? Fin

#* Convertir binario (en formato array) a int
def bin2int(cromosoma):
    # Trabajando con indexado de tipo bool para no usar un bucle y ahorrar calculos
    exp = np.arange(cromosoma.shape[0]-1, -1, -1) 
    res = np.sum(2**exp[cromosoma]) 
    return res

    # Otra forma trabajando con bucle
    # res = 0
    # for i in range(len(cromosoma)):
    #     if(cromosoma[i] == 1):
    #         exp = len(cromosoma)-1-i 
    #         res += np.sum(2**exp) 
    # return res

def decodificar(cromosoma, codCrom, xmin, xmax):
    """
    Entrada:
    cromosoma =  [1,1,1,0 ,0,1,0,1] (bit (+) <-Significativo-> (-))
    codCrom = [4 (bits), 4 (bits)]
    val = bin2int(cromosoma) = [14, 5]

    xmin = [-100, -100]
    xmax = [100, 100]

    Salida: 
    Recordar que queremos representarlo en un rango [a, b]. 
    La ecuacion es:
    x = a + (b - a) * val/(2**k - 1)

    Por ejemplo bin2int ajustado en el rango de [-100 (xmin), 100 (xmax)]:
    a = xmin[0] = xmin[1] = -100
    b = xmax[0] = xmax[1] =  100
    k = codCrom[0] = codCrom[1] = 4

    cromDecod[0] = [-100 + 200*(14/15) = 86.6]
    cromDecod[1] = [-100 + 200*(5/15) = -33.3]
    """
    cromDecod = np.empty_like(codCrom, dtype=float)

    right = 0
    for i in range(codCrom.shape[0]):
        # Definir el rango de la cromosoma a convertir en entero
        left = right
        right += codCrom[i]
        val = bin2int(cromosoma[left:right])

        # Ajustar el numero entero en el rango de [xmin, xmax]
        minVal = xmin[i]
        maxVal = xmax[i]

        rango = (maxVal - minVal)
        den = (2**codCrom[i] - 1)
        cromDecod[i] = minVal + val*rango/den
    
    return cromDecod

def evaluar(func, poblacion, codCrom, xmin, xmax):

    # Cantidad de individuos
    cantInd = poblacion.shape[0]

    fitness = np.empty(shape=(cantInd))
    valDecod = np.empty(shape=(cantInd, len(codCrom)))

    for i in range(cantInd):
        cromosoma = poblacion[i, :]
        val = decodificar(cromosoma, codCrom, xmin, xmax)
        valDecod[i, :] = val
        fitness[i] = func(val)
    
    # Retornar el maximo fitness, los valores de fitness 
    # y vector de los valores de cromosomas decodificados
    return np.max(fitness), fitness, valDecod

def algGenetico(func, xmin, xmax, cantInd, codCrom, probMutacion, probCruza):

    # Convertir las listas en arreglos
    codCrom = np.array(codCrom)
    xmin = np.array(xmin)
    xmax = np.array(xmax)

    # inisicalizar la poblacion de  cromosomas
    lenCromosoma = np.sum(codCrom)
    poblacion = np.random.randint(0, 2, size=(cantInd, lenCromosoma))

    # Pasar en boolean para agilizar los calculos posteriores
    poblacion = poblacion.astype(bool)

    # Evaluar el fitness de la poblacion
    maxFit, fitness, valDecod = evaluar(func, poblacion, codCrom, xmin, xmax)
    #! Guarda que los fitness son propiamente los valores de la funcion en ese
    #! punto que puede ser muy grande, habria que buscar una forma de normalizarla
    #! De hecho buscamos minimizar el error, mientras que fitness se hace grande 
    #! cuando la funcion evaluado en ese punto es grande.



#todo =======================[Test]=======================
#? test bin2int
cromosoma = np.array([True, True, True, False, True])
print(bin2int(cromosoma)) 

#? test decodificar
#* 1D [10 bits para x]
# cromosoma = np.array([0,1,1,1,1,0,1,0,1,0]) #= [490]
# cromosoma = cromosoma.astype(bool)
# print(bin2int(cromosoma))

# codCrom = np.array([10])
# xmin = [-512]
# xmax = [512]

# res = decodificar(cromosoma, codCrom, xmin, xmax)
# print(res)


#* 2D [4 bits para x, 4 bits para y]
# cromosoma = np.array([1,1,1,0 ,0,1,0,1]) #= [14, 5]
# cromosoma = cromosoma.astype(bool)
# codCrom = np.array([4, 4])
# xmin = [-100, -100]
# xmax = [100, 100]

# res = decodificar(cromosoma, codCrom, xmin, xmax)
# print(res)

#? test selectVent(ordenFit, cantPadres)
# ordenFit = np.random.permutation(10)
# cantPadres = 4
# print(selectVent(ordenFit, cantPadres))