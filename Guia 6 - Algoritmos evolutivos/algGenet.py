import numpy as np 
import matplotlib.pyplot as plt
from seleccion import *
from reproduccion import *
from graficas import *
from gradienteDescendente import *

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
        # Definir el rango del cromosoma a convertir en entero
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
    pobDecod = np.empty(shape=(cantInd, codCrom.shape[0]))

    # Recorremos los individuos (cromosomas) y los decodificamos para evaluar la funcion de fitness dada
    for i in range(cantInd):
        cromosoma = poblacion[i, :]
        val = decodificar(cromosoma, codCrom, xmin, xmax)
        pobDecod[i, :] = val
        fitness[i] = func(val)      # evaluo con la funcion de fitness
    
    # Convertirmos a un problema de maximizacion del fitness, ya que en realidad es minimizacion:
    fitness = 1 - (fitness/np.max(fitness))
    # fitness = 1/fitness
    # fitness = 1/(1 - fitness)
    # fitness = -fitness

    # Retornar el maximo fitness, valores de fitness y vector con valores de cromosomas decodificados
    return np.max(fitness), fitness, pobDecod

def algGenetico(func, xmin, xmax, cantIndividuos, cantPadres,
                codCrom, fitnessBuscado, cantMaxGeneracion, 
                probMutacion, probCruza, graf=0, df=None):
    """
        Funcion que implementa el algoritmo genetico.
    """
    
    # Convertir las listas en arreglos de Numpy
    codCrom = np.array(codCrom)
    xmin = np.array(xmin)
    xmax = np.array(xmax)

    #* Inicializar de forma aleatoria la poblacion 
    # Cada individuo es un cromosoma (una cadena binaria)
    lenCromosoma = np.sum(codCrom)

    # Formo un arreglo de arreglos, con 0s y 1s con cantIndividuos filas y lenCromosoma columnas,
    # asi tendremos un arreglo de 0s y 1s para cada invididuo de la poblacion (cromosomas)
    poblacion = np.random.randint(0, 2, size=(cantIndividuos, lenCromosoma))

    # Pasar a boolean para agilizar los calculos posteriores (arreglo con booleanos)
    poblacion = poblacion.astype(bool)

    #* Evaluar la poblacion con la funcion de fitness dada por el problema
    # Dentro de evaluar tambien se decodifica cada cromosoma para evaluar la funcion de fitness
    maxFit, fitness, pobDecod = evaluar(func, poblacion, codCrom, xmin, xmax)
    
    # copia de los puntos decodificados para usarlos luego en el gradiente descendente 
    # como puntos iniciales
    xInit = np.copy(pobDecod)

    # Cuando la entrada es entre [0, 1] (porcentaje de la poblacion total)
    if(isinstance(cantPadres, float)):
        cantPadres = int(cantIndividuos * cantPadres)

    #* Bucle hasta cumplir criterio de corte
    actualMaxFitness = maxFit
    cantGeneraciones = 0
    n = 0
    idxElite = 0 

    # Graficas (0 = No graficar, 1 = f1, 2 = f2)
    if (graf == 1):
        ax = grafica_f1()
        puntos = agregar_puntos_graf_f1(ax, pobDecod)
        
    if (graf == 2):
        ax = grafica_f2()
        puntos = agregar_puntos_graf_f2(ax, pobDecod)

    while(maxFit < fitnessBuscado):

        #* Aplicar metodo de seleccion
        idxPadres = selectVentana(fitness, cantPadres)
        # idxPadres = selectCompetencia(fitness, cantPadres)
        # idxPadres = selectRuleta(fitness, cantPadres)

        #* Aplicar operadores de cruza y mutacion
        poblacionCruza = repCruza(poblacion, idxPadres, codCrom, probCruza)
        newPoblacion = repMutacion(poblacionCruza, probMutacion, codCrom)

        #* Elitismo 
        # (pisar el ultimo hijo)
        idxElite = np.argsort(-fitness)[0]
        newPoblacion[-1] = poblacion[idxElite]

        # Volver a evaluar la nueva poblacion
        maxFit, fitness, pobDecod = evaluar(func, newPoblacion, codCrom, xmin, xmax)

        # Actualizar la poblacion
        poblacion = newPoblacion

        # Si el mejor fitness no mejora durante "n" generaciones seguidas, se corta
        # el algoritmo porque suponemos que convergio
        if(actualMaxFitness >= maxFit):
            n += 1
        else:
            actualMaxFitness = maxFit
            n = 0

        cantGeneraciones += 1
        # Criterio de corte, maxima generacion o 10 generaciones seguidas sin mejora
        if(cantGeneraciones >= cantMaxGeneracion or n >= 10):
            print(f"Finalizado por criterio de corte.")
            if(n >= 10):
                print(f"n >= {n}")
            else:
                print(f"Cantidad de generacion >= {cantGeneraciones}")
            break

        # Actualizar graficas
        if(cantGeneraciones % 4 == 0):
            if(graf == 1):
                actualizar_graf_f1(puntos, pobDecod)
            if(graf == 2):
                actualizar_graf_f2(puntos, pobDecod)

    # Cuando sale del bucle
    if(graf == 1):
        actualizar_graf_f1(puntos, pobDecod)
    if(graf == 2):
        actualizar_graf_f2(puntos, pobDecod)

    # Decodificamos el mejor individuo (solucion) para devolverlo
    idxMejor = np.argmax(fitness)
    mejorIndiv = decodificar(poblacion[idxMejor], codCrom, xmin, xmax)
    
    return mejorIndiv, xInit    

def algGradient(xInit, df, xmin, xmax, graf):

    valPuntos = -1000
    min_f = np.empty_like(xInit)

    if(graf == 1):
        # Graficar la parte de gradiente descendiente
        ax2 = grafica_f1()
        puntos = agregar_puntos_graf_f1(ax2, xInit, 0)

        for idx, val in enumerate(xInit):
            min_f[idx, :] = gradienteDescendente(df, xmin, xmax, val)

        plt.pause(1.5)
        valPuntos = actualizar_graf_f1(puntos, min_f)
        
    if(graf == 2):
        # Graficar la parte de gradiente descendiente
        ax2 = grafica_f2()
        puntos = agregar_puntos_graf_f2(ax2, xInit, 0)

        for idx, val in enumerate(xInit):
            min_f[idx, :] = gradienteDescendente(df, xmin, xmax, val)

        plt.pause(1.5)
        valPuntos = actualizar_graf_f2(puntos, min_f)

    # Mejor individuo despejado por gradiente descendiente
    idxMinValPuntos = np.argmin(valPuntos)
    mejorIndivGrad = min_f[idxMinValPuntos]
    
    return mejorIndivGrad
        
#todo =======================[Test]=======================
#? test bin2int
# cromosoma = np.array([True, True, True, False, True])
# print(bin2int(cromosoma)) 

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