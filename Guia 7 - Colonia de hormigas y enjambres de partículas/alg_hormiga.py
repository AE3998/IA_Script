import numpy as np
import matplotlib.pyplot as plt
from graficas import graficarFeromona, graficarCamino

def cargarDatos(nombArch):
    data = np.genfromtxt(nombArch, delimiter=",", max_rows=None)
    return data

def obtenerProximoNodo(idx_i, sigmaPorEta, idxBool, idxCamino):
    # Selecciona el proximo nodo segun la formula de probabilidad "p" que dimos
    # Sigma es la matriz con la cant de feromonas que hay para ir de un camino a otro, 
    # y ƞ (eta) es el costo de hacer ese paso, en nuestro caso es la inversa de la 
    # distancia entre dos ciudades (podria ser otro costo).

    # Sacar el numerador y el denominador de los caminos disponibles de ese nodo
    # Se filtra con el indexado booleano
    numerador = sigmaPorEta[idx_i, :][idxBool]
    denominador = np.sum(numerador)

    # Calcular las probabilidades
    probabilidad = numerador/denominador

    # Elegimos al azar un camino disponible (definido por idxBool) segun las probabilidades
    # Similar a un metodo de la ruleta tirando un valor al azar y segun las probabilidades elegimos
    idx_j = np.random.choice(idxCamino[idxBool], p=probabilidad)

    return idx_j

def depositarFeromona(metodo, matrizFeromona, matrizCaminos, todoCaminoRecorrido, distRecorridas, Q):
    # -Metodo uniforme: depositar una cantidad constante τ en cada transicion. 
    # -Metodo global: depositar una fraccion del total τ de feromonas: cantFeromona/long total del camino.
    # Cuanto mas corto el camino, mas feromonas vamos a dejar, pero en todas las transiciones deja la misma cantidad.
    # -Metodo local: cantFeromonas / dist. de una ciudad a la otra. En cada transicion sera distinta cantidad.

    cantCaminos = todoCaminoRecorrido.shape[0]
    longCaminos = todoCaminoRecorrido.shape[1]

    # Inicializar el denominador como 1 que seria para el metodo uniforme (metodo == 1)
    denominador = 1

    # Recorrer cada camino de hormiga k
    for k in range(cantCaminos):

        if(metodo == 0):    # metodo global
            denominador = distRecorridas[k]     # longitud total del camino

        # Recorrer cada nodo recorrido por hormiga k
        for i in range(longCaminos - 1):
            # Extraer el par de indices
            idx_i = todoCaminoRecorrido[k, i]
            idx_j = todoCaminoRecorrido[k, i + 1]

            if(metodo == 2):    # metodo Local
                denominador = matrizCaminos[idx_i, idx_j]   # longitud de esa transicion de i a j

            # Deposito de feromona en esa transicion del camino
            # Q es la cant de feromonas y se recibe como parametro
            matrizFeromona[idx_i, idx_j] += Q/denominador

    return matrizFeromona

def camino_optimo(nombArch, cantHorm, itMax, tasaEvap, metodoDepositoFeromona, nodoInit, Q, graf=False):
    """
        Algoritmo de colonia de hormigas para obtener camino optimo.
        Entradas: nombre del archivo de datos, cant de hormigas, numero max de iteraciones,
        tasa de evaporacion, metodo de deposito de feromonas, nodo inicial para las hormigas,
        cantidad de feromonas (Q) y si se desea graficar o no.
        Salida: mejor camino, distancias recorridas y numero de iteraciones.
    """
    
    matrizCaminos = cargarDatos(nombArch)

    # Con el archivo gr17.csv vamos a tener una matriz de 17x17, asi que la matriz de caminos
    # (ciudades por las que pasaria) y la matriz de feromonas seran de 17x17
    dim0 = matrizCaminos.shape[0]
    dim1 = matrizCaminos.shape[1]

    # Inicialzamos la matriz de feromonas con valores pequeños.
    # Recordar que haya Feromona por cada par de nodo i-j
    matrizFeromona = np.random.rand(dim0, dim1) * 0.05

    # sigma es la cantidad de feromonas, y ƞ (eta) es el costo de hacer ese paso, 
    # que en nuestro caso es la distancia del camino pero podria ser otro costo

    diag = np.diag_indices_from(matrizCaminos)  # indices de la diag principal
    # Despejar la matriz Eta que es 1/dist
    matrizEta = matrizCaminos.copy()
    matrizEta[diag] = 1         # 1 en la diagonal para evitar division por 0
    matrizEta = 1/matrizEta     # 1 / distancia entre los caminos
    matrizEta[diag] = 0         # 0 en la diagonal porque la dist de una ciudad a si misma es 0

    # Los indices de camino 
    idxCamino = np.arange(dim0)

    cantCaminoIgual = 0
    it = 0
    
    # Grafica
    if(graf):
        producto = 100  # para que los numeros sean mas visible
        _, ax = plt.subplots(figsize=(6, 6))
        title = "Inicio de feromonas"
        ax = graficarFeromona(ax, matrizFeromona, title, producto)

    # Bucle mientras no se llege a un max de iteraciones y hasta que todas las hormigas se alinean 
    # durante "n" iteraciones consecutivas (explicado en las anotaciones) 
    while(it < itMax and cantCaminoIgual < 4):
        it += 1

        # Inicializar con True por cada iteracion
        caminoIgual = True

        # Las distancias y el camino recorrido por las k hormigas
        distRecorridas = np.zeros(shape=(cantHorm), dtype=int)
        todoCaminoRecorrido = np.empty(shape=(cantHorm, dim0 + 1), dtype=int)

        # (Sigma ij)^alpha * (Eta ij)^beta
        sigmaPorEta = matrizFeromona * matrizEta    # numerador de la formula de probabilidad "p"

        # Recorrer cada hormiga
        for k in range(cantHorm):

            # Lista para ir guardando el camino recorrido por la hormiga k
            caminoRecorrido = np.empty(shape=(dim0 + 1), dtype=int)
            caminoRecorrido[0] = nodoInit
            caminoRecorrido[-1] = nodoInit

            # idx boolean para ir descatando los nodos ya visitados (por la lista de vecinos con tabu)
            idxBool = np.full(shape=(dim0), fill_value=True, dtype=bool)    # matriz de tamanio shape con valores fill_value
            idxBool[nodoInit] = False

            # Seleccion de caminos hasta el ultimo nodo disponible
            # Vuelve al inicio cuando sale del ciclo for
            for i in range(dim0 - 1):

                # El ultimo nodo que esta parado, al inicio es nodoInit
                idx_i = caminoRecorrido[i]
                idx_j = obtenerProximoNodo(idx_i, sigmaPorEta, idxBool, idxCamino)

                # Registrar el nodo, descartarlo de la lista de disponibles y sumar en la distancia recorrida
                caminoRecorrido[i+1] = idx_j
                idxBool[idx_j] = False
                distRecorridas[k] += matrizCaminos[idx_i, idx_j]

            ultimoNodo = caminoRecorrido[-2]    # nodo antes de volver al nodo inicial
            distRecorridas[k] += matrizCaminos[ultimoNodo, nodoInit]    # sumo la distancia para volver al nodo inicial

            # Chequear si el camino recorrido es igual que el camino de la 
            # hormiga previa, si son diferentes ya no chequeo mas 
            if(caminoIgual and k != 0):
                caminoIgual = np.all(caminoRecorrido == todoCaminoRecorrido[k-1])
            
            todoCaminoRecorrido[k] = caminoRecorrido

        # Cuando termino de recorrer todas las hormigas, verifico si fueron todos los caminos iguales
        if(caminoIgual):
            cantCaminoIgual += 1
        else:
            cantCaminoIgual = 0

        # Evaporacion de feromonas: actualizar la feromona segun la tasa de evaporacion
        matrizFeromona = (1 - tasaEvap) * matrizFeromona
        # Depositar feromona en el camino
        matrizFeromona = depositarFeromona(metodoDepositoFeromona, matrizFeromona, 
                                           matrizCaminos, todoCaminoRecorrido, distRecorridas, Q)

        # Actualizar grafica de feromonas cada n iteraciones
        if(graf and it % 25 == 0):
            title = f"Matriz Feromona {producto}x, {it} iteraciones"
            ax = graficarFeromona(ax, matrizFeromona, title, producto)
    
    # Cuando sale del bucle while
    if(itMax <= it):
        print("Finalizo por llegar al numero maximo de iteraciones.")
    else:
        print(f"Se logro alinear las hormigas por {cantCaminoIgual} veces.")
        print(f"En total {it} iteraciones.")

    # Buscamos la menor distancia recorrida y el mejor camino
    idxMejor = np.argmin(distRecorridas)
    mejorCamino = todoCaminoRecorrido[idxMejor]

    if(graf):
        title = f"Finalizado en {it} iteraciones.\n Camino: {str(mejorCamino)} \n" 
        graficarCamino(ax, mejorCamino, title)

    # Retornar el camino con menor distancia recorrida
    return mejorCamino, distRecorridas[idxMejor], it