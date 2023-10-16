import numpy as np
import matplotlib.pyplot as plt
from graficas import graficarFeromona, graficarCamino

def cargarDatos(nombArch):
    data = np.genfromtxt(nombArch, delimiter=",", max_rows=None)
    return data

def obtenerProximoNodo(idx_i, sigmaPorEta, idxBool, idxCamino):

    # Sacar el numerador y el denominador de los caminos disponibles de ese nodo
    # Se filtra con el indexado booleano
    numerador = sigmaPorEta[idx_i, :][idxBool]
    denominador = np.sum(numerador)

    # Despejar las probabilidades
    probabilidad = numerador/denominador

    # Elejir un camino "disponible" (definido por idxBool) segun las probabilidades
    idx_j = np.random.choice(idxCamino[idxBool], p=probabilidad)

    return idx_j

def depositarFermona(metodo, matrizFermona, matrizCaminos, todoCaminoRecorrido, distRecorridas):
    
    Q = 1
    cantCaminos = todoCaminoRecorrido.shape[0]
    longCaminos = todoCaminoRecorrido.shape[1]

    # Inicializar el denominador como 1 Uniforme
    denominador = 1

    # Recorrer cada camino de hormiga k
    for k in range(cantCaminos):

        if(metodo == 0):
            # Global
            denominador = distRecorridas[k]

        # Recorrer cada nodo recorrido por hormiga k
        for i in range(longCaminos - 1):
            # Extraer el par de indices
            idx_i = todoCaminoRecorrido[k, i]
            idx_j = todoCaminoRecorrido[k, i + 1]

            if(metodo == 2):
                # Local
                denominador = matrizCaminos[idx_i, idx_j]

            matrizFermona[idx_i, idx_j] += Q/denominador

    return matrizFermona

def camino_optimo(nombArch, cantHorm, itMax, tasaEvap, metodoActFermona, nodoInit):

    matrizCaminos = cargarDatos(nombArch)

    # Con el archivo gr17.csv vamos a tener una matriz de 17x17,
    # asi que la matriz de caminos y la matriz de feromonas seran de 17x17
    dim0 = matrizCaminos.shape[0]
    dim1 = matrizCaminos.shape[1]

    # Recordar que haya fermona por cada par de nodo i-j
    # y se inicializan con valores muy pequenos
    matrizFermona = np.random.rand(dim0, dim1) * 0.05

    diag = np.diag_indices_from(matrizCaminos)
    # Despejar la matriz Eta que es 1/dist
    matrizEta = matrizCaminos.copy()
    matrizEta[diag] = 1
    matrizEta = 1/matrizEta
    matrizEta[diag] = 0

    # Los indices de camino 
    idxCamino = np.arange(dim0)

    cantCaminoIgual = 0
    it = 0
    
    # Grafica
    producto = 100 # Para que los numeros sean mas visible
    _, ax = plt.subplots(figsize=(6, 6))
    title = "Inicio de feromonas"
    ax = graficarFeromona(ax, matrizFermona, title, producto)

    # Bucle mientras no se llege a un max de iteraciones y hasta que todas las hormigas se alinean 
    # durante "n" iteraciones consecutivas (explicado en las anotaciones) 
    while(it < itMax and cantCaminoIgual < 4):
        it += 1

        # Inicializar con True por cada iteracion
        caminoIgual = True

        # Las distancias y el camino recorrido por las k hormigas
        distRecorridas = np.zeros(shape=(cantHorm))
        todoCaminoRecorrido = np.empty(shape=(cantHorm, dim0 + 1), dtype=int)

        # (Sigma ij)^alpha * (Eta ij)^beta
        sigmaPorEta = matrizFermona * matrizEta

        # Recorrer cada hormiga
        for k in range(cantHorm):

            # Registrar el camino recorrido por la hormiga k
            caminoRecorrida = np.empty(shape=(dim0 + 1), dtype=int)
            caminoRecorrida[0] = nodoInit
            caminoRecorrida[-1] = nodoInit

            # idx boolean para descartar aquellos nodos ya visitados
            idxBool = np.full(shape=(dim0), fill_value=True, dtype=bool)
            idxBool[nodoInit] = False

            # Seleccion de caminos hasta el ultimo nodo disponible
            # Vuelve al inicio cuando sale del ciclo for
            for i in range(dim0 - 1):

                # El ultimo nodo que esta parado, al inicio es nodoInit
                idx_i = caminoRecorrida[i]
                idx_j = obtenerProximoNodo(idx_i, sigmaPorEta, idxBool, idxCamino)

                # Registrar el nodo, descartarlo de la lista y sumar en la distancia
                caminoRecorrida[i+1] = idx_j
                idxBool[idx_j] = False
                distRecorridas[k] += matrizCaminos[idx_i, idx_j]

            ultimoNodo = caminoRecorrida[-2]
            distRecorridas[k] += matrizCaminos[ultimoNodo, nodoInit]

            # Chequear si el camino recorrido es igual que el camino de la 
            # hormiga previa, si son diferentes ya no chequeo mas 
            if(caminoIgual and k != 0):
                caminoIgual = np.all(caminoRecorrida == todoCaminoRecorrido[k-1])
            
            todoCaminoRecorrido[k] = caminoRecorrida

        # Cuando termino de recorrer todas las hormigas, verifico si fueron todos los caminos iguales
        if(caminoIgual):
            cantCaminoIgual += 1
        else:
            cantCaminoIgual = 0

        # Actualizar la fermona segun la tasa de evaporacion
        matrizFermona = (1 - tasaEvap) * matrizFermona

        matrizFermona = depositarFermona(metodoActFermona, matrizFermona, matrizCaminos, todoCaminoRecorrido, distRecorridas)

        # Actualizar grafica de fermonas cada n iteraciones
        if(it % 25 == 0):
            title = f"Matriz fermona {producto}x, {it} iteraciones"
            ax = graficarFeromona(ax, matrizFermona, title, producto)
    
    # Cuando sale del bucle while
    if(itMax <= it):
        print("Finalizó por llegar al numero máximo de iteraciones.")
    else:
        print(f"Se logro alinear las hormigas por {cantCaminoIgual} veces.")
        print(f"En total {it} iteraciones.")

    # Buscamos la menor distancia recorrida y el mejor camino
    idxMejor = np.argmin(distRecorridas)
    mejorCamino = todoCaminoRecorrido[idxMejor]

    title = f"Finalizado en {it} iteraciones.\n Camino: {str(mejorCamino)} \n" 
    graficarCamino(ax, mejorCamino, title)

    # Retornar el camino con menor distancia recorrida
    return mejorCamino, distRecorridas[idxMejor]