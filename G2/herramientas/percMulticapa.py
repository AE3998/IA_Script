import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def cargarDatos(nombreArchivo, nCapaFinal):
    datos = np.genfromtxt(nombreArchivo, delimiter=",", max_rows=None)

    # Definir la condicion del caso que la capa final no sea unico
    xData = datos[:, :-nCapaFinal]
    yData = datos[:, -nCapaFinal:]
    
    # Definir la columna de -1 de la entrada x0
    colX0 = -1 * np.ones(shape=(np.size(xData, 0), 1))
    xData = np.hstack((colX0, xData))

    return xData, yData

def sigmoidea(Wji, Xi, alpha):
    """
    Entrada:
        Wji = pesos sinapticos 
        Xi = entrada de la neurona 
    Salida:
        Y = vector de num = [-1, 1]
    """
    Vi = Wji@Xi
    Y = 2/(1 + np.exp(-alpha * Vi)) - 1

    return Y


def entrenar(nombreArchivo, capas, alpha,tasaAp, 
             maxErr, maxEpoc, umbral = 1e-1, graf = False):
    
    np.random.seed(10000)
    # ===========[Inicializar los datos]===========
    
    # Pasar la ruta del archivo y la cantidad de neuronas de la capa de salida
    # Recibe 2 matrices 2D: Los patrones de entrada y las salidas esperadas 
    # como vector columna
    X, Yd = cargarDatos(nombreArchivo, capas[-1])

    cantCapas = len(capas)

    ##### Asumo que la entrada debe ser al menos 2D======
    nEntrada = X.shape[1] # columna
    
    # =========[Inicializar los matrices de pesos]=========
    # La cantidad de columnas SIEMPRE debe coincidir con la cantidad de entrada Xi (o Yi)
    Wji = [np.random.rand(capas[0], nEntrada) - 0.5]

    for i in range(cantCapas - 1):
        # Numero de columna debe coincidir con la cantidad de entradas
        # Recordar que habia X0 como entrada adicional, por eso se suma 1
        # a la cantidad de columna
        Wji.append(np.random.rand(capas[i + 1], capas[i] + 1) - 0.5)
    
    # =========[Inicializar las salidas y las deltas de cada capa]=========
    yy = []
    deltas = []

    for i in capas:
        yy.append(np.empty(i))
        deltas.append(np.empty(i))
    
    # ===========[Ciclo de entrenamiento]===========

    err = 1
    epoca = 0
    errPlot = np.array([])
    
    pbar = tqdm(total=maxEpoc)

    while(err > maxErr and epoca < maxEpoc):
        
        description = "Epoca " + str(epoca + 1) + ":"
        pbar.set_description(description)
        pbar.update(1)
        
        epoca += 1
        
        # Recorrer todo los patrones de entrada 
        for i in range(X.shape[0]):
            # ===========[Propagacion hacia adelante]===========

            entrada = X[i, :]

            for j in range(cantCapas):
                yy[j] = sigmoidea(Wji[j], entrada, alpha)
                
                # Recordar la entrada X0 que existe en cada capa 
                entrada = np.hstack((np.array([-1]), yy[j]))
            
            # ===========[Propagacion hacia atras]===========

            # Asignar directamente la delta de la capa de salida
            ySalida = yy[-1][:]
            deltas[-1] = 0.5 * (Yd[i, :] - ySalida) * (1 + ySalida) * (1 - ySalida)

            for j in range(cantCapas - 2, -1, -1):
                # Recordar que cada columna de Wji corresponde a una entrada particular
                # Por ejemplo Wj1 corresponde a cada peso sinaptico que conecta de
                # X1 hacia las j entradas.
                

                # delta de capa 1 = 1/2(dta2@Wji2)(1+y1)(1-y1)
                # d1 = 1/2([0.11]@[0.1, 0.2])(1+y2[])
                # Por un problema tecnico, debo agregar una dimension al vector 
                # de delta y pasarlo a un vector columna

                # [dta11; dta12] .* [1 1; 1 1] = [dta11 dta11 ; dta12 dta12] 
                # Y luego sumarlo, dando un vector 1D para asignar a las deltas
                # de la capa -.-

                # dTa = deltas[j + 1]
                # producto = np.multiply(dTa[:, np.newaxis], Wji[j + 1][:, 1:])
                # sumatoria = np.sum(producto, axis=0)

                sumatoria = (deltas[j + 1][np.newaxis]@Wji[j + 1][:, 1:])[0]

                ySalida = yy[j][:]
                deltas[j] = 0.5 * sumatoria * (1 + ySalida) * (1 - ySalida)


            # ===========[Ajuste de pesos]===========
            
            # Recorrer nuevamente cada capa ajustando los pesos
            # delta*entrada = 
            # vector columna(cant de delta) * vector fila(cant de entrada) 
            # dWji = matriz shape = (nNeurona, nEntrada)
            # Recordar que delta guarda solamente vector 1D
            # Lo mismo sucede cuando hago slicing de X[i, :] devuelve
            # un vector 1D. Hay que agregar dimension para que sean 2D
            # y aplicar producto matricial 

            # Delta Wji de capa 1
            dWji = tasaAp * deltas[0][:, np.newaxis] @ X[i, :][np.newaxis]

            for j in range(cantCapas - 1):
                Wji[j] += dWji 
                entradaCapa =  np.hstack((np.array([-1]), yy[j]))
                dWji = tasaAp * deltas[j + 1][:, np.newaxis] @ entradaCapa[np.newaxis]

            # Guardar el error del patron
            errPlot = np.hstack((errPlot, 0.5 * np.linalg.norm(Yd[i, :] - yy[-1][:], 2)))
            
            # =======Fin del patron=======
        plt.plot(errPlot, label="Err/Patron")
        plt.grid(True)
        plt.legend()

        # Una vez terminado todos los patrones, se procede a comprobar  
        # la tasa de acierto

        # ===========[Comprobar acierto]===========
        cantErr = 0
        
        # Recorrer cada patron de entrada y despejar su salida y
        for i in range(X.shape[0]):
            entrada = X[i, :]
            for j in range(cantCapas):
                yy[j] = sigmoidea(Wji[j], entrada, alpha)
               # Recordar la entrada X0 que existe en cada capa 
                entrada = np.hstack((np.array([-1]), yy[j]))
            
            # Comparar si el error del patron supera o no 
            # el umbral definido
            E = 0.5 * np.linalg.norm(Yd[i, :] - yy[-1][:], 2)
            cantErr += E > umbral

        err = cantErr/X.shape[0]

        # print(err*100, "%", " de error")
    
    pbar.close()
    if(err < maxErr):
        print("Entrenamiento finalizado por tasa de acierto " +
                str((1 - err) * 100) + "%")  
    else:
        print("Entrenamiento finalizado por Maxima Epoca con ", 
              "un tasa de error ", err * 100, "%")

    return Wji

            

def probar(nombreArchivo, Wji, alpha, umbral = 1e-1, graf = False):

    X, Yd = cargarDatos(nombreArchivo, Wji[-1].shape[0])
    

    # ===========[Comprobar acierto]===========
    cantErr = 0
    cantCapas = len(Wji)
    
    yy = []
    for i in range(cantCapas):
        yy.append(np.empty(Wji[i].shape[0]))
    
    # Recorrer cada patron de entrada y despejar su salida y
    # En = 0
    for i in range(X.shape[0]):
        entrada = X[i, :]
        for j in range(cantCapas):
            yy[j] = sigmoidea(Wji[j], entrada, alpha)
            # Recordar la entrada X0 que existe en cada capa 
            entrada = np.hstack((np.array([-1]), yy[j]))
        
        # Comparar si el error del patron supera o no 
        # el umbral definido
        E = 0.5 * np.linalg.norm(Yd[i, :] - yy[-1][:], 2)
        # En += 0.5 * np.linalg.norm(Yd[i, :] - yy[-1][:], 2)
        cantErr += E > umbral


    # Despejar el porcentaje de error 
    err = cantErr/X.shape[0]

    print("Prueba con tasa de acierto " + str((1 - err) * 100) + "%")

    # Verificar los datos
    # for i in Wji:
    #     print(i)

    # fig, ax = plt.subplots()
    # crearLegend(ax)
    # plt.show()





# capaFinal = 1
# X, Y = cargarDatos("datos/XOR_trn.csv", capaFinal)
# print(X)
# print(Y)
