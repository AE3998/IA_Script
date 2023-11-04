import matplotlib.pyplot as plt
import numpy as np

# =============[Sigmoidea]=============
def sigmoidea(Wji, Xi, alpha):

    Vi = Xi@Wji.T
    Y = 2/(1 + np.exp(-alpha * Vi)) - 1
    return Y

# =============[Grafica de XOR]=============
def crearLegendXOR(ax):
    # # Agregar legend
    col = ['r', 'k']
    desc = ['True', 'False']
    mark = ['s', 'x']

    handle = [(plt.Line2D([], [], color=col[i], label=desc[i], 
                marker=mark[i], linewidth=0)) for i in range(len(col))]
    ax.legend(handles=handle, loc='lower right')

def grafPuntosXOR(ax, X, Yd):
    idxYdPos = np.ravel(Yd > 0)
    ax.scatter(X[idxYdPos, 0], X[idxYdPos, 1], c='r', marker='s')
    ax.scatter(X[~idxYdPos, 0], X[~idxYdPos, 1], c='k', marker='x')

# =============[Grafica de Concent]=============

# def crearLegendConc(ax):
#     # fake_blue, fake_red, blue, red
#     colores = ["#00EEEE", "#EE4000", "#0000FF", "#FF0000"]
#     legend = ["F-neg", "F-pos", "V-neg", "V-pos"]
#     n = len(colores)
#     handle = [(plt.Line2D([], [],
#                           color = colores[i], label=legend[i],
#                           marker="o", linewidth=0)) for i in range(n)]
#     # Agregar los legend en axes
#     legend1 = ax.legend(handles = handle[:n//2], loc = "upper left")
#     ax.add_artist(legend1)
#     ax.legend(handles = handle[n//2:], loc = "upper right")

# def grafPuntosConc(ax, X, Yd):
#     colores = np.full(shape=Yd.shape[0], fill_value= "#EE4000", dtype='U7')
#     colores[np.ravel(Yd)<0] = "#00EEEE"
#     return ax.scatter(X[:, 0], X[:, 1], c=colores)

def crearLegendConc(ax):
    # # Agregar legend
    col = ['r', 'k']
    desc = ['True', 'False']
    mark = ['s', 'x']

    handle = [(plt.Line2D([], [], color=col[i], label=desc[i], 
                marker=mark[i], linewidth=0)) for i in range(len(col))]
    ax.legend(handles=handle, loc='lower right')

def grafPuntosConc(ax, X, Yd):
    idxYdPos = np.ravel(Yd > 0)
    ax.scatter(X[idxYdPos, 0], X[idxYdPos, 1], c='r', marker='s')
    ax.scatter(X[~idxYdPos, 0], X[~idxYdPos, 1], c='k', marker='x')


# =============[Grafica de Mesh]=============
# def initMesh(ax, n):
#     borde = 1.2
#     x0 = np.linspace(-borde, borde, n)
#     x1 = np.linspace(-borde, borde, n)
#     A = np.zeros(shape=(n, n))
#     return ax.pcolormesh(x0, x1, A, cmap='coolwarm')

def actualizarMesh(ax, X, Yd, Wji, alpha, nMesh, XOR, epoca, err):
    bordeMin = -1.2
    bordeMax = 1.2

    if not(XOR):
        bordeMin = -0.1
        bordeMax = 1.1

    x0 = -1 * np.ones(shape=(nMesh, 1))       # sesgo [-1; -1; -1]
    x2 = np.linspace(bordeMin, bordeMax, nMesh)  
    x2 = np.transpose([x2])                 # entrada [1; 2; 3]

    A = np.empty(shape=(nMesh, nMesh))
    # Recorrer cada punto de mesh, en este caso en forma
    # matricial por columna
    for idx, val in enumerate(x2):
        x1 = val * np.ones(shape=(nMesh, 1))   # 1er entreada  [ 1;  1;  1]

        entrada = np.hstack((x0, x1, x2))
        # entrada = [-1 1 1;
        #            -1 1 2;
        #            -1 1 3]
        Ytemp = 0
        for W in Wji:
            Ytemp = sigmoidea(W, entrada, alpha)
            # if(len(Ytemp.shape) == 1): break
            entrada = np.hstack((x0, Ytemp))

        A[:, idx] = np.ravel(Ytemp)

    # Actualizar Mesh
    x0 = np.linspace(bordeMin, bordeMax, nMesh)
    ax.pcolormesh(x0, x0, A, cmap='coolwarm')
    
    # Graficar los puntos y asignar legend
    X_temp = X[:, 1:]
    if XOR:
        grafPuntosXOR(ax, X_temp, Yd)
        crearLegendXOR(ax)
    else:
        grafPuntosConc(ax, X_temp, Yd)
        crearLegendConc(ax)

    title = "Epoca " + str(epoca) + \
            " Acierto: " + str(round((1 - err) * 100, 3)) + "%"
    ax.set_title(title)
    # Ver si hay que retornar eso
            


# =============[inicializar grafica]=============

def initGraf(title, X, Yd, nMesh, XOR):
    fig, ax = plt.subplots(layout='constrained')

    
    bordeMin = -1.2
    bordeMax = 1.2
    if not(XOR):
        bordeMin = -0.1
        bordeMax = 1.1

    ax.axis([bordeMin, bordeMax, bordeMin, bordeMax])
    ax.axis('equal')
    ax.set_title(title)

    X_temp = X[:, 1:]
    # Graficar mesh
    # mesh = initMesh(ax, nMesh)

    x0 = np.linspace(bordeMin, bordeMax, nMesh)
    A = np.zeros(shape=(nMesh, nMesh))

    ax.pcolormesh(x0, x0, A, cmap='coolwarm')
    # Graficar los puntos y asignar legend
    if XOR:
        grafPuntosXOR(ax, X_temp, Yd)
        crearLegendXOR(ax)
    else:
        grafPuntosConc(ax, X_temp, Yd)
        crearLegendConc(ax)

    return fig, ax


