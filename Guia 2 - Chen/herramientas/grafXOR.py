import matplotlib.pyplot as plt
import numpy as np

def grafPuntosXOR(X, Y, ax):
    #  
    ax.axis([-1.5, 1.5, -1.5, 1.5])

    idxYdNegativo = np.ravel(Y)<0
    # Asignar color
    colores = np.full(shape=Y.shape[0], fill_value="#FF0000", dtype='U7')
    colores[idxYdNegativo] = "#000000"

    # # Asignar markers
    # markers = np.full(shape=Y.shape[0], fill_value='s', dtype='U7')
    # markers[np.ravel(Y)<0] = 'X'

    ax.scatter(X[:, 0], X[:, 1], c=colores)
    # Graficar
    # neg = ax.scatter(X[idxYdNegativo, 0], X[idxYdNegativo, 1], marker='x')
    # pos = ax.scatter(X[~idxYdNegativo, 0], X[~idxYdNegativo, 1], marker='s')
    

    # neg.set_color = "#000000"
    # pos.set_color = "#FF0000"
    
    # # Agregar legend
    col = ['r', 'k']
    desc = ['True', 'False']
    mark = ['s', 'x']

    handle = [(plt.Line2D([], [], color=col[i], label=desc[i], 
                marker=mark[i], linewidth=0)) for i in range(len(col))]
    ax.legend(handles=handle)

def grafErrorXOR(ax, Err, epoca):
    # fig, ax = plt.subplots()
    ax.cla()
    ax.grid(True)
    ax.plot(Err, label="Error/Epoca"+str(epoca))
    ax.legend()
    ax.set_title("Epoca" + str(epoca))
    plt.pause(0.05)


    

def actualizarColor(ax, X, Yd, Y):

    return

X = np.array([[1, 1], [-1, 1], [-1, -1]])
Y = np.array([[-1], [1], [-1]])
print(Y<0)
fig, ax = plt.subplots(layout='constrained')

grafPuntosXOR(X, Y, ax)
plt.show()
