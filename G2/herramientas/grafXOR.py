import matplotlib.pyplot as plt
import numpy as np

def grafPuntosXOR(X, Y, ax):
    #  
    ax.axis([-1.5, 1.5, -1.5, 1.5])

    # Asignar color
    colores = np.full(shape=Y.shape[0], fill_value="#FF0000", dtype='U7')
    colores[np.ravel(Y)<0] = "#000000"
    # Graficar
    ax.scatter(X[:, 0], X[:, 1], c = colores)
    
    # Agregar legend
    col = ['r', 'k']
    desc = ['True', 'False']
    handle = [(plt.Line2D([], [], color=col[i], label=desc[i], 
                marker='o', linewidth=0)) for i in range(len(col))]
    ax.legend(handles=handle)

def grafErrorXOR(ax, Err, epoca):
    # fig, ax = plt.subplots()
    ax.cla()
    ax.grid(True)
    ax.plot(Err, label="Error/Epoca"+str(epoca))
    ax.legend()
    ax.set_title("Epoca" + str(epoca))
    plt.pause(0.1)



# X = np.array([[1, 1], [-1, 1], [-1, -1]])
# Y = np.array([[-1], [1], [-1]])
# print(Y<0)
# fig, ax = plt.subplots(layout='constrained')

# grafPuntos(X, Y, ax)
# plt.show()
