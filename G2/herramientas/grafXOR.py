import matplotlib.pyplot as plt
import numpy as np

def grafPuntos(X, Y, ax):
    colores = []
    for i in Y:
        # print(i)
        colores.append('#FF0000' if(i>0) else '#000000')
    # print(colores)

    ax.scatter(X[:, 0], X[:, 1], c = colores)
    col = ['r', 'k']
    desc = ['True', 'False']
    handle = [(plt.Line2D([], [], color=col[i], label=desc[i], 
                marker='o', linewidth=0)) for i in range(len(col))]
    ax.legend(handles=handle)


X = np.array([[1, 1], [-1, 1], [-1, -1]])
Y = np.array([[-1], [1], [-1]])
fig, ax = plt.subplots(layout='constrained')

grafPuntos(X, Y, ax)
plt.show()
