import matplotlib as plt
import numpy as np

def asignarColor(Yd):
    colores = np.full(shape=Yd.shape[0], fill_value= "#EE4000", dtype='U7')
    colores[np.ravel(Yd)<0] = "#00EEEE"
    return colores

def crearLegend(ax):
    # fake_blue, fake_red, blue, red
    colores = ["#00EEEE", "#EE4000", "#0000FF", "#FF0000"]
    legend = ["F-neg", "F-pos", "V-neg", "V-pos"]
    n = len(colores)
    handle = [(plt.Line2D([], [],
                          color = colores[i], label=legend[i],
                          marker="o", linewidth=0)) for i in range(n)]
    # Agregar los legend en axes
    legend1 = ax.legend(handles = handle[:n//2], loc = "upper left")
    ax.add_artist(legend1)
    ax.legend(handles = handle[n//2:], loc = "upper right")

def grafPuntosConc(ax, X, Yd):
    colores = asignarColor(Yd)
    return ax.scatter(X[:, 0], X[:, 1], c=colores)