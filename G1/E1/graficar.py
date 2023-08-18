import matplotlib.pyplot as plt
import numpy as np

# Funcion que me ayuda a graficar las cosas
def iniciarRecta(Wi, ax):
    x = np.linspace(-1.1, 1.1, 10)
    y = Wi[0]/Wi[2] - Wi[1]/Wi[2] * x
    return ax.plot(x, y, 'b')[0]

def actualizarRecta(Wi, recta):
    x = np.linspace(-1.1, 1.1, 10)
    y = Wi[0]/Wi[2] - Wi[1]/Wi[2] * x
    recta.set_xdata(x)
    recta.set_ydata(y)
    plt.pause(0.2)

def graficarPuntos(Xi, ax):
    x = Xi[:, 1]
    y = Xi[:, 2]
    ax.plot(x, y, 'ro', linewidth = 0.2)

def iniciarGrafica(Xi):
    fig, ax = plt.subplots(layout = 'constrained')
    # Delimitar el borde
    ax.axis([-1.5, 1.5, -1.5, 1.5])
    ax.grid(True)

    # Graficar los ejes
    ax.plot([-1.1, 1.1], [0, 0],'k', linewidth = 3)
    ax.plot([0, 0], [-1.1, 1.1], 'k', linewidth = 3)
    
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    
    # Graficar los patrones de entrada
    graficarPuntos(Xi, ax)

    return fig, ax