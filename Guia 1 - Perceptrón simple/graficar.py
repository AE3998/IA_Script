import matplotlib.pyplot as plt
import numpy as np

# Funciones para graficar (ejercicio 2)

# x1w1 + x2w2 – w0 = 0
# x2 = w0/w2 – x1*w1/w2
# Recta de pendiente negativa w1/w2 y ordenada al origen w0/w2.
# Recta: y = W[0]/W[2] - W[1]/W[2]*x     
# Recordar que y = x2, lo cambie para que sea mas intuitivo nada mas

# linspace de Numpy recibe: inicio, fin, cantidad de subintervalos.
# np.linspace(0, 1, 10)
# Devuelve array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Generar recta solucion del problema, la cual separa las categorias de datos
def generarRecta(W, ax):
    x = np.linspace(-1.5, 1.5, 10)  # para tener una recta de -1.5 hasta 1.5
    y = W[0]/W[2] - W[1]/W[2] * x
    return ax.plot(x, y, 'b')[0]

# Luego de actualizar los pesos (W) en el entrenamiento se actualiza la recta
def actualizarRecta(W, recta):      
    x = np.linspace(-1.5, 1.5, 10)
    y = W[0]/W[2] - W[1]/W[2] * x
    recta.set_xdata(x)
    recta.set_ydata(y)
    plt.pause(0.2)

# Graficar nube de puntos (patrones de entrada)
def graficarNubePuntos(X, ax):      
    x = X[:, 1]
    y = X[:, 2]
    ax.plot(x, y, 'ro', linewidth = 0.2)

# subplots() sin argumentos devuelve una figura y sus ejes (ax). Sino se le puede indicar mas figuras (en fila o columna)
# layout = 'constrained' en subplots ajusta automaticamente los ejes y titulos y demas para que entren bien en la figura
def iniciarGrafica(X):
    (fig, ax) = plt.subplots(layout = 'constrained')    
    ax.axis([-1.5, 1.5, -1.5, 1.5])     # limite de los ejes
    ax.grid(True)

    # Graficar los ejes
    ax.plot([-1.5, 1.5], [0, 0],'k', linewidth = 2)
    ax.plot([0, 0], [-1.5, 1.5], 'k', linewidth = 2)
    
    ax.set_title("Resolución del problema")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    
    # Graficar los patrones de entrada (nube de puntos)
    graficarNubePuntos(X, ax)
    
    return (fig, ax)