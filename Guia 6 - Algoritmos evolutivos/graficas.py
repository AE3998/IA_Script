import numpy as np
import matplotlib.pyplot as plt
# from funciones_ej1 import *

def grafF1(x):
    return -x*np.sin(np.sqrt(np.abs(x)))   # x = [-512, 512]

def grafF2(val):
    x, y = val[0], val[1]   # x, y = [-100, 100]
    square = (x*x + y*y)
    res = square**0.25 * (np.sin(50 * square**0.1)**2 + 1)
    return res

#* ----- Funcion f1 y su grafica para ver el minimo global -----
def grafica_f1():
    # Crear un rango de valores x en el intervalo [-512, 512]
    x = np.linspace(-512, 512, 1000)

    # Calcular los valores de f1 para cada valor de x
    y = grafF1(x)

    # Crear la grafica
    # plt.figure()
    fig, ax = plt.subplots(layout='constrained')
    ax.plot(x, y, label='$f_1(x) = -x \cdot \sin(\sqrt{|x|})$')
    ax.set_xlabel('x')
    ax.set_ylabel('f1(x)')
    ax.set_title('Gráfica de f1(x)')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    fig.show()

    # plt.show()
    return ax


#* ----- Funcion f2 y su grafica para ver el minimo global -----
def grafica_f2():

    # Crear un rango de valores para x e y en el intervalo [-100, 100]
    x = np.linspace(-100, 100, 400)

    # Crear una malla de valores x, y para calcular f2 en cada punto
    X, Y = np.meshgrid(x, x)
    val = (X, Y)
    Z = grafF2(val)

    # Crear la grafica
    plt.figure()
    contour = plt.contourf(X, Y, Z, levels=100, cmap='rainbow')
    plt.colorbar(contour, label='f2(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gráfica de f2(x, y)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Crear la grafica 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='rainbow', alpha=0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f2(x, y)')
    ax.set_title('Gráfica 3D de f2(x, y)')
    return ax

def agregar_puntos_graf_f2(ax, pobDecod):
    if(len(pobDecod.shape) > 1):
        X = pobDecod[:, 0]
        Y = pobDecod[:, 1]
    else:
        X = pobDecod[0]
        Y = pobDecod[1]

    Z = grafF2((X, Y))
    puntos = ax.scatter(X, Y, Z, c='k', linewidth = 2)

    plt.pause(0.2)

    return puntos

def actualizar_graf_f2(puntos, pobDecod):
    if(len(pobDecod.shape) > 1):
        X = pobDecod[:, 0]
        Y = pobDecod[:, 1]
        Z = grafF2((X, Y))
    else:
        X = pobDecod[0]
        Y = pobDecod[1]
        Z = grafF2((X, Y))
        # Hay cuestion de indexado, habria que convertirle a una matriz
        X = np.array([X])
        Y = np.array([Y])
        Z = np.array([Z])

    puntos._offsets3d = (X, Y, Z)
    plt.pause(0.2)

#? test
# grafica_f1()
# grafica_f2()
# plt.show()
