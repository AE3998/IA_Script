import numpy as np
import matplotlib.pyplot as plt


def iniciarGrafica(func, xmin, xmax, puntosEntrada):

    if(xmin.shape[0] == 1):
        ax = plt.subplot()
        nums = 300
        x = np.linspace(xmin, xmax, nums)
        x = np.ravel(x)
        y = func(x)

        ax.plot(x, y)
        y = func(puntosEntrada)
        puntos = ax.scatter(puntosEntrada, y, c='k', linewidths=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Gráfica 2D de f1(x)')
        ax.grid(True)
    
    else:
        
        def f2(val):
            x, y = val[0], val[1]
            square = (x*x + y*y)
            res = square**0.25 * (np.sin(50 * square**0.1)**2 + 1)
            return res

        ax = plt.figure().add_subplot(projection='3d')
        nums = 80
        x = np.linspace(start=xmin[0], stop=xmax[0], num=nums)
        y = np.linspace(start=xmin[1], stop=xmax[1], num=nums)
        X, Y = np.meshgrid(x, y)
        Z = f2((X, Y))

        # val = np.column_stack((X.ravel(), Y.ravel()))
        # Z = func(val)

        # Graficar la superficie
        ax.plot_surface(X, Y, Z, cmap='rainbow', alpha=0.2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f2(x, y)')
        ax.set_title('Gráfica 3D de f2(x, y)')

        # Graficar los puntos
        X = puntosEntrada[:, 0]
        Y = puntosEntrada[:, 1]
        # Z = func((X, Y))

        val = np.column_stack((X, Y))
        Z = func(val)


        puntos = ax.scatter(X, Y, Z, c='k', linewidth=2)
    
    plt.pause(0.5)

    return ax, puntos


def actualizarGrafica(func, dim, puntosEntrada, ax, puntos, title="Funcion"):

    ax.set_title(title)
    if(dim == 1):
        y = func(puntosEntrada)

        offset = np.column_stack((puntosEntrada, y))
        puntos.set_offsets(offset)
    
    else:
        X, Y = puntosEntrada[:, 0], puntosEntrada[:, 1]
        Z = func(puntosEntrada)

        # ax.scatter(X, Y, Z)
        puntos._offsets3d = (X, Y, Z)
    
    plt.pause(0.2)
    return puntos

