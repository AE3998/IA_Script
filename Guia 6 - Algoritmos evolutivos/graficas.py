import numpy as np
import matplotlib.pyplot as plt

#* ----- Funcion f1 y su grafica para ver el minimo global -----
def grafica_f1():
    # Definir la funcion f1
    def f1(x):
        return -x * np.sin(np.sqrt(np.abs(x)))

    # Crear un rango de valores x en el intervalo [-512, 512]
    x = np.linspace(-512, 512, 1000)

    # Calcular los valores de f1 para cada valor de x
    y = f1(x)

    # Crear la grafica
    plt.figure()
    plt.plot(x, y, label='$f_1(x) = -x \cdot \sin(\sqrt{|x|})$')
    plt.xlabel('x')
    plt.ylabel('f1(x)')
    plt.title('Gráfica de f1(x)')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

#* ----- Funcion f2 y su grafica para ver el minimo global -----
def grafica_f2():
    # Definir la funcion f2
    def f2(val):
        x, y = val[0], val[1]  
        square = (x*x + y*y)
        res = square**0.25 * (np.sin(50 * square**0.1)**2 + 1)
        return res

    # Crear un rango de valores para x e y en el intervalo [-100, 100]
    x = np.linspace(-100, 100, 400)
    y = np.linspace(-100, 100, 400)

    # Crear una malla de valores x, y para calcular f2 en cada punto
    X, Y = np.meshgrid(x, y)
    Z = f2([X, Y])

    # Crear la grafica
    plt.figure()
    contour = plt.contourf(X, Y, Z, levels=100, cmap='viridis')
    plt.colorbar(contour, label='f2(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gráfica de f2(x, y)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Crear la grafica 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f2(x, y)')
    ax.set_title('Gráfica 3D de f2(x, y)')
    plt.show()