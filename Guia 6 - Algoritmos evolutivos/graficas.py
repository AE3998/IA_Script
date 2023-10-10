import matplotlib.pyplot as plt
import numpy as np

#* Funcion f1 y su grafica para ver el minimo global
def grafica_f1():
    # Definir la funcion f1
    def f1(x):
        return -x * np.sin(np.sqrt(np.abs(x)))

    # Crear un rango de valores x en el intervalo [-512, 512]
    x = np.linspace(-512, 512, 1000)

    # Calcular los valores de f1 para cada valor de x
    y = f1(x)

    # Crear la grafica
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='$f_1(x) = -x \cdot \sin(\sqrt{|x|})$')
    plt.xlabel('x')
    plt.ylabel('f1(x)')
    plt.title('Gráfica de f1(x)')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()