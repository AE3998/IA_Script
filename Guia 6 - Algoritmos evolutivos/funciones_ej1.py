import numpy as np
from graficas import *
from gradienteDescendente import *

# todo ===================[funcion 1]===================
def f1(x):
    return -x*np.sin(np.sqrt(np.abs(x)))   # x = [-512, 512]

# Derivada de la funcion f1 (gradiente)
def df1(x):
    if x > 0:
        return -np.sin(np.sqrt(x)) - (x / (2 * np.sqrt(x))) * np.cos(np.sqrt(x))
    elif x < 0:
        return -np.sin(np.sqrt(-x)) + (x / (2 * np.sqrt(-x))) * np.cos(np.sqrt(-x))
    else:
        return 0  # la derivada en x=0 es 0

# Grafica de la funcion
# grafica_f1()

#? Resultado con metodo de gradiente descendente 
#? (se puede jugar con los parametros del metodo de gradiente descentende)
# Hay veces que llega o se acerca al minimo global viendo la grafica, pero debido a la inicializacion
# al azar otras veces cae en minimos locales. Si en la inicializacion del gradienteDescendente
# se le pone x = 400 por ejemplo, ahi si siempre llega al minimo global cerca de x = 420

# x_min_f1 = gradienteDescendente(df1, -512, 512)
# print("Minimo global de f1:", x_min_f1)
# print("Valor minimo f1(x) =", f1(x_min_f1))

# todo ===================[funcion 2]===================
def f2(val):
    x, y = val[0], val[1]   # x, y = [-100, 100]
    square = (x*x + y*y)
    res = square**0.25 * (np.sin(50 * square**0.1)**2 + 1)
    return res

def df2(val):
    x, y = val[0], val[1]
    square = x*x + y*y
    adentro = 50*(square**0.1)

    factor1 = 0.5*(np.sin(adentro)**2 + 1) * square**-0.75
    factor2 = 20 * np.sin(adentro) * np.cos(adentro) * square**-0.65

    base = factor1 + factor2

    dx = x*base
    dy = y*base

    return np.array([dx, dy])

# Grafica de la funcion
# grafica_f2()

# Derivadas parciales de f2 con respecto a x e y
def df2_dx(val):
    x, y = val[0], val[1]
    square = x*x + y*y
    factor1 = (square**(-3/4)) * (2*x) * np.sin(50 * (square**0.1))**2
    factor2 = (50/10) * (square**(-0.9)) * (2*x*y) * np.cos(50 * (square**0.1))
    return 0.25 * factor1 * factor2

def df2_dy(val):
    x, y = val[0], val[1]
    square = x*x + y*y
    factor1 = (square**(-3/4)) * (2*y) * np.sin(50 * (square**0.1))**2
    factor2 = (50/10) * (square**(-0.9)) * (2*x*y) * np.cos(50 * (square**0.1))
    return 0.25 * factor1 * factor2

#! ¿Ahora podriamos usar df2_dx y df2_dy en el metodo del gradiente descendente?
#! FALTA MODIFICAR EL ALGORITMO gradienteDescendiente.py PARA QUE FUNCIONE PASANDO [-100, 100]
#? Resultado con metodo de gradiente descendente 

# x_min_f2 = gradienteDescendente(df2_dx, [-100, 100])
# print("Minimo global de f1 respecto a x:", x_min_f2)
# y_min_f2 = gradienteDescendente(df2_dy, [-100, 100])
# print("Minimo global de f1 respecto a y:", y_min_f2)

xmin = [-100, -100]
xmax = [100, 100]
# print(xmin)
# print(xmax)
min_f2, xInit = gradienteDescendente(df2, xmin, xmax)
print(f"Inicializado en: {xInit}")
print("Minimo global de f2:", min_f2)

ax = grafica_f2()
puntos = agregar_puntos_graf_f2(ax, xInit)
plt.pause(2)
# agregar_puntos_graf_f2(ax, min_f2)
actualizar_graf_f2(puntos, min_f2)

plt.show()
#! Controlar si todo eso de las derivadas esta bien asi o si hay otra forma de hacerlo