import numpy as np 
import matplotlib.pyplot as plt

#* ---------- Funcion inciso i) ----------
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

# x = np.linspace(-512, 512, 100)
# plt.plot(f1(x))
# plt.show()

#* i) f(x) = -x*sin(sqrt(abs(x))) 
#  con x = [-512, 512], necesito 10 bits para representar el rango 
#  de x, con esto alcanza para representar un individuo

cantIndividuo = 100
cromosoma = [10]
probMutacion = 0.1
probCruza = 0.8

#* ---------- Funcion inciso ii) ----------

def f2(val):
    x, y = val[0], val[1]   # x, y = [-100, 100]
    square = (x*x + y*y)
    res = square**0.25 * (np.sin(50 * square**0.1)**2 + 1)
    return res

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

# Ahora podriamos usar df2_dx y df2_dy en el metodo del gradiente descendente

#! Controlar si eso esta bien asi o si hay otra forma de hacerlo