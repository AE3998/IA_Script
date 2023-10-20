import numpy as np

def perceptronSimple(X, W):
    """
    Perceptron simple. 
    Entrada: vector de entradas "X" y vector pesos "W".
    Salida: +1 si el producto interno entre las entradas y pesos es mayor o igual a 0, sino -1.
    """
    # np.inner calcula el producto interno entre dos matrices: np.inner(a, b) = sum(a[:]*b[:])
    return 1 if (np.inner(X, W) >= 0) else -1
    
    ## La idea era calcular el producto interno entre salidas y pesos, eso nos da la salida lineal (z),
    ## y se le aplica una funcion de activacion no lineal para obtener la salida no lineal o final (y).
    ## Aca ya lo hace todo junto, porque devolver +1 o -1 seria el equivalente a aplicar la funcion 
    ## signo (funcion de activacion no lineal) al resultado obtenido del producto interno