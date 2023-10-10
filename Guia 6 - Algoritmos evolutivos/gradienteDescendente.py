import numpy as np

def gradienteDescendente(grad_func, xmin, xmax):
    """
        Metodo de gradiente descendente para buscar el minimo global de una funcion.
        Entradas: derivada de la funcion (grad_func) y limites del rango [xmin, xmax]
        Salida: minimo global (x)
    """
    # Parametros del algoritmo
    alpha = 0.1  # tasa de aprendizaje
    maxIteraciones = 1000  # numero maximo de iteraciones
    tolerancia = 1e-6  # tolerancia para la convergencia

    # Limites del rango [xmin, xmax]
    xmin = np.array(xmin)
    xmax = np.array(xmax)
    # Iniciamos con un punto al azar en el rango dado
    x = np.random.rand(np.size(xmin)) * (xmax-xmin) + xmin

    # Gradiente descendente
    for i in range(maxIteraciones):
        grad = grad_func(x)  # Calcular el gradiente en x
        x_new = x - alpha * grad  # Actualizar x usando el gradiente descendente
        
        # Verificar y ajustar si es necesario, para mantener x dentro del rango [xmin, xmax]
        x_new = max(xmin, min(x_new, xmax))

        # Verificar la convergencia
        if abs(x_new - x) < tolerancia:
            break
        
        x = x_new  # Actualizar x para la siguiente iteración

    return x