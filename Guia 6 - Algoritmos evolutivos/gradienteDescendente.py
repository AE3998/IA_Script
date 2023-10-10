import numpy as np

def gradienteDescendente(grad_func, xmin, xmax):
    """
        Metodo de gradiente descendente para buscar el minimo global de una funcion.
        Entradas: derivada de la funcion (grad_func) y limites del rango [xmin, xmax]
        Salida: minimo global (x)
    """

    # Parametros del algoritmo
    alpha = 0.01  # tasa de aprendizaje
    maxIteraciones = 5000  # numero maximo de iteraciones
    tolerancia = 1e-6  # tolerancia para la convergencia

    # Limites del rango [xmin, xmax]
    xmin = np.array(xmin)
    xmax = np.array(xmax)

    # Iniciamos con un punto al azar en el rango dado (queda en el rango [xmin, xmax))
    x = np.random.uniform(xmin, xmax)
    # x = 400     # para probar que la funcion f1 llegue al minimo global cerca de x = 420

    # Gradiente descendente
    for _ in range(maxIteraciones):
        grad = grad_func(x)  # calcular el gradiente en x
        x_new = x - alpha * grad  # actualizar x usando el gradiente descendente
        
        # Verificar y ajustar si es necesario, para mantener x dentro del rango [xmin, xmax]
        x_new = max(xmin, min(x_new, xmax))

        # Verificar la convergencia
        if abs(x_new - x) < tolerancia:
            break
        
        x = x_new  # actualizar x para la siguiente iteracion

    return x    # devuelve el minimo global obtenido