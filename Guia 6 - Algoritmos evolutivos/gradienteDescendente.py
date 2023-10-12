import numpy as np

def gradienteDescendente(grad_func, xmin, xmax, xInit):
    """
        Metodo de gradiente descendente para buscar el minimo global de una funcion.
        Entradas: derivada de la funcion (grad_func) y limites del rango [xmin, xmax]
        Salida: minimo global (x). Recordar que dependiendo de la inicializacion al azar  
        va a converger al minimo local mas cercano.
    """

    # Parametros del algoritmo
    alpha = 0.1  # tasa de aprendizaje
    maxIteraciones = 5000  # numero maximo de iteraciones
    tolerancia = 1e-5  # tolerancia para la convergencia

    #? Limites del rango [xmin, xmax]
    # xmin = np.array(xmin)
    # xmax = np.array(xmax)

    #? # Iniciamos con un punto al azar en el rango dado (queda en el rango [xmin, xmax))
    # x = np.random.uniform(low=xmin, high=xmax)
    # xInit = np.copy(x)
    # # x = 400     # para probar que la funcion f1 llegue al minimo global cerca de x = 420 de la f1

    x = xInit
    if not(isinstance(x, np.ndarray)):  
        x = np.array([x])   # para convertirlo en un array de Numpy si no lo es

    # Gradiente descendente
    for _ in range(maxIteraciones):
        grad = grad_func(x)  # calcular el gradiente en x
        x_new = x - alpha * grad  # actualizar x usando el gradiente descendente
        
        # Verificar y ajustar si es necesario, para mantener x dentro del rango [xmin, xmax]
        x_new = np.max([xmin, np.min([xmax, x_new], axis=0)], axis=0)

        # Verificar la convergencia
        if np.linalg.norm(x_new - x) < tolerancia:
            break
        
        x = x_new  # actualizar x para la siguiente iteracion

    return x  # devuelve el minimo global obtenido 