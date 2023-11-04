import numpy as np

# Para probar como desordenar las filas de una matriz, para luego utilizarlo para desordenar
# las filas del archivo de datos, por si vienen ordenados primero los de una categoria y luego 
# la otra, para que el percetrón no se aprenda primero una categoria y luego la otra.

# Matriz de ejemplo
matriz = np.array([[9, 8, 7],
                   [6, 5, 4],
                   [3, 2, 1]])

# Obtener un índice aleatorio para permutar las filas
indices_aleatorios = np.random.permutation(matriz.shape[0])  # matriz.shape[0] = cantidad de filas
# devuelve por ejemplo indices_aleatorios = [2 0 1]

# Ordenar las filas de la matriz de forma aleatoria
matriz_ordenada_aleatoriamente = matriz[indices_aleatorios]

print("Matriz original:")
print(matriz)

print("\nMatriz ordenando las filas de forma aleatoria:")
print(matriz_ordenada_aleatoriamente)

print(matriz.shape[0])