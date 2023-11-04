import numpy as np

yy = [2, 3, 1]
Yd = [1]
print(yy[-1])
# print(yy[-1][0] - Yd[0][0])
# -- Reservar espacio para las salidas de todas las capas --
capas = np.array([3, 2, 1])
arquitectura = [2, 1, 3]
# print(capas)
# print(arquitectura)
print("Cant. capas: ", len(arquitectura))
print("Cant. capas: ", np.size(arquitectura))
# print("Cant. capas: ", np.size(capas))

for i in arquitectura[:-1]: 
    print(i)

Y_vec = []
for i in capas[:-1]:
    # Y_vec.append(np.concatenate((np.array([-1]), np.zeros(i))))
    Y_vec.append(np.zeros(i+1)) 
Y_vec.append(np.zeros(capas[-1]))
print("Y_vec:", Y_vec)   #[array([-1.,  0.,  0.]), array([0.])]
Y_vec[0][0] = -1
for j in range(1, np.size(capas)-1):
    Y_vec[j][0] = -1    # agrego -1 en la primera columna para el sesgo
print("Y_vec2:", Y_vec) 

# --- INICIALIZAR MATRICES DE PESOS PARA CADA CAPA --
X = np.array([[0.8], [0.2]])
## Para la primera capa usamos las entradas X:
W_mat = [np.random.rand(capas[0], np.size(X, 1))-0.5]
## Para las capas ocultas y la final (capas desde 1 hasta el final), tomamos el tamano de los 
## vectores de las salidas de las capas anteriores: y^I, y^II, etc. que estan en "Y_vec"
## W_mat[1] usa el tamano de Y_vec[0] que seria y^I, W_mat[2] el de Y_vec[1] que seria y^II y asi sucesivamente.
for i in range(1, np.size(capas)):
     W_mat.append(np.random.rand(capas[i], np.size(Y_vec[i-1]))-0.5)
print(W_mat)

# -- INICIALIZAR DELTAS PARA CADA CAPA --
deltas = []
for i in capas:
    deltas.append(np.zeros(i))
print("Deltas:")
print(deltas)   # [array([0., 0.]), array([0.])] 
# como es capa = [2 1] nos da dos deltas para la primera capa y uno para la capa final

#* --- PARTE DE PROPAGACION HACIA ATRAS ---

W = np.array([[1,2,3],
              [4,5,6]])
deltas = np.array([7,8])

# np.transpose(W[:, 1:]) -> [[2 5]
#                             3 6]

aux = np.multiply(deltas, np.transpose(W[:, 1:]))

# aux -> [[14 40]
#         [21 48]]

sumatoria = np.sum(aux, axis=1)
# sumatoria = [54 69]

#* ------------