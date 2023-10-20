from entrenamiento_MLP import *
from prueba_MLP import *
import matplotlib.pyplot as plt

# --- Resolucion ejercicio 3 guia 2: Iris ---

# La arquitectura mas basica con una capa podria ser [3], que seria como tener tres perceptrones
# simples, pero habria que entrenar cada neurona para una de las clases.

# Con dos capas, la arquitecutra minima es [1, 3]. Ese “1” proyecta las cuatro dimensiones 
# (los cuatro valores de ancho y largo que tenemos) en una dimension.
# arquitectura = [1, 3]
# gamma = 0.1    

# arquitectura = [4, 3]
# gamma = 0.1    # 0.1 o 0.01

arquitectura = [3, 3]
gamma = 0.01     # con 0.1 muy bien, con 0.01 demora un poquito mas pero los errores bajan mucho 
                # mas suaves y con 0.001 se estanca varias veces (el error de clasificacion) y demora mas 

tasaErrorAceptable = 0.05  
numMaxEpocas = 500 
bSigmoidea = 1      
grafError = True

W_mat = entrenamiento_MLP("datos/irisbin_trn.csv", arquitectura, tasaErrorAceptable, numMaxEpocas, gamma, bSigmoidea, grafError)
prueba_MLP("datos/irisbin_tst.csv", W_mat, arquitectura, bSigmoidea)

plt.show()  # muestro la grafica aca para que no se cierre al terminar el entrenamiento