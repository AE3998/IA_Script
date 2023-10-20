from entrenamiento_MLP import *
from prueba_MLP import *

# --- Resolucion XOR con perceptron multicapa ---

arquitectura = [2, 1]
tasaErrorAceptable = 0   # debe tener 100% de acierto porque el XOR es un problema facil de resolver
numMaxEpocas = 100
gamma = 0.1    # desde 1 hasta 0.1 muy rapido y en una epoca. Con 0.01 demora un poco mas pero sirve y valores mas chicos ya funciona mal
bSigmoidea = 1      #* la derivada que usamos es con b=1, recordar cambiarla si se prueban otros valores

grafError = True
grafCategorias = True

W_mat = entrenamiento_MLP("datos/XOR_trn.csv", arquitectura, tasaErrorAceptable, numMaxEpocas, gamma, bSigmoidea, grafError)
prueba_MLP("datos/XOR_tst.csv", W_mat, arquitectura, bSigmoidea, grafCategorias)

plt.show()  # muestro la grafica aca para que no se cierre al terminar el entrenamiento