from entrenamiento_MLP import *
from prueba_MLP import *

# --- Resolucion ejercicio 2 guia 2: Concent ---

# arquitectura = [3, 1]
# gamma = 0.01    # con 0.01 la grafica de error cuadratico es mucho mas suave

arquitectura = [4, 1]
gamma = 0.05    

# arquitectura = [5, 1]
# gamma = 0.01 

tasaErrorAceptable = 0.03    # hasta 0.03 lo resuelve en menos de 100 epocas      
numMaxEpocas = 500 
bSigmoidea = 5      #* la derivada que usamos es con b=1, si se cambia el valor de "b" deberia cambiar la derivada.
                    # lo unico que en este ejercicio con b=1 no converge, queda en 30% de error aproximadamente (hasta 2000 epocas probe), 
                    # se ve que la red aprende pero muy despacito, en cambio si uso b=5 funciona perfecto.
grafError = False
grafCategorias = True

W_mat = entrenamiento_MLP("datos/concent_trn.csv", arquitectura, tasaErrorAceptable, numMaxEpocas, gamma, bSigmoidea, grafError)
prueba_MLP("datos/concent_tst.csv", W_mat, arquitectura, bSigmoidea, grafCategorias)

plt.show()  # muestro la grafica aca para que no se cierre al terminar el entrenamiento