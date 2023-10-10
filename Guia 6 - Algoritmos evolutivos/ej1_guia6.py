import numpy as np 
from gradienteDescendente import *
from graficas import *
from funciones_ej1 import *

#* ---------- Funcion inciso i) ----------
#* i) f(x) = -x*sin(sqrt(abs(x))) 
#  con x = [-512, 512], necesito 10 bits para representar el rango 
#  de x, con esto alcanza para representar un individuo

cantIndividuo = 100
cromosoma = [10]    # es como se codifica el cromosoma
cantMaxGeneracion = 1000      # recordar poner un valor alto porque no sabemos cuando va a converger

#? Ya que el fitness es entre 0 (minimo) y 1 (maximo), propongo un 0.95
fitnessBuscado = 0.9     #! VER QUE VALOR PODEMOS USAR
probMutacion = 0.1
probCruza = 0.8



#* ---------- Funcion inciso ii) ----------
