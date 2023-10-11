import numpy as np 
from gradienteDescendente import *
from graficas import *
from funciones_ej1 import *
from algGenet import *


#* ---------- Funcion inciso i) ----------
#* i) f(x) = -x*sin(sqrt(abs(x))) 
#  con x = [-512, 512], necesito 10 bits para representar el rango 
#  de x, con esto alcanza para representar un individuo

cantIndividuos = 100
cantPadres = 20
codCrom = [10]    # es como se codifica el cromosoma
cantMaxGeneracion = 1000      # recordar poner un valor alto porque no sabemos cuando va a converger

#! Ver como mejorar la funcion fitness!
fitnessBuscado = 355     #! VER QUE VALOR PODEMOS USAR
probMutacion = 0.1
probCruza = 0.8
xmin = [-512]
xmax = [512]
graf = 1

poblacion = algGenetico(f1, xmin, xmax, cantIndividuos, cantPadres,
                        codCrom, fitnessBuscado, cantMaxGeneracion, 
                        probMutacion, probCruza, graf, df1)

#* ---------- Funcion inciso ii) ----------

cantIndividuos = 100
cantPadres = 0.20

codCrom = [8, 8]    # es como se codifica el cromosoma
cantMaxGeneracion = 1000      # recordar poner un valor alto porque no sabemos cuando va a converger

fitnessBuscado = -4     #! VER QUE VALOR PODEMOS USAR
probMutacion = 0.05
probCruza = 0.6
xmin = [-100, -100]
xmax = [100, 100]
graf = 2

poblacion = algGenetico(f2, xmin, xmax, cantIndividuos, cantPadres,
                        codCrom, fitnessBuscado, cantMaxGeneracion, 
                        probMutacion, probCruza, graf, df2)

plt.show()
