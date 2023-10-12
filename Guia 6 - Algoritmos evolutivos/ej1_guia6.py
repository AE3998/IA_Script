import numpy as np 
from gradienteDescendente import *
from graficas import *
from funciones_ej1 import *
from algGenet import *

# Si uso por ejemplo cantIndividuos = 100, en la grafica del algoritmo genetico (punto azules) voy
# a ver que al comienzo se ubican los 100 individuos (soluciones, que serian puntos), y al ir 
# calculando la cruza, mutacion, evaluar, etc. se van ubicando en los minimos a los que llegan.
# Mientras que en la grafica del gradiente descendente aparecen los 100 individuos (puntos) y 
# luego se actualiza quedan los puntos en los minimos que alcanzaron cada uno.

#* ---------- Funcion inciso i) ----------
#* i) f(x) = -x*sin(sqrt(abs(x))) 
#  con x = [-512, 512], necesito 10 bits para representar el rango 
#  de x, con esto alcanza para representar un individuo

cantIndividuos = 100
cantPadres = 0.4
codCrom = [10]    # es como se codifica el cromosoma
cantMaxGeneracion = 1000      # recordar poner un valor alto porque no sabemos cuando va a converger

fitnessBuscado = 390
probMutacion = 0.1
probCruza = 0.8
xmin = [-512]
xmax = [512]
graf = 1

print(f"Funcion 1:")
mejorIndv, xInit = algGenetico(f1, xmin, xmax, cantIndividuos, cantPadres,
                        codCrom, fitnessBuscado, cantMaxGeneracion, 
                        probMutacion, probCruza, graf, df1)

print(f"Mejor individuo por algoritmo genetico es: {mejorIndv}")
print(f"Valor en ese punto es: {f1(mejorIndv)}")
print("\n")

mejorIndv = algGradient(xInit, df1, np.array(xmin), np.array(xmax), graf=1)
print(f"Mejor individuo por gradiente descendiente es: {mejorIndv}")
print(f"Valor en ese punto es: {f1(mejorIndv)}")
print("\n"*2)

#* ---------- Funcion inciso ii) ----------

cantIndividuos = 100
cantPadres = 0.50

codCrom = [8, 8]    # es como se codifica el cromosoma
cantMaxGeneracion = 1000      # recordar poner un valor alto porque no sabemos cuando va a converger

fitnessBuscado = 4
probMutacion = 0.05
probCruza = 0.6
xmin = [-100, -100]
xmax = [100, 100]
graf = 2

print(f"Funcion 2:")
mejorIndv, xInit = algGenetico(f2, xmin, xmax, cantIndividuos, cantPadres,
                        codCrom, fitnessBuscado, cantMaxGeneracion, 
                        probMutacion, probCruza, graf, df2)

print(f"Mejor individuo es: {mejorIndv}")
print(f"Valor en ese punto es: {f2(mejorIndv)}")
print("\n")

mejorIndv = algGradient(xInit, df2, np.array(xmin), np.array(xmax), graf=2)
print(f"Mejor individuo por gradiente descendiente es: {mejorIndv}")
print(f"Valor en ese punto es: {f2(mejorIndv)}")
plt.show()
