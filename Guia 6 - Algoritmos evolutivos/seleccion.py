import numpy as np

# Para implementar operadores de seleccion que utilicemos, ya sea ruleta, ventana, competencia.

def selectRuleta(fitness, cantPadres):
    # Se debe normalizar los fitness para la probabilidad sumen 1
    sum = np.sum(fitness)
    normFit = fitness/sum
    
    idxFit = np.arange(fitness.shape[0])
    # p es el parametro de probabilidad, debe ser un vector 1D que sus componentes sumen 1
    idxPadres = np.random.choice(idxFit, size=cantPadres, p=normFit)
    return idxPadres

def selectVentana(fitness, cantPadres):
    # Supongo que la funcion admite repeticion porque 
    # siempre trata de dejar los mejores
    
    ordenFit = np.argsort(-fitness)
    idxPadres = np.empty(shape=(cantPadres), dtype=int)

    cantInd = ordenFit.shape[0]
    paso =  cantInd // cantPadres
    for i in range(cantPadres):
        idxPadres[i] = np.random.choice(ordenFit[:cantInd - (paso*i)])
        # print(ordenFit[:cantInd - (paso*i)])

    return idxPadres

# Se eligen “n” individuos al azar y luego de esos “n” nos quedamos con el de mejor fitness.
# Mientras mas grande sea ese parametro, mas chances tendremos de elegir los individuos mas altos. 
def selectCompetencia(fitness, cantPadres):
    idxFit = np.arange(fitness.shape[0])
    idxBool = np.full(shape=(fitness.shape[0]), fill_value=True)

    cantCompetencia = fitness.shape[0] // cantPadres

    idxPadres = []

    for i in range(cantPadres):
        #? Asumo que no admiete repeticion
        idxSelec = np.random.choice(idxFit[idxBool], size=cantCompetencia, replace=False)
        idxBool[idxSelec] = False
        
        idxMaxFit = np.argmax(fitness[idxSelec])
        idxPadres.append(idxSelec[idxMaxFit])

    return np.array(idxPadres)

# -------------------------------------------------------

#? test seleccion
# fitness = np.random.randint(0, 7, size=(25))
# cantPadres = 8
# print(fitness)

# idxPadres = selectRuleta(fitness, cantPadres)
# print("\nRuleta: ")
# print(f"Los indices seleccionados son: {idxPadres}")
# print(f"Los valores de fitness son: {fitness[idxPadres]}")

# idxPadres = selectVentana(fitness, cantPadres)
# print("\nVentana: ")
# print(f"Los indices seleccionados son: {idxPadres}")
# print(f"Los valores de fitness son: {fitness[idxPadres]}")

# idxPadres = selectCompetencia(fitness, cantPadres)
# print("\nCompentencia: ")
# print(f"Los indices seleccionados son: {idxPadres}")
# print(f"Los valores de fitness son: {fitness[idxPadres]}")
