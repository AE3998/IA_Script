import numpy as np 

#* Convertir binario (en formato array) a int
res = 0
cromosoma = np.array([0,0,1,0])

for i in range(len(cromosoma)):
    if(cromosoma[i] == 1):
        exp = len(cromosoma)-1-i 
        res += np.sum(2**exp) 
print(res)

# -----------------------------------------------


