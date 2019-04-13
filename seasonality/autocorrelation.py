import numpy as np
#datos random
a = np.asarray((list(range(20))+list(reversed(range(20))))*20)
a = a + 20 *(np.random.rand(a.shape[0]) * 0.2 - 0.1)
a[300] = np.nan
import matplotlib.pyplot as plt
plt.plot(a)
plt.show()

#Genero listas de obsevaciones con corrimientos hasta en 70 unidades de tiempo
b = [a[i:-(71-i)] for i in range(70)]
#Calculo la autocorrelación y me quedo con la autocorrelación de mis datos originales contra todos los corrimientos
c = np.corrcoef(b)[0, :]
#"nuetralizo" la autocorrelación consigo mismo ya que es 1 siempre
c[0] = 0

#Veo el maximo, supongo que me indica seasonality
print(np.argmax(c))
print(c[np.argmax(c)])

print(np.argmin(c))
print(c[np.argmin(c)])