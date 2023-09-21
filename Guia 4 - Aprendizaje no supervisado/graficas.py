import matplotlib.pyplot as plt
import numpy as np

#* ----- Graficas SOM -----

def iniciarGraficaSOM(data, neurSom, iris=False):
    fig, ax = plt.subplots(layout='constrained')
    ax.set_title('Grafica inicial')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    if iris:
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(4, 8), ylim=(1, 5))

    if iris:
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(4, 8), ylim=(1, 5))

    ax.scatter(data[:, 0], data[:, 1], linewidths=1)
    
    rectHoriz = []
    rectVert = []
    for i in range(neurSom.shape[0]):
        rectVert.append(ax.plot(neurSom[i, :, 0], neurSom[i, :, 1], 'ko-')[0])
    for i in range(neurSom.shape[1]):
        rectHoriz.append(ax.plot(neurSom[:, i, 0], neurSom[:, i, 1], 'ko-' )[0])

    fig.show()
    return fig, ax, rectHoriz, rectVert

def actualizarGraficaSOM(fig, ax, title, neurSom, rectHoriz, rectVert):
    ax.set_title(title)

    for idx, val in enumerate(rectHoriz):
        val.set_xdata(neurSom[:, idx, 0])
        val.set_ydata(neurSom[:, idx, 1])
    for idx, val in enumerate(rectVert):
        val.set_xdata(neurSom[idx, :, 0])
        val.set_ydata(neurSom[idx, :, 1])

    fig.show()
    plt.pause(0.1)

#* ----- Graficas k-medias -----

def iniciarGraficaKM2D(data, centroide):

    colores = ["#EE0000", "#FF8000", "#FFFF00", "#00C957", 
               "#1C86EE", "#104E8B", "#9400D3", "#8B008B",
               "#333333", "#FF82AB"]

    # Si tengo 3 centroide, tendre solo 3 colores
    # idxColor = np.linspace(0, 10, centroide.shape[0]).astype(int)
    # col = colores[idxColor]
    col = colores[:centroide.shape[0]]

    fig, ax = plt.subplots()
    ax.set_title("Estado inicial")
    ax.set_xlabel("1ra categoria")
    ax.set_ylabel("2da categoria")

    ax.set_aspect('equal', 'box')
    ax.set(xlim=(4, 8), ylim=(1, 5))
    ax.grid(True)

    # Mostrar los datos en gris
    dataPlot = ax.scatter(data[:, 0], data[:, 1], c="#666666", linewidths=0)

    # Mostrar los centroides colorados
    centPlot = ax.scatter(centroide[:, 0], centroide[:, 1], 
                          c=col, marker="s", linewidths=5)

    return fig, ax, dataPlot, centPlot


def actualizarGraficaKM2D(ax, title, dataPlot, centPlot, centroide, clusters):

    colores = ["#EE0000", "#FF8000", "#FFFF00", "#00C957", 
               "#1C86EE", "#104E8B", "#9400D3", "#8B008B",
               "#333333", "#FF82AB"]
    
    ax.set_title(title)

    # Calcular el total de los datos
    totalData = 0
    for cluster in clusters:
        totalData += len(cluster)

    # Inicializar la lista de color
    color = np.full(shape=(totalData), fill_value=colores[0], dtype='U7')

    for i in range(1, len(clusters)):
        color[clusters[i]] = colores[i]

    # Actualizar los colores de los patrones
    dataPlot.set_facecolors(color)

    # Actualizar las posiciones de los centroides
    newCentX = centroide[:, 0]
    newCentY = centroide[:, 1]
    newOffset = np.column_stack((newCentX, newCentY))
    centPlot.set_offsets(newOffset)

    plt.pause(0.2)

def iniciarGraficaKM3D(data, centroide):

    colores = ["#EE0000", "#FF8000", "#FFFF00", "#00C957", 
               "#1C86EE", "#104E8B", "#9400D3", "#8B008B",
               "#333333", "#FF82AB"]

    # Si tengo 3 centroide, tendre solo 3 colores
    # idxColor = np.linspace(0, 10, centroide.shape[0]).astype(int)
    # col = colores[idxColor]
    col = colores[:centroide.shape[0]]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_title("Estado inicial")
    ax.set_xlabel("Largo de sepalo")
    ax.set_ylabel("Ancho de sepalo")
    ax.set_zlabel("Largo de petalo")

    # Mostrar los datos en gris
    dataPlot = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c="#666666", linewidths=0)

    # Mostrar los centroides colorados
    centPlot = ax.scatter(centroide[:, 0], centroide[:, 1], centroide[:, 2], 
                          c=col, marker="s", linewidths=5)

    plt.pause(1)

    return fig, ax, dataPlot, centPlot

def actualizarGraficaKM3D(ax, title, dataPlot, centPlot, centroide, clusters):
    colores = ["#EE0000", "#FF8000", "#FFFF00", "#00C957", 
               "#1C86EE", "#104E8B", "#9400D3", "#8B008B",
               "#333333", "#FF82AB"]
    
    ax.set_title(title)

    # Calcular el total de los datos
    totalData = 0
    for cluster in clusters:
        totalData += len(cluster)

    # Inicializar la lista de color
    color = np.full(shape=(totalData), fill_value=colores[0], dtype='U7')

    for i in range(1, len(clusters)):
        color[clusters[i]] = colores[i]

    # Actualizar los colores de los patrones
    dataPlot.set_facecolor(color)

    # Actualizar las posiciones de los centroides
    newCentX = centroide[:, 0]
    newCentY = centroide[:, 1]
    newCentZ = centroide[:, 2]
    centPlot._offsets3d = (newCentX, newCentY, newCentZ)

    plt.pause(0.2)

#? =======[Test KM]=======
#* Data 2D
# data = np.random.rand(20, 2)
# centroide = np.random.rand(3, 2)
# fig, ax, dataPlot, centPlot = iniciarGraficaKM2D(data, centroide)
# plt.show()

# * Data 3D
# categoria = 5

# data = np.random.rand(100, 3)
# centroide = np.random.rand(categoria, 3)
# fig, ax, dataPlot, centPlot = iniciarGraficaKM3D(data, centroide)


# # determinar clusters
# for epoca in range(20):

#     # Habria que vaciar la lista de cluster
#     clusters = []
#     newClusters = []
#     for i in range(categoria):
#         clusters.append([])
#         newClusters.append([])

#     # Agrupar los patrones en cada cluster
#     for idx, dati in enumerate(data):
#         idxCent = np.argmin(np.linalg.norm(centroide - dati, axis=1))
#         clusters[idxCent].append(idx)

#     # Actualizar los centroides
#     for idx, val in enumerate(centroide):
#         centroide[idx] = np.mean(data[clusters[idx]], axis=0)
#     # print(centroide)
        
#     title = "Epoca " + str(epoca)
#     # ax.set_title(title)
#     actualizarGraficaKM3D(ax, title, dataPlot, centPlot, centroide, clusters)

# plt.show()


# #? =================[Test Ej 1]=================
# ii = np.arange(3)
# jj = np.arange(4)
# j, i = np.array(np.meshgrid(jj, ii))


# cont = np.array((i, j))
# print(cont)

# pp = np.ones(shape=(3, 4, 2))
# pp[:, :, 0] = i
# pp[:, :, 1] = j

# np.random.seed(10000)
# data = np.random.rand(20, 2) * 2 - 1 
# print(data[:, 0], data[:, 1])

# fig, ax, rectHoriz, rectVert = iniciarGrafica(data=data, neurSom=pp)
# plt.pause(1)

# pp[:, :, 0] = j
# pp[:, :, 1] = i

# actualizarGrafica(fig, ax, "hola 1", pp, rectHoriz, rectVert)
# plt.pause(1)

# pp[:, :, 0] = i
# pp[:, :, 1] = j

# actualizarGrafica(fig, ax, "hola 2", pp, rectHoriz, rectVert)
# plt.pause(1)

# plt.show()
