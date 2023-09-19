import matplotlib.pyplot as plt
import numpy as np


def iniciarGrafica(data, neurSom):
    fig, ax = plt.subplots(layout='constrained')
    ax.set_title('Grafica inicial')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set(xlim=(-1, 1), ylim=(-1, 1))

    # fake_blue, fake_red, blue, red
    # colores = ["#00EEEE", "#EE4000", "#0000FF", "#FF0000"]
    ax.scatter(data[:, 0], data[:, 1], linewidths=1, c="#00EEEE")
    
    rectHoriz = []
    rectVert = []
    for i in range(neurSom.shape[0]):
        rectVert.append(ax.plot(neurSom[i, :, 0], neurSom[i, :, 1], 'ko-')[0])
    for i in range(neurSom.shape[1]):
        rectHoriz.append(ax.plot(neurSom[:, i, 0], neurSom[:, i, 1], 'ko-' )[0])

    fig.show()
    return fig, ax, rectHoriz, rectVert

def actualizarGrafica(fig, ax, title, neurSom, rectHoriz, rectVert):
    ax.set_title(title)

    for idx, val in enumerate(rectHoriz):
        val.set_xdata(neurSom[:, idx, 0])
        val.set_ydata(neurSom[:, idx, 1])
    for idx, val in enumerate(rectVert):
        val.set_xdata(neurSom[idx, :, 0])
        val.set_ydata(neurSom[idx, :, 1])

    fig.show()
    plt.pause(0.1)


# #? =================[Test]=================
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
