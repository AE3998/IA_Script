import matplotlib as plt

def asignarColor(yd):
    color = []
    for i in yd:
        color.append("#00EEEE" if i < 0 else "#EE4000")
    return color

def crearLegend(ax):
    # fake_blue, fake_red, blue, red
    colores = ["#00EEEE", "#EE4000", "#0000FF", "#FF0000"]
    legend = ["F-neg", "F-pos", "V-neg", "V-pos"]
    n = len(colores)
    handle = [(plt.Line2D([], [],
                          color = colores[i], label=legend[i],
                          marker="o", linewidth=0)) for i in range(n)]
    # Agregar los legend en axes
    legend1 = ax.legend(handles = handle[:n//2], loc = "upper left")
    ax.add_artist(legend1)
    ax.legend(handles = handle[n//2:], loc = "upper right")
