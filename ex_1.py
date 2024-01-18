import matplotlib.pyplot as plt
import numpy as np
import math as m
from matplotlib.widgets import Slider

# -----ACTIVITE 1-----
# chute libre

# Equation differentielle:
def function(vz, g, frottements, mass):
    return g - ((frottements / mass) * vz)

# Avec la vitesse
def euler_exp(f, t0, tf, v0, nb_iterations, g, frottements, mass):
    pas = (tf - t0) / nb_iterations
    vz = v0
    t = t0
    liste_t = [t0]
    liste_vz = [v0]

    for i in range(nb_iterations):
        vz += pas * f(vz, g, frottements, mass)
        t += pas
        liste_vz.append(vz)
        liste_t.append(t)
    
    return liste_t, liste_vz

# Avec la position
def euler_exp2(f, t0, tf, z0, nb_iterations, g, frottements, mass, v0):
    t_values, v_values = euler_exp(f, t0, tf, 0, nb_iterations, g, frottements, mass)
    pas = (tf - t0) / nb_iterations
    z = np.zeros_like(t_values)
    z[0] = z0
    v = v0

    for i in range(1, nb_iterations):
        v += pas * f(v_values[i], -g, frottements, mass)
        z[i] = z[i-1] +  pas * v

    #replace the last value of z 
    z[-1] = z[-2]

    #replace all the values of z that are negative by 0
    for i in range(len(z)):
        if z[i] < 0:
            z[i] = 0

    return t_values, z



if __name__ == '__main__':
    plt.style.use('seaborn-v0_8')

    # Definition des variables
    mass_init = 1  # kg
    g_init = 9.81  # m/s^2
    v0_init = 0  # m
    z0_init = 250  # m
    frottements_init = 5  # Kg/s
    pas_init = 0.05
    simulation_size_init = 10

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(left=0.1, bottom=0.4)

    # Vitesse numerique
    t_values, z_values = euler_exp(function, 0, simulation_size_init, v0_init, int(simulation_size_init / pas_init),g_init, frottements_init, mass_init)
    # Vitesse theorique
    t_values_theorique = np.linspace(0, simulation_size_init, int(simulation_size_init / pas_init))

    # Solution de l'equation differentielle : mg/a + (-mg/a + v0) * exp(-a/m * t)
    z_values_theorique = [(((mass_init * g_init) / frottements_init) + (((-mass_init * g_init) / frottements_init) + v0_init) * np.exp((-frottements_init / mass_init) * t)) for t in t_values_theorique]

    z_values_copy = z_values.copy()
    z_values_theorique_copy = z_values_theorique.copy()

    erreur = []

    # Calcul de l'erreur
    for i in range(len(z_values_theorique_copy)):
        erreur.append(z_values_copy[i] - z_values_theorique_copy[i])

    # Affichage de l'erreur et l'effacer a chaque fois
    text = plt.text(0.860, 0.085, "", horizontalalignment='center', verticalalignment='center',transform=ax1.transAxes, fontsize=8)
    text.set_text(f'{round(np.mean(erreur) * 100, 2)}%')

    line1, = ax1.plot(t_values, z_values, label="Vitesse numerique", c="red")
    line2, = ax1.plot(t_values_theorique, z_values_theorique, label="Vitesse theorique", c="blue")
    line3, = ax1.plot(t_values_theorique, erreur, label="Erreur", c="green", ls="--")

    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlabel("t")
    ax1.set_ylabel("vz")
    ax1.set_title("Modelisation numerique et theorique\n de la vitesse en fonction du temps")

    # Position numerique
    t_values, z_values_pos = euler_exp2(function, 0, simulation_size_init, z0_init, int(simulation_size_init / pas_init),g_init, frottements_init, mass_init, v0_init)

    line4, = ax2.plot(t_values, z_values_pos, label="Position numerique", c="red")

    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlabel("t")
    ax2.set_ylabel("z")
    ax2.set_title("Modelisation numerique et theorique\n de la position en fonction du temps")

    # Creation des sliders
    axe_color = 'lightgoldenrodyellow'
    axe_mass = plt.axes([0.2, 0.25, 0.65, 0.03], facecolor=axe_color)
    axe_g = plt.axes([0.2, 0.20, 0.65, 0.03], facecolor=axe_color)
    axe_v0 = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axe_color)
    axe_frottements = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor=axe_color)
    axe_z0 = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axe_color)

    slider_mass = Slider(axe_mass, 'Masse', 0.25, 10.0, valinit=mass_init)
    slider_g = Slider(axe_g, 'Gravitation', 1.0, 20.0, valinit=g_init)
    slider_v0 = Slider(axe_v0, 'Vitesse initiale', -10.0, 10.0, valinit=v0_init)
    slider_frottements = Slider(axe_frottements, 'Frottements', 0.1, 10.0, valinit=frottements_init)
    slider_z0 = Slider(axe_z0, 'Position initiale', 0.0, 500, valinit=z0_init)

    # Fonction de mise a jour des sliders
    def update(val):
        mass = slider_mass.val
        g = slider_g.val
        v0 = slider_v0.val
        frottements = slider_frottements.val
        z0 = slider_z0.val

        # Velocity update
        t_values, z_values = euler_exp(function, 0, simulation_size_init, v0, int(simulation_size_init / pas_init),g, frottements, mass)
        t_values_theorique = np.linspace(0, simulation_size_init, int(simulation_size_init / pas_init))
        z_values_theorique = [(((mass * g) / frottements) + (((-mass * g) / frottements) + v0) *np.exp((-frottements / mass) * t)) for t in t_values_theorique]

        line1.set_data(t_values, z_values)
        line2.set_data(t_values_theorique, z_values_theorique)

        # Error update
        erreur = []
        z_values_copy = z_values.copy()
        z_values_theorique_copy = z_values_theorique.copy()
        for i in range(len(z_values_theorique_copy)):
            erreur.append(abs(z_values_copy[i] - z_values_theorique_copy[i]))
        text.set_text(f'{round(np.mean(erreur) * 100, 2)}%')
        line3.set_data(t_values_theorique, erreur)

        # Position update
        t_values, z_values_pos = euler_exp2(function, 0, simulation_size_init, z0, int(simulation_size_init / pas_init), g, frottements, mass, v0)

        line4.set_data(t_values, z_values_pos)

        fig.canvas.draw_idle()

    slider_mass.on_changed(update)
    slider_g.on_changed(update)
    slider_v0.on_changed(update)
    slider_frottements.on_changed(update)
    slider_z0.on_changed(update)
    plt.show()