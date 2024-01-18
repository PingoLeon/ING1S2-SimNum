import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def f(t, x, v, frottements, mass, raideur) : # mx'' + ax' + kx = 0 ==> x'' = -ax'/m - kx/m
    return -(frottements * v / mass) - (raideur * x / mass)


def euler_exp_2(f, pos_initiale, vit_initiale, t0, tf, h, a, mass, raideur) :

    pas = int((tf - t0) / h)

    vx = vit_initiale
    x  = pos_initiale
    t  = t0

    liste_vx = [vit_initiale]
    liste_x  = [pos_initiale]
    liste_t  = [t0]

    for i in range(pas):
        x  += h * vx
        vx += h * f(t, x, vx, a, mass, raideur)
        t  += h
        liste_x.append(x)
        liste_t.append(t)
        liste_vx.append(vx)

    return liste_x, liste_vx, liste_t


if __name__ == '__main__':

    plt.style.use('seaborn-v0_8')

    #* -----VARIABLES-----
    pos_initiale_init = 0.05 # m
    vit_initiale_init = 1    # m/s
    frottements_init  = 10   # Kg/s
    simulation_size   = 10
    raideur_init      = 100  #Kg/s
    mass_init         = 1    # kg
    pas_init          = 0.01
    quality_init      = (np.sqrt(raideur_init / mass_init) * mass_init) / frottements_init
    omega_init        = np.sqrt(raideur_init / mass_init - ((frottements_init / (mass_init * 2)) ** 2))
    alpha_init        = frottements_init / (2 * mass_init)
    A_init            = (-vit_initiale_init / np.sqrt(alpha_init ** 2 - (raideur_init / mass_init)) + pos_initiale_init) / 2
    B_init            = vit_initiale_init / np.sqrt(alpha_init ** 2 - (raideur_init / mass_init))

    fig, axe_ = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.4)

    plt.ylim(-0.5, 0.5)

    plt.xlabel('temps (s)')
    plt.ylabel('position (m) / vitesse (m/s)')
    plt.title('Modelisation de la position et de la vitesse en fonction du temps')

    text = plt.text(0.5, 0.5, "", horizontalalignment='center', verticalalignment='center', transform=axe_.transAxes, fontsize=8)
    text.set_text(f'{round(np.mean(quality_init), 2)}')

    x, v, t = euler_exp_2(f, pos_initiale_init, vit_initiale_init, 0, simulation_size, pas_init, frottements_init, mass_init, raideur_init)

    t_values_theorique = np.linspace(0, simulation_size, int(simulation_size / pas_init))

    #* -------EQUATIONS-------
    if quality_init > 1/2 :
        text.set_text('Pseudo périodique')
        position_theorique = [np.exp(-alpha_init * time) * (pos_initiale_init * np.cos(omega_init * time) + ((vit_initiale_init + alpha_init * pos_initiale_init) / omega_init) * np.sin(omega_init * time)) for time in t_values_theorique]
    elif quality_init == 1/2 :
        text.set_text('Critique')
        position_theorique = [(pos_initiale_init + (vit_initiale_init + np.sqrt(raideur_init / mass_init) * pos_initiale_init) * time) * np.exp(-np.sqrt(raideur_init / mass_init) * time) for time in t_values_theorique]
    else :
        text.set_text('Apériodique')
        position_theorique = [np.exp(-alpha_init * time) * ((A_init * np.exp(-np.sqrt(alpha_init ** 2 - (raideur_init / mass_init)) * time)) + (B_init * np.exp(np.sqrt(alpha_init ** 2 - (raideur_init / mass_init)) * time))) for time in t_values_theorique]

    #* -------PLOT-------
    line1, = plt.plot(t, x, label='position numerique', c='red')
    line2, = plt.plot(t, v, label='vitesse numerique',  c='blue')
    line3, = plt.plot(t_values_theorique, position_theorique, label='position theorique', c='green')

    plt.legend(loc="upper left", fontsize=8)

    #* -------SLIDERS-------
    ax_color = 'lightgoldenrodyellow'

    ax_pos_initiale = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=ax_color)
    ax_vit_initiale = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=ax_color)
    ax_frottements  = plt.axes([0.15, 0.2,  0.65, 0.03], facecolor=ax_color)
    ax_raideur      = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=ax_color)
    ax_mass         = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=ax_color)

    s_pos_initiale = Slider(ax_pos_initiale, 'Pos initiale', 0.01, 10.0,  valinit=pos_initiale_init)
    s_vit_initiale = Slider(ax_vit_initiale, 'Vit initiale', 0.1,  10.0,  valinit=vit_initiale_init)
    s_frottements  = Slider(ax_frottements,  'Frottements',  0.1,  10.0,  valinit=frottements_init)
    s_raideur      = Slider(ax_raideur,      'Raideur',      0.1,  100.0, valinit=raideur_init)
    s_mass         = Slider(ax_mass,         'Masse',        0.1,  10.0,  valinit=mass_init)

    def update(val) :
        pos_initiale = s_pos_initiale.val
        vit_initiale = s_vit_initiale.val
        frottements  = s_frottements.val
        raideur      = s_raideur.val
        mass         = s_mass.val
        quality      = (np.sqrt(raideur / mass) * mass) / frottements
        omega        = np.sqrt(raideur / mass - ((frottements / (mass * 2)) ** 2))
        alpha        = frottements / (2 * mass)
        A            = (-vit_initiale / np.sqrt(alpha ** 2 - (raideur / mass)) + pos_initiale_init) / 2
        B            = vit_initiale / np.sqrt(alpha ** 2 - (raideur / mass))

        x, v, t = euler_exp_2(f, pos_initiale, vit_initiale, 0, simulation_size, pas_init, frottements, mass, raideur)
        line1.set_ydata(x)
        line2.set_ydata(v)

        t_values_theorique = np.linspace(0, simulation_size, int(simulation_size / pas_init))

        if quality > 1/2 :
            text.set_text('Pseudo périodique')
            position_theorique = [np.exp((-alpha) * time) * (pos_initiale * np.cos(omega * time) + ((vit_initiale + alpha * pos_initiale) / omega) * np.sin(omega * time)) for time in t_values_theorique]
        elif quality == 1/2 :
            text.set_text('Critique')
            position_theorique = [(pos_initiale + (vit_initiale + np.sqrt(raideur / mass) * pos_initiale) * time) * np.exp(-np.sqrt(raideur / mass) * time) for time in t_values_theorique]
        else :
            text.set_text('Apériodique')
            position_theorique = [np.exp(-alpha * time) * ((A * np.exp(-np.sqrt(alpha ** 2 - (raideur / mass)) * time)) + (B * np.exp(np.sqrt(alpha ** 2 - (raideur / mass)) * time))) for time in t_values_theorique]


        line3.set_ydata(position_theorique)


        fig.canvas.draw_idle()

    s_pos_initiale.on_changed(update)
    s_vit_initiale.on_changed(update)
    s_frottements.on_changed(update)
    s_raideur.on_changed(update)
    s_mass.on_changed(update)

    #* creation du bouton reset
    axe_reset    = plt.axes([0.87, 0.12, 0.1, 0.1])
    button_reset = plt.Button(axe_reset, 'Reset', color='#4c72b0', hovercolor='#6498ed')

    #* fonction de reset
    def reset(event):
        s_pos_initiale.reset()
        s_vit_initiale.reset()
        s_frottements.reset()
        s_raideur.reset()
        s_mass.reset()

    #* mise a jour du bouton reset
    button_reset.on_clicked(reset)

    plt.show()