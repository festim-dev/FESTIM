import matplotlib.pyplot as plt

import csv


def read(filename, label):
    arc_length = []
    y = []
    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        for row in plots:
            if 'arc_length' in row:
                index_arc_length = row.index('arc_length')
                index_y = row.index(label)
            else:
                arc_length.append(float(row[index_arc_length]))
                y.append(float(row[index_y]))
    return arc_length, y


def plot_concentration(period, plot_solute=True):
    """
    plot the concentrations for one period
    """
    for trap in [1, 2, 3, 4]:
        arc_length, y = read(
            folder + "profile_trap" + str(trap) + '_' + period + '.csv',
            str(trap))
        plt.plot(arc_length, y, label="Trap " + str(trap))

    if plot_solute:
        arc_length, y = read(
            folder + "profile_solute_" + period + '.csv', "solute_m3")
        plt.plot(arc_length, y, label="Solute")

    plt.ylim(bottom=1e17)
    plt.xlabel('arc length (m)')
    plt.ylabel(r'Concentration (H m$^{-3}$)')
    plt.yscale("log")
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.3)
    plt.grid(which='major', alpha=0.7)
    plt.show()
    return


def plot_T(periods):
    for period in periods:
        arc_length, y = read(
            folder + "profile_T_" + period + '.csv', 'T')
        plt.plot(arc_length, y, label=period)
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.3)
    plt.grid(which='major', alpha=0.7)
    plt.show()


folder = 'results/05_ITER_case_theta_sol2_99950/'

plot_concentration("implantation")
plot_concentration("rest")
plot_concentration("baking")
plot_T(["implantation", "rest", "baking"])
