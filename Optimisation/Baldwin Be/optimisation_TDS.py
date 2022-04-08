import csv
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import numpy as np

from sim_desorption import simu_recomb, simu_sievert, resting_time, tds_time

j = 0


def read_ref(filename):
    '''
    Reads the data in filename
    '''
    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=';')
        res = []
        for row in plots:
            if 'd' not in row and 'T' not in row and 't (s)' not in row:
                res.append([float(row[i]) for i in [0, 1]])

    return res


def mean_absolute_error(a, b, bounds=[], p=1):
    val = 0
    count = 0
    coeff = 1
    for e in b:
        for b in bounds:
            if e[0] > b[0] and e[0] < b[1]:
                coeff = p
            else:
                coeff = 1
        val += coeff*abs(e[1] - a(e[0]))
        count += coeff
    val *= 1/count
    return val


def RMSD(a, b):
    val = 0
    for e in b:
        val += (e[1] - a(e[0]))**2
    val /= len(b)
    val = val**0.5
    return val


def error(p):
    '''
    Compute average absolute error between simulation and reference
    '''
    print('-' * 40)
    global j
    j += 1
    print('i = ' + str(j))
    print('New simulation.')
    print('Point is:')
    print(p)
    for e in p:
        if e < 0 or e>2:
            return 1e30
    res = simu_recomb(p)
    res.pop(0)  # remove header
    res = np.array(res)
    # create d(ret)/dt
    t = []
    flux = []
    for i in range(0, len(res) - 1):
        if res[i][0] >= resting_time:
            t.append(res[i][0])
            flux.append(-(res[i+1][4] - res[i][4])/(res[i+1][0] - res[i][0]))
    t = np.array(t)
    flux = np.array(flux)
    interp_tds = interp1d(t, flux, fill_value='extrapolate')
    err = mean_absolute_error(interp_tds, ref)
    # err = RMSD(interp_tds, ref)
    err /= 1

    print('Average absolute error is :' + str(err) + ' ' + str(fatol) + ' ' + str(xatol))
    # with open(folder + '/simulations_results.csv', 'a') as f:
    #     writer = csv.writer(f, lineterminator='\n', delimiter=',')
    #     writer.writerow([*p, err])
    return err


folder = 'Results'
ref = read_ref('ref.csv')

if __name__ == "__main__":
    j = 0

    # x0 = np.array([0.101, 0.805, 0.7, 0.032, 1.07, 0.3])
    # result is (with sievert)
    # trap 1 :
    # - n1 = 0.13118969 at.fr
    # - E1 = 0.82560736 eV
    # - f1 = 0.57341477
    # trap 2 :
    # - n2 = 0.0360109 at.fr
    # - E2 = 1.11370884 eV
    # - f2 = 0.38074201
    fatol = 1e15
    xatol = 1e-3

    x0 = np.array([0.098, 0.743, 0.7, 0.032, 0.940, 0.3])
    # result is (with recomb)
    # trap 1 :
    # - n1 = 0.10919773 at.fr
    # - E1 = 0.74808897 eV
    # - f1 = 0.73197268
    # trap 2 :
    # - n2 = 0.03383912 at.fr
    # - E2 = 0.92968304 eV
    # - f2 = 0.27848899
    def minimise_with_neldermead(ftol, xtol, x):
        global fatol
        global xatol
        fatol = ftol
        xatol = xtol
        res = minimize(error, x, method='Nelder-Mead',
                       options={'disp': True, 'fatol': ftol, 'xatol': xtol})
        print('Solution is: ' + str(res.x))
        goon = True
        while goon:
            a = input('Do you wish to restart ?')
            if a == 'no' or a == 'No':
                goon = False
            elif a == 'Yes' or a == 'yes':
                new_fatol = fatol
                new_xatol = xatol
                b = input('Choose fatol :')
                if b != '':
                    new_fatol = float(b)
                c = input('Choose xatol :')
                if c != '':
                    new_xatol = float(c)
                minimise_with_neldermead(new_fatol, new_xatol, np.array([res.x[0], res.x[1]]))
    minimise_with_neldermead(fatol, xatol, x0)
