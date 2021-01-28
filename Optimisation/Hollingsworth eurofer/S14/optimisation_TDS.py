import csv
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import numpy as np

from sim import simu, implantation_time, resting_time, tds_time

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
        for bound in bounds:
            if e[0] > bound[0] and e[0] < bound[1]:
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
        if e < 0:
            return 1e30
    res = simu(p)
    res.pop(0)  # remove header
    res = np.array(res)
    # create d(ret)/dt
    T = []
    flux = []
    for i in range(0, len(res) - 1):
        if res[i][0] >= implantation_time + resting_time:
            T.append(res[i][1])
            flux.append(-(res[i+1][2] - res[i][2])/(res[i+1][0] - res[i][0]))
    T = np.array(T)
    flux = np.array(flux)
    interp_tds = interp1d(T, flux, fill_value='extrapolate')
    # err = mean_absolute_error(interp_tds, ref, bounds=[(445, 492)], p=10)
    err = mean_absolute_error(interp_tds, ref)
    # err = RMSD(interp_tds, ref)
    err /= 1

    print('Average absolute error is :' + str(err) + ' ' + str(fatol) + ' ' + str(xatol))
    # with open(folder + '/simulations_results.csv', 'a') as f:
    #     writer = csv.writer(f, lineterminator='\n', delimiter=',')
    #     writer.writerow([*p, err])
    return err


folder = 'Results'
ref = read_ref('S14.csv')

if __name__ == "__main__":
    j = 0

    x0 = np.array([1.1, 2.6])
    # result is : E = 1.067 eV ; n =  2.82e-2 at. fr.
    fatol = 1e15
    xatol = 1e-3

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
