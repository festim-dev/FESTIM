from scipy.interpolate import interp2d, griddata
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import time
from cycler import cycler
E = []
n = []
error = []

data = np.genfromtxt('cost_function_data_very_refined.csv', delimiter=',')
N = 2000
E = data[:N, 0]
n = data[:N, 1]*1e3
error = data[:N, 2]
error /= max(error)

zi = interp2d(E, n, error, kind="cubic", fill_value=1e32)
i = 0


def fun(x):
    global i
    i += 1
    return zi(x[0], x[1])


N = 100
x0_E = np.random.uniform(0.6, 1.1, size=N)
x0_n = np.random.uniform(0.1, 4.5, size=N)
norm_to_optimal = []
for i in range(0, N):
    norm_to_optimal.append(((x0_E[i] - 0.858)**2 + (x0_n[i] - 1.222)**2)**0.5)

x0_E_sorted = [x for _,x in sorted(zip(norm_to_optimal, x0_E))]
x0_n_sorted = [x for _,x in sorted(zip(norm_to_optimal, x0_n))]


results = []
for i in range(0, N):
    x0 = [x0_E_sorted[i], x0_n_sorted[i]]
    results.append([])

    res = minimize(fun, x0, method="Nelder-Mead", options={"disp": False})
    if res.x[0] < 0.87 and res.x[0] > 0.85 and res.x[1] < 1.25 and res.x[1] > 1.22:
        results[-1].append(res.nfev)
    else:
        results[-1].append(0)

    res = minimize(fun, x0, method="Powell", options={"disp": False})
    if res.x[0] < 0.87 and res.x[0] > 0.85 and res.x[1] < 1.25 and res.x[1] > 1.22:
        results[-1].append(res.nfev)
    else:
        results[-1].append(0)

    i = 0
    res = minimize(fun, x0, method="TNC", options={"disp": False, 'gtol':1e-5})
    if res.x[0] < 0.87 and res.x[0] > 0.85 and res.x[1] < 1.25 and res.x[1] > 1.22:
        results[-1].append(i)
    else:
        results[-1].append(0)

    i = 0
    res = minimize(fun, x0, method="CG", options={"disp": False, 'gtol':1e-5})
    if res.x[0] < 0.87 and res.x[0] > 0.85 and res.x[1] < 1.25 and res.x[1] > 1.22:
        results[-1].append(i)
    else:
        results[-1].append(0)
    # results[-1][1].append(res.x)


print(results)

labels = ['Nelder-Mead', 'Powell', "TNC", 'CG']
results = np.asarray(results)
x = np.arange(len(labels))
gap = 0.05
x = np.asarray([i*(gap+1) for i in range(len(labels))])
width = (len(labels) - 7*gap)/(len(labels)*N)  # the width of the bars
print(width)
colors = ['#E89005', '#EC7505', '#D95D39', '#D84A05', '#B8141C']
custom_cycler = cycler(color=colors)
fig, ax = plt.subplots()
ax.set_prop_cycle(custom_cycler)
for i in range(N):
    color = colors[int(i*5/N)]
    ax.bar(x + i*width - (len(results)-1)/2*width, results[i], width, color=color,  edgecolor=color, label=str(i), align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of function evaluation')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)

fig.tight_layout()
plt.savefig("algorithms_perfs.svg", bbox_inches='tight')
plt.savefig("algorithms_perfs.pdf", bbox_inches='tight')
plt.show()
