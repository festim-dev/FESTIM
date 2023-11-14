import matplotlib.pyplot as plt
import numpy as np

times = np.genfromtxt("times_tds.txt")
flux = np.genfromtxt("outgassing_flux_tds.txt")
flux = np.abs(flux)

plt.figure()
plt.plot(times, flux)
plt.xlim(450, 500)
# plt.ylim(top=1e18)
plt.ylim(bottom=0, top=1e19)
# plt.ylim(bottom=1.25e18, top=0.6e19)
plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Time (s)")

plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Time (s)")
plt.show()
