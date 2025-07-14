import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.io import ascii

filename = 'tuc_iso.dat'

data=astropy.io.ascii.read("tuc_iso.dat", format="csv", delimiter=" ", comment=None, header_start=13)
iso_g = data['gmag'][:122]
iso_r = data['rmag'][:122]
# iso_g

plt.figure(figsize=(4,6))
plt.scatter(iso_g-iso_r, iso_g, c=np.arange(len(iso_g)))
plt.colorbar()
#plt.xlim(0,1)
plt.ylim(20, -5)

iso_sp = interp1d(iso_g-iso_r, iso_g)

plt.plot(iso_g-iso_r, iso_sp(iso_g-iso_r), 'k-')

plt.show()