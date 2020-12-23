import os

import numpy as np

import sys
if sys.platform == 'darwin':  # MacOS
    import matplotlib
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import parsec_airfoil as parsec


# I/O files path
path = './data/'

airfoilname = "RAE2822"

# TE & LE of airfoil (normalized, chord = 1)
xte = 1.0
yte = 0.0

parsec_params = dict()
# # the following ones are close to RAE2822
parsec_params["rle"] = 8.300e-03
parsec_params["x_pre"] = 3.441e-01
parsec_params["y_pre"] = -5.880e-02
parsec_params["d2ydx2_pre"] = 7.018e-01
parsec_params["th_pre"] = -1.
parsec_params["x_suc"] = 4.312e-01
parsec_params["y_suc"] = 6.290e-02
parsec_params["d2ydx2_suc"] = -4.273e-01
parsec_params["th_suc"] = -12

xy_coords = parsec.get_coord(xte, yte, parsec_params, x_arr=None)

# save to coordinate file
fpath = os.path.join(path, f'{airfoilname}.dat')
np.savetxt(fpath, xy_coords, delimiter=" ", newline="\n")

# Plot airfoil contour
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.plot(xy_coords[:, 0], xy_coords[:, 1], linewidth=2, label="RAE2822 imitate")

pparray_ref = np.genfromtxt(f"{path}RAE2822.txt", dtype=float, skip_header=1)
ax.plot(pparray_ref[:, 0], pparray_ref[:, 1], "k--", label="Reference RAE 2822")

ax.legend()
ax.grid(True)
ax.set_xlim([0, 1])
ax.axis('equal')
#plt.yticks([])
ax.set_xticks(np.arange(0, 1.1, 0.1))

parnv = [f"{name}={val:.4e}".replace('\n', '') for name, val in parsec_params.items()]
parnv = [', '.join(parnv[:5]),
         ', '.join(parnv[5:])]
ax.set_title(f"PARSEC airfoil with parameters:\n{parnv[0]}\n{parnv[1]}")
fig.tight_layout()

fig.savefig(os.path.join(path, f'{airfoilname}.png'))

plt.show()
