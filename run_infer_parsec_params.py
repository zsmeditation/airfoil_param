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

# reference coordinates
airfoilname = "RAE2822"
# airfoilname = "b737a-il"
# airfoilname = "e387"
# airfoilname = "e387_refine"
# airfoilname = "whitcomb"
# airfoilname = "nasasc2-0714"
fname_input_airfoil = os.path.join(path, f"{airfoilname}.txt")
xy_ref = np.genfromtxt(fname_input_airfoil, dtype=float, skip_header=1)

j_le = np.argmin(xy_ref[:, 0])
x_arr_ref = xy_ref[j_le:, 0]  # extract x coordinates in ascending form starting from 0.0

# airfoil inversion
airfoilname_inverse = f"{airfoilname}_inverse"

# TE & LE of airfoil
# xte, yte = 1.0, 0.0  # normalized to chord = 1
xte, yte = 0.5 * (xy_ref[0, :] + xy_ref[-1, :])

# initial guess of parsec parameters
parsec_params_init = dict()
print("FYI, here is the bounds for parsec parameters:")
print(f"{parsec.get_parsec_params_bounds(xte)[0]}")

if False:
    # # the following ones are close to RAE2822
    parsec_params_init["rle"] = 8.300e-03
    parsec_params_init["x_pre"] = 3.441e-01
    parsec_params_init["y_pre"] = -5.880e-02
    parsec_params_init["d2ydx2_pre"] = 7.018e-01
    parsec_params_init["th_pre"] = -1.
    parsec_params_init["x_suc"] = 4.312e-01
    parsec_params_init["y_suc"] = 6.290e-02
    parsec_params_init["d2ydx2_suc"] = -4.273e-01
    parsec_params_init["th_suc"] = -12
else:
    # the following parameters are quite arbitrary
    parsec_params_init["rle"] = 4.300e-01
    parsec_params_init["x_pre"] = 5.441e-01
    parsec_params_init["y_pre"] = -5.880e-01
    parsec_params_init["d2ydx2_pre"] = 7.018e-01
    parsec_params_init["th_pre"] = 1.
    parsec_params_init["x_suc"] = 7.312e-01
    parsec_params_init["y_suc"] = 0.02
    parsec_params_init["d2ydx2_suc"] = -4.273e-01
    parsec_params_init["th_suc"] = -1.

var_init = [val for val in parsec_params_init.values()]  # assuming dictionary follows the order of key addition

# plot initial guess of airfoil
xy_coords_init = parsec.get_coord(xte, yte, parsec_params_init)  #TODO: fix upper & lower x

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.plot(xy_coords_init[:, 0], xy_coords_init[:, 1], "-.", linewidth=2, label="Initial guess")
ax.plot(xy_ref[:, 0], xy_ref[:, 1], "k.--", label="Input")

ax.legend()
ax.grid(True)
ax.set_xlim([0, 1])
ax.axis('equal')
ax.set_xticks(np.arange(0, 1.1, 0.1))
plt.show(block=True)

# infer parsec parameters by optimization
parsec_params_init = parsec.infer_parsec_params(xte, var_init, xy_ref)
xy_coords = parsec.get_coord(xte, yte, parsec_params_init)

# save to coordinate file
fpath = os.path.join(path, f'{airfoilname_inverse}.dat')
np.savetxt(fpath, xy_coords, delimiter=" ", newline="\n")

# plot airfoil
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.plot(xy_coords[:, 0], xy_coords[:, 1], marker=".", linewidth=2, label=airfoilname_inverse)
ax.plot(xy_coords_init[:, 0], xy_coords_init[:, 1], "-.", linewidth=2, label="Initial guess")
ax.plot(xy_ref[:, 0], xy_ref[:, 1], "k.--", label="Input")

ax.legend()
ax.grid(True)
ax.set_xlim([0, 1])
ax.axis('equal')
ax.set_xticks(np.arange(0, 1.1, 0.1))

parnv = [f"{name}={val:.4e}".replace('\n', '') for name, val in parsec_params_init.items()]
parnv = [', '.join(parnv[:5]),
         ', '.join(parnv[5:])]
ax.set_title(f"PARSEC airfoil with parameters:\n{parnv[0]}\n{parnv[1]}")

fig.tight_layout()

fig.savefig(os.path.join(path, f'{airfoilname_inverse}.png'))
