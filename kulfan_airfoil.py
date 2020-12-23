from math import factorial
import numpy as np
import scipy.optimize


def get_airfoil_coord(wu, wl, yute, ylte, x=None):
    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1

    # Create x coordinate
    if x is None:  # if x_arr is not provided
        N = 161  #TODO: hard coded for now
        # x is listed in XFoil format, i.e. counter-clockwise starting from upper trailing edge, wrapping around leading edge and ending at lower trailing edge
        x = 0.5 * (1 + np.cos(np.linspace(0, 2*np.pi, N)))

    j_le = np.argmin(x)
    xu = x[:j_le]  # upper surface x-coordinates
    xl = x[j_le:]  # lower surface x-coordinates

    yu = eval_shape_fcn(wu, xu, N1, N2, yute)  # upper surface y-coordinates
    yl = eval_shape_fcn(wl, xl, N1, N2, ylte)  # lower surface y-coordinates

    y = np.concatenate([yu, yl])

    return np.array([x, y]).T


def eval_shape_fcn(w, x, N1, N2, yte):
    """
    compute class and shape function
    :param w:
    :param x:
    :param N1:
    :param N2:
    :param yte: trailing edge y coordinate
    :return:
    """
    C = x**N1 * (1-x)**N2

    n = len(w) - 1  # degree of Bernstein polynomials

    S = np.zeros_like(x)
    for j in range(0, n+1):
        K = factorial(n)/(factorial(j)*(factorial(n-j)))
        S += w[j]*K*x**j * ((1-x)**(n-j))

    return C * S + x * yte


def list_to_kulfan_params(var):
    nw = len(var) // 2
    wu = var[0:nw]
    wl = var[nw:(2*nw)]

    assert len(var) == (2 * nw)
    return wu, wl


def obj_fcn_airfoil_inversion(var, xy_ref):
    """
    objective function to minimize: discrepancy between reference coordinates and parametrized airfoil coordinates
    :param var: variable to optimize
    :param xy_ref: n-by-2 array of reference coordinates following XFoil format (CCW)
    :return:
    """
    wu, wl = list_to_kulfan_params(var)
    yute = xy_ref[0, 1]
    ylte = xy_ref[-1, 1]

    xy_kulfan = get_airfoil_coord(wu, wl, yute, ylte, x=xy_ref[:, 0])

    # scale for normalization
    y_scale = np.max(xy_ref[: 1]) - np.min(xy_ref[:, 1])
    # y_scale = 1e-2  # an arbitrary scale
    # y_scale = np.maximum(1e-2, np.fabs(xy_ref[:, 1]))
    # y_scale = np.max(xy_ref[: 1]) - np.min(xy_ref[:, 1]) * (2 - np.cos(2*pi*xy_ref[:, 0]/xte))

    npts = xy_ref.shape[0]

    return np.sum(((xy_ref[:, 1] - xy_kulfan[:, 1]) / y_scale)**2.0) / npts


def infer_kulfan_params(xy_ref, var_init):
    """
    infer kulfan parameters given reference/target airfoil coordinates
    """
    nvar = len(var_init)
    lb = [-2.] * nvar
    ub = [2.] * nvar

    if True:
        bounds = scipy.optimize.Bounds(lb, ub, keep_feasible=False)
        options_opt = {"maxiter": 1e6}

        opt_method = "SLSQP"
        # opt_method = "L-BFGS-B"
        # opt_method = "TNC"
        opt_result = scipy.optimize.minimize(obj_fcn_airfoil_inversion, var_init, args=xy_ref,
                                             method=opt_method, tol=1e-16, bounds=bounds, options=options_opt)
    else:
        obj_fcn = lambda x: obj_fcn_airfoil_inversion(x, xy_ref)
        opt_result = scipy.optimize.dual_annealing(obj_fcn, bounds=list(zip(lb, ub)), seed=1234)

    print(f"Optimization result:\n{opt_result}")

    return list_to_kulfan_params(opt_result.x)


def main():
    import sys
    if sys.platform == 'darwin':  # MacOS
        import matplotlib
        matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import os

    path = './data/'

    # reference coordinates
    # airfoilname = "RAE2822"
    # airfoilname = "b737a-il"
    airfoilname = "e387"
    # airfoilname = "whitcomb"
    # airfoilname = "nasasc2-0714"
    fname_input_airfoil = os.path.join(path, f"{airfoilname}.txt")
    xy_ref = np.genfromtxt(fname_input_airfoil, dtype=float, skip_header=1)

    airfoilname_inverse = f"{airfoilname}_kulfan"

    # fix trailing edge y coordinates based on reference airfoil coordinates
    yute = xy_ref[0, 1]
    ylte = xy_ref[-1, 1]

    poly_deg = 5
    wu = [0.1] * (poly_deg + 1)
    wl = [-0.1] * (poly_deg + 1)

    var_init = list()
    var_init.extend(wu)
    var_init.extend(wl)

    xy_coords_init = get_airfoil_coord(wu, wl, yute, ylte)
    # plot initial guess of airfoil
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot(xy_ref[:, 0], xy_ref[:, 1], "k.--", label="Input")
    ax.plot(xy_coords_init[:, 0], xy_coords_init[:, 1], "-.", linewidth=2, label="Initial guess")

    ax.legend()
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.axis('equal')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    plt.show(block=True)

    # infer Kulfan parameters by optimization
    wu, wl = infer_kulfan_params(xy_ref, var_init)

    # get inferred Kulfan airfoil coordinates
    xy_coords = get_airfoil_coord(wu, wl, yute, ylte)

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

    ax.set_title(f"Kulfan airfoil with parameters: yute={yute:.4e}, ylte={ylte:.4e}\n"
                 f"wu={wu}\nwl={wl}")

    fig.tight_layout()

    fig.savefig(os.path.join(path, f'{airfoilname_inverse}.png'))


if __name__ == '__main__':
    main()


