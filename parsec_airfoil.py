from math import sqrt, tan, pi
import numpy as np
import scipy.optimize


def get_coef(xte, yte, rle, x_ctrl, y_ctrl, d2ydx2_ctrl, th_ctrl, surface):
    """
    evaluate the PARSEC coefficients
    Assuming leading edge is at (x,y)=(0,0), and sharp trailing edge at (xte, yte)
    """

    # Initialize coefficients
    coef = np.zeros(6)

    if surface == "pressure":
        coef[0] = -sqrt(2*rle)
    elif surface == "suction":
        coef[0] = sqrt(2*rle)
    else:
        raise ValueError(f"surface = {surface} not recognized! It has to be either pressure or suction")
 
    # form linear system
    A = np.array([
                 [xte**1.5, xte**2.5, xte**3.5, xte**4.5, xte**5.5],
                 [x_ctrl ** 1.5, x_ctrl ** 2.5, x_ctrl ** 3.5, x_ctrl ** 4.5, x_ctrl ** 5.5],
                 [1.5*sqrt(xte), 2.5*xte**1.5, 3.5*xte**2.5, 4.5*xte**3.5, 5.5*xte**4.5],
                 [1.5 * sqrt(x_ctrl), 2.5 * x_ctrl ** 1.5, 3.5 * x_ctrl ** 2.5, 4.5 * x_ctrl ** 3.5, 5.5 * x_ctrl ** 4.5],
                 [0.75 * (1 / sqrt(x_ctrl)), 3.75 * sqrt(x_ctrl), 8.75 * x_ctrl ** 1.5, 15.75 * x_ctrl ** 2.5, 24.75 * x_ctrl ** 3.5]
                 ])

    B = np.array([
                 [yte - coef[0]*sqrt(xte)],
                 [y_ctrl - coef[0] * sqrt(x_ctrl)],
                 [tan(th_ctrl * pi / 180) - 0.5 * coef[0] * (1 / sqrt(xte))],
                 [-0.5 * coef[0] * (1 / sqrt(x_ctrl))],
                 [d2ydx2_ctrl + 0.25 * coef[0] * x_ctrl ** (-1.5)]
                 ])
    
    # X = np.linalg.solve(A, B)  # Solve system of linear equations.  Not robust against singular A matrix
    X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  # solve least-squares problem in case A is ill-conditioned

    coef[1:6] = X[0:5, 0]

    return coef


def eval_coord_from_coef(x_arr, coef):
    """
    evaluate x,y coordinates based on parsec polynomials and coefficients
    :param x_arr:
    :param coef:
    :return:
    """
    pwrs = (1/2, 3/2, 5/2, 7/2, 9/2, 11/2)
    y = np.zeros_like(x_arr)
    for j, c in enumerate(coef):
        y = y + c * x_arr ** pwrs[j]
    xy = np.array([x_arr, y]).T
    return xy


def get_coord(xte, yte, parsec_params, x_arr=None):
    """
    assuming leading edge is at (x,y)=(0,0)
    :param xte:
    :param yte:
    :param parsec_params:
    :param x_arr: listed in XFoil format, i.e. counter-clockwise starting from upper trailing edge, wrapping around leading edge and ending at lower trailing edge
    :return: xy_arr: listed in XFoil format, i.e. counter-clockwise starting from upper trailing edge, wrapping around leading edge and ending at lower trailing edge
    """

    if x_arr is None:  # if x_arr is not provided
        npts = 161
        x_arr = (1 + np.cos(np.linspace(0, 2*np.pi, npts))) / 2  # cosine spacing
        x_arr *= xte

    xle = 0.0  # Assuming leading edge is at x,y=0,0
    tol = 1e-3
    assert np.isclose(xle, np.min(x_arr), rtol=tol, atol=tol)
    np.isclose(xte, x_arr[-1], rtol=tol, atol=tol)
    np.isclose(xte, x_arr[0], rtol=tol, atol=tol)

    coef_suc = get_coef(xte, yte, parsec_params["rle"],
                        parsec_params["x_suc"], parsec_params["y_suc"],
                        parsec_params["d2ydx2_suc"], parsec_params["th_suc"],
                        'suction')

    coef_pre = get_coef(xte, yte, parsec_params["rle"],
                        parsec_params["x_pre"], parsec_params["y_pre"],
                        parsec_params["d2ydx2_pre"], parsec_params["th_pre"],
                        'pressure')

    j_le = np.argmin(x_arr)
    xy_suc = eval_coord_from_coef(x_arr[:j_le], coef_suc)
    xy_pre = eval_coord_from_coef(x_arr[j_le:], coef_pre)
    # indexing here avoids repeating the leading edge point

    xy_arr = np.concatenate((xy_suc, xy_pre), axis=0)

    return xy_arr


def parsec_params_list_to_dict(var):
    """
    convert parsec parameter array to dictionary
    :param var:
    :return:
    """
    parsec_params = dict()
    parsec_params["rle"] = var[0]
    parsec_params["x_pre"] = var[1]
    parsec_params["y_pre"] = var[2]
    parsec_params["d2ydx2_pre"] = var[3]
    parsec_params["th_pre"] = var[4]
    parsec_params["x_suc"] = var[5]
    parsec_params["y_suc"] = var[6]
    parsec_params["d2ydx2_suc"] = var[7]
    parsec_params["th_suc"] = var[8]
    return parsec_params


def obj_fcn_airfoil_inversion(var, xy_ref):
    """
    objective function to minimize: discrepancy between reference coordinates and parsec airfoil coordinates
    :param var: variable to optimize
    :param xy_ref: n-by-2 array of reference coordinates following XFoil format (CCW)
    :return:
    """
    xte, yte = 0.5 * (xy_ref[0, :] + xy_ref[-1, :])

    parsec_params = parsec_params_list_to_dict(var)

    # Evaluate suction (upper) surface coefficients
    coef_suc = get_coef(xte, yte, parsec_params["rle"],
                        parsec_params["x_suc"], parsec_params["y_suc"],
                        parsec_params["d2ydx2_suc"], parsec_params["th_suc"],
                        'suction')

    # Evaluate pressure (lower) surface coefficients
    coef_pre = get_coef(xte, yte, parsec_params["rle"],
                        parsec_params["x_pre"], parsec_params["y_pre"],
                        parsec_params["d2ydx2_pre"], parsec_params["th_pre"],
                        'pressure')

    j_le = np.argmin(xy_ref[:, 0])
    xy_suc = eval_coord_from_coef(xy_ref[0:j_le, 0], coef_suc)
    xy_pre = eval_coord_from_coef(xy_ref[j_le:, 0], coef_pre)
    xy_parsec = np.concatenate((xy_suc, xy_pre), axis=0)

    # scale for normalization
    y_scale = np.max(xy_ref[: 1]) - np.min(xy_ref[:, 1])
    # y_scale = 1e-2  # an arbitrary scale
    # y_scale = np.maximum(1e-2, np.fabs(xy_ref[:, 1]))
    # y_scale = np.max(xy_ref[: 1]) - np.min(xy_ref[:, 1]) * (2 - np.cos(2*pi*xy_ref[:, 0]/xte))

    npts = xy_ref.shape[0]

    return np.sum(((xy_ref[:, 1] - xy_parsec[:, 1]) / y_scale)**2.0) / npts


def get_parsec_params_bounds(xte):
    """
    get a dictionary of (lower, upper) bounds pair for parsec parameters
    :param xte:
    :return:
    """
    eps_zero = 1e-6
    # eps_inf = np.inf
    eps_inf = 1e5
    parsec_params_bounds = dict()
    parsec_params_bounds["rle"] = (eps_zero, eps_inf)
    parsec_params_bounds["x_pre"] = (eps_zero, xte)
    parsec_params_bounds["y_pre"] = (-xte, xte)
    parsec_params_bounds["d2ydx2_pre"] = (-eps_inf, eps_inf)
    parsec_params_bounds["th_pre"] = (-89., 89.)
    parsec_params_bounds["x_suc"] = (eps_zero, xte)
    parsec_params_bounds["y_suc"] = (-xte, xte)
    parsec_params_bounds["d2ydx2_suc"] = (-eps_inf, eps_inf)
    parsec_params_bounds["th_suc"] = (-89., 89.)

    lb = [val[0] for val in parsec_params_bounds.values()]  # assuming dictionary follows the order of key addition
    ub = [val[1] for val in parsec_params_bounds.values()]  # assuming dictionary follows the order of key addition

    return parsec_params_bounds, lb, ub


def infer_parsec_params(xte, var_init, xy_ref):
    """
    infer parsec parameters given reference/target airfoil coordinates
    """

    _, lb, ub = get_parsec_params_bounds(xte)

    if True:
        bounds = scipy.optimize.Bounds(lb, ub, keep_feasible=True)
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

    parsec_params = parsec_params_list_to_dict(opt_result.x)
    return parsec_params


def main():
    import sys
    if sys.platform == 'darwin':  # MacOS
        import matplotlib
        matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import os

    path = './data/'

    # reference coordinates
    airfoilname = "RAE2822"
    # airfoilname = "b737a-il"
    # airfoilname = "e387"
    # airfoilname = "whitcomb"
    # airfoilname = "nasasc2-0714"
    fname_input_airfoil = os.path.join(path, f"{airfoilname}.txt")
    xy_ref = np.genfromtxt(fname_input_airfoil, dtype=float, skip_header=1)

    # airfoil inversion
    airfoilname_inverse = f"{airfoilname}_parsec"

    # xte, yte = 1.0, 0.0  # normalized to chord = 1
    xte, yte = 0.5 * (xy_ref[0, :] + xy_ref[-1, :])

    # initial guess of parsec parameters
    parsec_params_init = dict()
    print("FYI, here is the bounds for parsec parameters:")
    print(f"{get_parsec_params_bounds(xte)[0]}")

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
    xy_coords_init = get_coord(xte, yte, parsec_params_init)

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
    parsec_params_init = infer_parsec_params(xte, var_init, xy_ref)
    xy_coords = get_coord(xte, yte, parsec_params_init)

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


if __name__ == '__main__':
    main()
