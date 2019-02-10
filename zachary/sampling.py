import numpy as np
import scipy.interpolate as si


def bspline(cv, n=100, degree=3, periodic=False):
    # If periodic, extend the point array by count+degree+1
    if degree < 1:
        raise ValueError('degree cannot be less then 1!')
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)

    # If opened, prevent degree from exceeding count-1
    else:
        if count < degree + 1:
            raise ValueError('number of cvs must be higher than degree + 1')

    # Calculate knot vector
    if periodic:
        kv = np.arange(0 - degree, count + degree + degree - 1, dtype='int')
    else:
        kv = np.array([0] * degree + list(range(count - degree + 1)) + [count - degree] * degree, dtype='int')

    # Calculate query range
    u = np.linspace(periodic, (count - degree), n)

    # Calculate result
    arange = np.arange(len(u))
    points = np.zeros((len(u), cv.shape[1]))
    for i in range(cv.shape[1]):
        points[arange, i] = si.splev(u, (kv, cv[:, i], degree))

    return points


def sample_z(z_dims, mean, std, num_cv, resolution, degree, is_periodic):
    # Generates splines of random lengths in z_dims dimensions
    # num_cv = np.random.randint(64, 128)
    cv = np.random.normal(mean, std, (num_cv, z_dims))
    num_points = num_cv * resolution
    spline = bspline(cv, num_points, degree, is_periodic)
    return spline
