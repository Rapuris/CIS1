import numpy as np
from scipy.optimize import least_squares

def fit_sphere(points):
    """
    Fits a sphere to a set of 3D points using nonlinear least squares optimization.

    Returns:
    center (numpy.ndarray): The (x, y, z) coordinates of the sphere's center.
    radius (float): The radius of the sphere.
    residuals (float): The sum of squared residuals of the fit.
    """

    points = np.asarray(points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x0 = np.mean(x)
    y0 = np.mean(y)
    z0 = np.mean(z)
    r0 = np.mean(np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2))

    initial_guess = np.array([x0, y0, z0, r0])

    def residuals(params, x, y, z):
        xc, yc, zc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2) - r

    result = least_squares(residuals, initial_guess, args=(x, y, z))
    xc, yc, zc, r = result.x
    residual_sum = 2 * result.cost 

    center = np.array([xc, yc, zc])
    radius = r

    return center, radius, residual_sum