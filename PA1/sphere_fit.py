import numpy as np
from scipy.optimize import least_squares

def fit_sphere(points):
    """
    Fits a sphere to a set of 3D points using nonlinear least squares optimization.

    Parameters:
    points (array-like): An Nx3 array or list of (x, y, z) coordinates.

    Returns:
    center (numpy.ndarray): The (x, y, z) coordinates of the sphere's center.
    radius (float): The radius of the sphere.
    residuals (float): The sum of squared residuals of the fit.
    """

    # Convert input to a NumPy array
    points = np.asarray(points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Initial guess for the sphere's center (mean of the points)
    x0 = np.mean(x)
    y0 = np.mean(y)
    z0 = np.mean(z)
    r0 = np.mean(np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2))

    initial_guess = np.array([x0, y0, z0, r0])

    # Define the residuals function
    def residuals(params, x, y, z):
        xc, yc, zc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2) - r

    # Perform the least squares optimization
    result = least_squares(residuals, initial_guess, args=(x, y, z))

    # Extract the optimized parameters
    xc, yc, zc, r = result.x
    residual_sum = 2 * result.cost  # total sum of squared residuals

    center = np.array([xc, yc, zc])
    radius = r

    return center, radius, residual_sum



# Basic usage:
if __name__ == "__main__":
    # Generate sample points on the surface of a sphere centered at (10, 15, 20) with radius 5
    import random
    import math

    def generate_sphere_points(center, radius, num_points, noise=0.0):
        """Generates points on the surface of a sphere with optional noise."""
        x0, y0, z0 = center
        points = []
        for _ in range(num_points):
            theta = random.uniform(0, 2 * math.pi)
            phi = math.acos(random.uniform(-1, 1))
            x = x0 + radius * math.sin(phi) * math.cos(theta)
            y = y0 + radius * math.sin(phi) * math.sin(theta)
            z = z0 + radius * math.cos(phi)
            # Add optional Gaussian noise
            x += np.random.normal(0, noise)
            y += np.random.normal(0, noise)
            z += np.random.normal(0, noise)
            points.append((x, y, z))
        return points

    # Generate sample data with some noise
    true_center = (10, 15, 20)
    true_radius = 5
    num_points = 10
    noise_level = 0.1  # Adjust noise level as needed
    sample_points = generate_sphere_points(true_center, true_radius, num_points, noise=noise_level)

    # Fit a sphere to the generated points
    estimated_center, estimated_radius, residuals = fit_sphere(sample_points)

    # Output the results
    print("Estimated Center of Sphere: ({:.4f}, {:.4f}, {:.4f})".format(*estimated_center))
    print("Estimated Radius of Sphere: {:.4f}".format(estimated_radius))
    print("Sum of Squared Residuals: {:.4e}".format(residuals))
    print("\nTrue Center: ({}, {}, {})".format(*true_center))
    print("True Radius: {}".format(true_radius))