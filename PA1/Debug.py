import numpy as np
import LinAlg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def vectors_to_numpy(points):
    """
    Converts a list of Vector objects to a numpy array.
    """
    return np.array([point.as_array() for point in points])

def create_transformation_matrix(theta, phi, translation):
    """
    Create a Frame object (transformation matrix) from given rotation angles and translation vector.

    Parameters:
    theta (float): Rotation angle around the Z-axis in radians.
    phi (float): Rotation angle around the Y-axis in radians.
    translation (Vector): Translation vector of class Vector.

    Returns:
    Frame: Frame object containing the rotation and translation.
    """
    # Rotation matrix for theta (around Z-axis)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Rotation matrix for phi (around Y-axis)
    R_y = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])

    # Combined rotation matrix (3x3)
    R = R_y @ R_z

    # Translation vector (3,)
    t = translation.as_array()

    # Create and return the Frame object
    return LA.Frame(R, t)


def plot_original_vs_transformed(original_points, transformed_points):
    """
    Plots the original points vs the transformed points in a 3D scatter plot.

    Parameters:
    original_points (list): List of original Vector objects.
    transformed_points (list): List of transformed Vector objects.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates for original points
    original_x = [point.coords[0] for point in original_points]
    original_y = [point.coords[1] for point in original_points]
    original_z = [point.coords[2] for point in original_points]

    # Extract coordinates for transformed points
    transformed_x = [point.coords[0] for point in transformed_points]
    transformed_y = [point.coords[1] for point in transformed_points]
    transformed_z = [point.coords[2] for point in transformed_points]

    # Plot original points (in blue)
    ax.scatter(original_x, original_y, original_z, c='b', label='Original Points')

    # Plot transformed points (in red)
    ax.scatter(transformed_x, transformed_y, transformed_z, c='r', label='Transformed Points')

    # Label the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.legend()
    plt.title('Original Points vs Transformed Points')
    plt.show()



def visualize_H0_vectors(H0_vectors):
    """
    Visualize the H0 vectors in 3D space, with each vector originating from the origin.
    
    Parameters:
    H0_vectors (list of Vector): List of Vector objects representing H0 vectors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Origin point (0, 0, 0) for all vectors
    origin = np.array([0, 0, 0])

    # Plot each vector from the origin to the point defined by the vector
    for idx, vec in enumerate(H0_vectors):
        vec_array = vec.as_array()
        ax.plot([origin[0], vec_array[0]], [origin[1], vec_array[1]], [origin[2], vec_array[2]], color='b', alpha=0.7, label=f"Frame {idx + 1}" if idx == 0 else "")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()



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