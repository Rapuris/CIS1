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
