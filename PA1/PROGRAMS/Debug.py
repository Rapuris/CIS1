import numpy as np
import random
import LinAlg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

"""
This file contains helper functions for debugging and visualization purposes. Not used in main code.
"""

def vectors_to_numpy(points):
    """
    Converts a list of Vector objects to a numpy array.
    """
    return np.array([point.as_array() for point in points])

def create_transformation_matrix(theta, phi, translation):
    """
    Create a Frame object (transformation matrix)
    """
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])


    R = R_y @ R_z
    t = translation.as_array()
    return LA.Frame(R, t)


def plot_original_vs_transformed(original_points, transformed_points):
    """
    Plots the original points vs the transformed points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    original_x = [point.coords[0] for point in original_points]
    original_y = [point.coords[1] for point in original_points]
    original_z = [point.coords[2] for point in original_points]

    transformed_x = [point.coords[0] for point in transformed_points]
    transformed_y = [point.coords[1] for point in transformed_points]
    transformed_z = [point.coords[2] for point in transformed_points]

    ax.scatter(original_x, original_y, original_z, c='b', label='Original Points')
    ax.scatter(transformed_x, transformed_y, transformed_z, c='r', label='Transformed Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.title('Original vs Transformed Points')
    plt.show()

def plot_3d_transformed_vs_target(frame_num, transformed_vectors, target_vectors):
    """
    Plot the transformed source vectors and the target vectors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    transformed_points = np.array([vector.as_array() for vector in transformed_vectors])
    target_points = np.array([vector.as_array() for vector in target_vectors])


    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c='r', marker='o', label='Transformed points')
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='b', marker='^', label='Target points')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.title(f'Transformed points vs Target points (Frame {frame_num})')
    plt.show()


def visualize_vectors(d_vectors, a_vectors, c_vectors):
    """
    Visualize the vectors from CALBODY.TXT
    """
    d_coords = np.array([vec.coords for vec in d_vectors])
    a_coords = np.array([vec.coords for vec in a_vectors])
    c_coords = np.array([vec.coords for vec in c_vectors])

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(d_coords[:, 0], d_coords[:, 1], d_coords[:, 2], c='r', marker='o')
    ax1.set_title('d_i Vectors (Optical Markers on base of EM Tracker)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot a_vectors
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(a_coords[:, 0], a_coords[:, 1], a_coords[:, 2], c='g', marker='^')
    ax2.set_title('a_i Vectors (Optical Markers on Calibration Object)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Plot c_vectors
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(c_coords[:, 0], c_coords[:, 1], c_coords[:, 2], c='b', marker='s')
    ax3.set_title('c_i Vectors (EM Markers on Calibration Object)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Show plots
    plt.tight_layout()
    plt.show()

def visualize_H0_vectors(H0_vectors):
    """
    Visualize the H0 vectors with tail
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    origin = np.array([0, 0, 0])

    for idx, vec in enumerate(H0_vectors):
        vec_array = vec.as_array()
        ax.plot([origin[0], vec_array[0]], [origin[1], vec_array[1]], [origin[2], vec_array[2]], color='b', alpha=0.7, label=f"Frame {idx + 1}" if idx == 0 else "")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()

def generate_random_points(num_points: int, noise = 0.0):
    """
    Generate random points with or without noise
    """
    points = []
    for _ in range(num_points):
        x = random.uniform(0,200)
        y = random.uniform(0,200)
        z = random.uniform(0,200)

        x = x + np.random.normal(-noise, noise)
        y = y + np.random.normal(-noise, noise)
        z = z + np.random.normal(-noise, noise)

        points.append(LA.Vector(x, y, z))
    return points


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