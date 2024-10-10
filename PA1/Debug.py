import numpy as np
import random
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

def plot_3d_transformed_vs_target(frame_num, transformed_vectors, target_vectors):
    """
    Plot the transformed source vectors and the target vectors in 3D space.
    
    Args:
    frame_num (int): Frame number to display in the title.
    transformed_vectors (list of Vector): List of transformed Vector objects.
    target_vectors (list of Vector): List of target Vector objects.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert transformed_vectors and target_vectors from lists of Vector objects to NumPy arrays
    transformed_points = np.array([vector.as_array() for vector in transformed_vectors])
    target_points = np.array([vector.as_array() for vector in target_vectors])

    # Plot transformed points in red
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c='r', marker='o', label='Transformed points')

    # Plot target points in blue
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='b', marker='^', label='Target points')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend and title
    plt.legend()
    plt.title(f'3D Plot of Transformed points vs Target points (Frame {frame_num})')
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

# Function to generate random points within the box (1,1,1) to (2,2,2)
def generate_random_points(num_points: int):
    points = []
    for _ in range(num_points):
        x = random.uniform(0,200)
        y = random.uniform(0,200)
        z = random.uniform(0,200)
        points.append(LA.Vector(x, y, z))
    return points