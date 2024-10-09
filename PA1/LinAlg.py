import numpy as np
from typing import Union
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Vector:
    def __init__(self, x: float, y: float, z: float) -> None:
        """
        Initializes a Vector object with x, y, z coordinates.

        Parameters:
        x (float): X coordinate
        y (float): Y coordinate
        z (float): Z coordinate
        """
        self.coords = np.array([x, y, z], dtype=np.float32)

    def __str__(self) -> str:
        """
        String representation of the Vector.
        """
        return f"Vector({self.coords[0]:.2f}, {self.coords[1]:.2f}, {self.coords[2]:.2f})"

    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Adds two Vector objects.

        Args:
        other (Vector): Another Vector.

        Returns:
        Vector: Sum of the two vectors.
        """
        result = self.coords + other.coords
        return Vector(result[0], result[1], result[2])

    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Subtracts two Vector objects.

        Args:
        other (Vector): Another Vector.

        Returns:
        Vector: Difference of the two vectors.
        """
        result = self.coords - other.coords
        return Vector(result[0], result[1], result[2])

    def dot(self, other: 'Vector') -> float:
        """
        Computes the dot product of two Vector objects.

        Args:
        other (Vector): Another Vector.

        Returns:
        float: Dot product of the two vectors.
        """
        return np.dot(self.coords, other.coords)

    def cross(self, other: 'Vector') -> 'Vector':
        """
        Computes the cross product of two Vector objects.

        Args:
        other (Vector): Another Vector.

        Returns:
        Vector: Cross product of the two vectors.
        """
        result = np.cross(self.coords, other.coords)
        return Vector(result[0], result[1], result[2])

    def magnitude(self) -> float:
        """
        Computes the magnitude of the Vector.

        Returns:
        float: Magnitude of the vector.
        """
        return np.linalg.norm(self.coords)
    
    def as_array(self):
        """
        Return the vector as a NumPy array.
        """
        return self.coords




class Frame:
    def __init__(self, r: np.ndarray, t: np.ndarray) -> None:
        """
        Initializes a Frame object with rotation and translation.

        Parameters:
        r (np.ndarray): Rotation matrix (3x3)
        t (np.ndarray): Translation vector (3,)
        """
        self.rotation = np.array(r)
        self.translation = np.array(t)

    def __array__(self):
        """
        Returns the Frame as a 4x4 homogeneous transformation matrix.
        """
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = self.rotation
        out[:3, 3] = self.translation
        return out

    def __str__(self) -> str:
        """
        String representation of the Frame as a numpy array.
        """
        return np.array_str(np.array(self), precision=4, suppress_small=True)

    def inv(self) -> 'Frame':
        """
        Returns the inverse of the Frame.

        Returns:
        Frame: The inverted Frame.
        """
        return Frame(self.rotation.T, -(self.rotation.T @ self.translation))

    def __matmul__(self, other):
        """
        Overloading the @ operator for Frame-to-Frame, Frame-to-Numpy array, or Frame-to-Vector multiplication.
        """
        if isinstance(other, Vector):
            # Apply rotation and translation to the vector
            transformed_coords = (self.rotation @ other.as_array().T).T + self.translation
            # Return a new Vector object with the transformed coordinates
            return Vector(transformed_coords[0], transformed_coords[1], transformed_coords[2])
        if isinstance(other, np.ndarray):
            # Apply rotation and then translation to a numpy array
            return (self.rotation @ other.T).T + self.translation
        elif isinstance(other, Frame):
            # Multiply rotation matrices and apply combined translation
            new_rotation = self.rotation @ other.rotation
            new_translation = self.rotation @ other.translation + self.translation
            return Frame(new_rotation, new_translation)
        elif isinstance(other, Vector):
            # Apply rotation and translation to a Vector object
            return (self.rotation @ other.as_array().T).T + self.translation
        else:
            raise TypeError(f"Unsupported operand type(s) for @: 'Frame' and '{type(other).__name__}'")


def rodrigues_to_rotation_matrix(rvec):
    """Convert a Rodrigues vector to a 3x3 rotation matrix."""
    return R.from_rotvec(rvec).as_matrix()

def residuals(params, source_points, target_points):
    """Compute the residuals (differences) between transformed source_points and target_points."""
    rvec = params[:3]  # First three parameters are the axis-angle representation of rotation
    t = params[3:]  # Last three parameters are the translation vector

    # Compute rotation matrix from the axis-angle representation
    R_matrix = rodrigues_to_rotation_matrix(rvec)

    # Transform source_points using the current R and t
    transformed_source = np.dot(source_points, R_matrix.T) + t

    # Compute residuals (difference between transformed source_points and target_points)
    return (transformed_source - target_points).ravel()  # Flatten to a 1D array for least_squares

def point_cloud_registration_least_squares(target_points, source_points):
    """
    Perform point cloud registration between target_points and source_points using least squares.
    
    Args:
    target_points (np.ndarray): Nx3 array of points from calreadings (e.g., A_vectors, D_vectors, C_vectors).
    source_points (np.ndarray): Nx3 array of points from calbody.
    
    Returns:
    R (np.ndarray): 3x3 optimal rotation matrix.
    t (np.ndarray): 3x1 translation vector.
    """
    # Initial guess for parameters: rotation as [0, 0, 0] (no rotation), translation as [0, 0, 0]
    initial_params = np.zeros(6)

    # Use least squares to minimize the residuals
    result = least_squares(residuals, initial_params, args=(source_points, target_points))

    # Extract the optimal parameters from the result
    rvec_optimal = result.x[:3]
    t_optimal = result.x[3:]

    # Convert the optimal Rodrigues vector to a rotation matrix
    R_optimal = rodrigues_to_rotation_matrix(rvec_optimal)

    return R_optimal, t_optimal

def perform_registration_for_frames(calreadings_frames, calbody_vectors, vector_type):
    """
    Perform point cloud registration for each frame in calreadings, selecting the type of vectors (A, D, or C).
    
    Args:
    calreadings_frames (dict): Dictionary where each key is a frame number and value is a dict of A_vectors, D_vectors, or C_vectors.
    calbody_vectors (list): List of vectors from calbody (corresponding to the selected vector type).
    vector_type (str): The type of vector to use ('A', 'D', or 'C').
    
    Returns:
    dict: Dictionary with frame numbers as keys and (R, t) as values where R is the rotation matrix and t is the translation vector.
    """
    registration_results = {}

    # Convert calbody vectors (for the selected type) to a numpy array
    source_points = np.array([vec.coords for vec in calbody_vectors])

    # Select the appropriate vectors from calreadings_frames based on vector_type
    vector_key = f'{vector_type.lower()}_vectors'  # e.g., 'a_vectors', 'd_vectors', 'c_vectors'

    # Perform registration for each frame
    for frame_num, frame_data in calreadings_frames.items():
        # Check if the vector type exists in the frame data
        if vector_key not in frame_data:
            raise KeyError(f"Vector type '{vector_type}' not found in frame {frame_num}. Available keys: {list(frame_data.keys())}")

        # Extract the target points (e.g., A_vectors, D_vectors, C_vectors) for the current frame
        target_points = np.array([vec.coords for vec in frame_data[vector_key]])

        # Perform point cloud registration using least squares to find R and t
        R_optimal, t_optimal = point_cloud_registration_least_squares(target_points, source_points)

        # Store the result for this frame
        registration_results[frame_num] = Frame(R_optimal, t_optimal)

    return registration_results

def transform_points(frame, source_vectors):
    """
    Apply the transformation (rotation + translation) to a list of source Vector objects.
    
    Args:
    frame (Frame): Frame object containing rotation matrix and translation vector.
    source_vectors (list of Vector): List of Vector objects to be transformed.
    
    Returns:
    list of Vector: List of transformed Vector objects.
    """
    transformed_vectors = []

    # Loop through each Vector in source_vectors and transform using the Frame
    for vector in source_vectors:
        # Apply the Frame transformation to the Vector
        transformed_vector = frame @ vector
        transformed_vectors.append(transformed_vector)

    return transformed_vectors

def compute_rmse(transformed_points, target_points):
    """
    Compute the Root Mean Squared Error (RMSE) between transformed points and target points.
    
    Args:
    transformed_points (np.ndarray): Nx3 array of transformed points.
    target_points (np.ndarray): Nx3 array of target points.
    
    Returns:
    float: The RMSE value.
    """
    return np.sqrt(np.mean((transformed_points - target_points) ** 2))

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

