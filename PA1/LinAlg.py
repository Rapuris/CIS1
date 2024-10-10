import numpy as np
from typing import Union
from scipy.optimize import least_squares
from scipy.optimize import lsq_linear
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
        Frame object initialized with rotation and translation.

        Parameters:
        r (np.ndarray): Rotation matrix (3x3)
        t (np.ndarray): Translation vector (3x1)
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
            transformed_coords = (self.rotation @ other.as_array().T).T + self.translation
            return Vector(transformed_coords[0], transformed_coords[1], transformed_coords[2])
        if isinstance(other, np.ndarray):
            return (self.rotation @ other.T).T + self.translation
        elif isinstance(other, Frame):
            new_rotation = self.rotation @ other.rotation
            new_translation = self.rotation @ other.translation + self.translation
            return Frame(new_rotation, new_translation)
        elif isinstance(other, Vector):
            return (self.rotation @ other.as_array().T).T + self.translation
        else:
            raise TypeError(f"Unsupported operand type(s) for @: 'Frame' and '{type(other).__name__}'")


def rodrigues_to_rotation_matrix(rvec):
    """Convert a Rodrigues vector to a 3x3 rotation matrix."""
    return R.from_rotvec(rvec).as_matrix()

def residuals(params, source_points, target_points):
    """Compute the residuals (differences) between transformed source_points and target_points."""
    rvec = params[:3]  #rodigues vector
    t = params[3:]  

    R_matrix = rodrigues_to_rotation_matrix(rvec)
    transformed_source = np.dot(source_points, R_matrix.T) + t

    return (transformed_source - target_points).ravel() 

def point_cloud_registration(target_points, source_points):
    """
    Perform point cloud registration between target_points and source_points using least squares.
    
    Args:
    target_points (np.ndarray): Nx3 array of points from calreadings (e.g., A_vectors, D_vectors, C_vectors).
    source_points (np.ndarray): Nx3 array of points from calbody. (e.g., a_vectors, d_vectors, c_vectors).
    
    Returns:
    R (np.ndarray): 3x3 optimal rotation matrix.
    t (np.ndarray): 3x1 translation vector.
    """

    # Initial guess for parameters: rotation = I, translation = difference between midpoints
    initial_params = np.zeros(6)
    initial_t = compute_midpoint(target_points) - compute_midpoint(source_points)
    initial_params[3:] = initial_t


    result = least_squares(residuals, initial_params, args=(source_points, target_points))

    rvec_optimal = result.x[:3]
    t_optimal = result.x[3:]
    R_optimal = rodrigues_to_rotation_matrix(rvec_optimal)
    return R_optimal, t_optimal

def perform_calibration_registration(calreadings_frames, calbody_vectors, vector_type):
    """
    Performs point cloud registration for each frame in calreadings, selecting the type of vectors (A, D, or C).
    
    Args:
    calreadings_frames (dict): Dictionary where each key is a frame number and value is a dict of A_vectors, D_vectors, or C_vectors.
    calbody_vectors (list): List of vectors from calbody (corresponding to the selected vector type).
    vector_type (str): The type of vector to use ('A', 'D', or 'C',).
    
    Returns:
    dict: Dictionary with frame numbers as keys and (R, t) as values where R is the rotation matrix and t is the translation vector.
    """
    registration_results = {}

    source_points = np.array([vec.coords for vec in calbody_vectors])

    vector_key = f'{vector_type}_vectors'  # e.g., 'A_vectors', 'D_vectors', 'C_vectors'
    
    for frame_num, frame_data in calreadings_frames.items():

        if vector_key not in frame_data:
            raise KeyError(f"Vector type '{vector_type}' not found in frame {frame_num}. Available keys: {list(frame_data.keys())}")

        target_points = np.array([vec.coords for vec in frame_data[vector_key]])
        R_optimal, t_optimal = point_cloud_registration(target_points, source_points)
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

    for vector in source_vectors:
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

def compute_C_expected(F_D_dict, F_A_dict, c_vectors):
    """
    Compute C(expected) for each frame using the Frame class and the formula C(expected) = F_D^−1 • F_A • c_i.
    
    Args:
    F_D_list (list of Frame): List of Frame objects F_D for each frame.
    F_A_list (list of Frame): List of Frame objects F_A for each frame.
    c_vectors (np.ndarray): Nx3 or Nx4 array of c_i vectors (Nx3 for 3D or Nx4 for homogeneous 3D).

    Returns:
    dict: A dictionary where keys are frame numbers and values are Nx3 or Nx4 arrays of C_expected vectors.
    """
    assert F_D_dict.keys() == F_A_dict.keys(), "F_D_dict and F_A_dict must have the same frame numbers"

    C_expected_results = {}

    for frame_num in F_D_dict.keys():
        F_D = F_D_dict[frame_num]
        F_A = F_A_dict[frame_num]

        if not isinstance(F_D, Frame) or not isinstance(F_A, Frame):
            raise TypeError(f"Expected F_D and F_A to be Frame objects, but got {type(F_D)} and {type(F_A)}")


        F_D_inv = F_D.inv()
        C_expected_list = []
        for c_i in c_vectors:
            F_A_c_i = F_A @ c_i
            C_expected = F_D_inv @ F_A_c_i
            C_expected_list.append(C_expected)
        C_expected_results[frame_num] = np.array(C_expected_list)

    return C_expected_results
    
def compute_midpoint(observations):
    """
    Compute the midpoint from the observed points.
    observations: List of 3D points (Nx3 array).
    Returns the midpoint.
    """
    return np.mean(observations, axis=0)

def compute_centroid_vectors(frames_data, vector_type):
    """
    Create a list of centroid vectors by computing the midpoint of vectors for each frame.

    Parameters:
    frames_data (dict): Dictionary containing frame data with N_* for each frame.
    vector_type (str): The type of vector to use ('H' or 'G').

    Returns:
    list of Vector: List of vectors for each frame.
    """

    vector_key = f'{vector_type}_vectors'
    centroid_vectors = []
    for frame_num, frame_data in frames_data.items():
        coords = np.array([vec.coords for vec in frame_data[vector_key]])
        centroid_coords = compute_midpoint(coords)
        centroid = Vector(*centroid_coords)
        centroid_vectors.append(centroid)

    return centroid_vectors

def compute_local_marker_vectors(frames_data, vector_type):
    """
    Compute the local marker vectors (g_j or h_j) for each frame, where the vectors
    are calculated relative to the mean of the vectors in the first frame.

    Parameters:
    frames_data (dict): Dictionary containing frame data with vectors for each frame.
    vector_type (str): The type of vector to use ('H' or 'G').

    Returns:
    np.ndarray: Array of local vectors (g_j or h_j) for the first frame.
    """
    vector_key = f'{vector_type}_vectors'


    G_first = np.array([vec.coords for vec in frames_data[1][vector_key]])
    G_zero = np.mean(G_first, axis=0)
    g_j = G_first - G_zero
    g_j = [Vector(*coords) for coords in g_j]

    return g_j




def perform_pivot_registration(G_points_frames, small_g_j, vector_type):
    """
    Perform point cloud registration for each frame
    
    
    Returns:
    dict: Dictionary with frame numbers as keys and (R, t) as values where R is the rotation matrix and t is the translation vector.
    """
    registration_results = {}
    vector_key = f'{vector_type}_vectors'

    for frame_num, frame_data in G_points_frames.items():
        source_points = np.array([vec.coords for vec in small_g_j])
        target_points = np.array([vec.coords for vec in frame_data[vector_key]])

        R_optimal, t_optimal = point_cloud_registration(target_points, source_points)
        registration_results[frame_num] = Frame(R_optimal, t_optimal)

    return registration_results


def combine_frames(F_D_opt_point_cloud, F_H_opt_point_cloud):
    """
    Combine two point cloud frame dictionaries by multiplying their transformations in order.

    Parameters:
    F_D_opt_point_cloud (dict): Dictionary containing the first set of frames.
    F_H_opt_point_cloud (dict): Dictionary containing the second set of frames.

    Returns:
    dict: Combined dictionary with frame numbers as keys and Frame objects as values.
    """
    combined_frames = {}

    for frame_num in F_D_opt_point_cloud:
        if frame_num in F_H_opt_point_cloud:
            frame_D = F_D_opt_point_cloud[frame_num]
            frame_H = F_H_opt_point_cloud[frame_num]

            combined_frame = frame_D @ frame_H
            combined_frames[frame_num] = combined_frame

    return combined_frames


