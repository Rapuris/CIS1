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
    source_points (np.ndarray): Nx3 array of points from calbody. (e.g., a_vectors, d_vectors, c_vectors).
    
    Returns:
    R (np.ndarray): 3x3 optimal rotation matrix.
    t (np.ndarray): 3x1 translation vector.
    """
    # Initial guess for parameters: rotation as [0, 0, 0] (no rotation), translation as [0, 0, 0]
    initial_params = np.zeros(6)
    initial_t = compute_midpoint(target_points) - compute_midpoint(source_points)
    initial_params[3:] = initial_t

    # Use least squares to minimize the residuals
    result = least_squares(residuals, initial_params, args=(source_points, target_points))

    # Extract the optimal parameters from the result
    rvec_optimal = result.x[:3]
    t_optimal = result.x[3:]

    # Convert the optimal Rodrigues vector to a rotation matrix
    R_optimal = rodrigues_to_rotation_matrix(rvec_optimal)

    return R_optimal, t_optimal

def perform_calibration_registration_for_frames(calreadings_frames, calbody_vectors, vector_type):
    """
    Perform point cloud registration for each frame in calreadings, selecting the type of vectors (A, D, or C).
    
    Args:
    calreadings_frames (dict): Dictionary where each key is a frame number and value is a dict of A_vectors, D_vectors, or C_vectors.
    calbody_vectors (list): List of vectors from calbody (corresponding to the selected vector type).
    vector_type (str): The type of vector to use ('A', 'D', or 'C',).
    
    Returns:
    dict: Dictionary with frame numbers as keys and (R, t) as values where R is the rotation matrix and t is the translation vector.
    """
    registration_results = {}

    # Convert calbody vectors (for the selected type) to a numpy array
    source_points = np.array([vec.coords for vec in calbody_vectors])

    # Select the appropriate vectors from calreadings_frames based on vector_type
    vector_key = f'{vector_type}_vectors'  # e.g., 'A_vectors', 'D_vectors', 'C_vectors'
    
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

def compute_C_expected_for_all_frames(F_D_dict, F_A_dict, c_vectors):
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

    # Loop over each common frame number in both dictionaries
    for frame_num in F_D_dict.keys():
        F_D = F_D_dict[frame_num]
        F_A = F_A_dict[frame_num]

        # Debugging: Check what F_D and F_A actually are
        #print(f"Frame {frame_num}:")
        #print(f"F_D: {F_D}")
        #print(f"F_A: {F_A}")

        # Check if F_D and F_A are Frame objects
        if not isinstance(F_D, Frame) or not isinstance(F_A, Frame):
            raise TypeError(f"Expected F_D and F_A to be Frame objects, but got {type(F_D)} and {type(F_A)}")

        # Step 1: Compute the inverse of F_D for this frame
        F_D_inv = F_D.inv()

        # Step 2: Initialize an empty list to store the C_expected results for this frame
        C_expected_list = []

        # Step 3: Loop over each vector c_i in c_vectors
        for c_i in c_vectors:
            # Step 4: Compute F_A • c_i (this applies F_A's transformation to c_i)
            F_A_c_i = F_A @ c_i

            # Step 5: Compute F_D^−1 • (F_A • c_i) (this applies the inverse transformation of F_D)
            C_expected = F_D_inv @ F_A_c_i

            # Append the result to the list for this frame
            C_expected_list.append(C_expected)

        # Store the results for this frame in the dictionary
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

    # Get the vectors for the first frame (assuming frames_data uses keys like 'frame1', 'frame2', etc.)
    G_first = np.array([vec.coords for vec in frames_data[1][vector_key]])

    # Compute the centroid (mean) of the vectors in the first frame
    G_zero = np.mean(G_first, axis=0)

    # Calculate the local marker vectors by subtracting the mean
    g_j = G_first - G_zero
    g_j = [Vector(*coords) for coords in g_j]

    return g_j




def translate_points(observations, midpoint):
    """
    Translate the observations relative to the midpoint G0.
    observations: List of 3D points (Nx3 array).
    midpoint: The calculated midpoint G0 (1x3 array).
    Returns the translated points g_j = G_j - G0.
    """
    return observations - midpoint


def perform_pivot_registration_for_frames(G_points_frames, small_g_j, vector_type):
    """
    Perform point cloud registration for each frame
    
    
    Returns:
    dict: Dictionary with frame numbers as keys and (R, t) as values where R is the rotation matrix and t is the translation vector.
    """
    registration_results = {}
    vector_key = f'{vector_type}_vectors'
# Perform registration for each frame
    for frame_num, frame_data in G_points_frames.items():
       
        source_points = np.array([vec.coords for vec in small_g_j])
        target_points = np.array([vec.coords for vec in frame_data[vector_key]])

        # Perform point cloud registration using least squares to find R and t
        R_optimal, t_optimal = point_cloud_registration_least_squares(target_points, source_points)

        # Store the result for this frame
        registration_results[frame_num] = Frame(R_optimal, t_optimal)

    return registration_results








def visualize_vectors(d_vectors, a_vectors, c_vectors):
    """
    Visualize the vectors using matplotlib.

    Parameters:
    d_vectors (list of Vector): List of d_i vectors.
    a_vectors (list of Vector): List of a_i vectors.
    c_vectors (list of Vector): List of c_i vectors.
    """
    # Extract coordinates from Vector objects
    d_coords = np.array([vec.coords for vec in d_vectors])
    a_coords = np.array([vec.coords for vec in a_vectors])
    c_coords = np.array([vec.coords for vec in c_vectors])

    # Plot d_vectors
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


def combine_point_cloud_frames(F_D_opt_point_cloud, F_H_opt_point_cloud):
    """
    Combine two point cloud frame dictionaries by multiplying their transformations in order.

    Parameters:
    F_D_opt_point_cloud (dict): Dictionary containing the first set of frames.
    F_H_opt_point_cloud (dict): Dictionary containing the second set of frames.

    Returns:
    dict: Combined dictionary with frame numbers as keys and Frame objects as values.
    """
    combined_frames = {}

    # Iterate over frame numbers present in both dictionaries
    for frame_num in F_D_opt_point_cloud:
        if frame_num in F_H_opt_point_cloud:
            # Get the Frame objects from both dictionaries
            frame_D = F_D_opt_point_cloud[frame_num]
            frame_H = F_H_opt_point_cloud[frame_num]

#Multiply the frames to combine transformations
            combined_frame = frame_D @ frame_H

#Store the result in the combined dictionary
            combined_frames[frame_num] = combined_frame

    return combined_frames


