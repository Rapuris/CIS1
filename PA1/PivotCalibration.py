import numpy as np

def solve_for_pointer_and_dimple(point_cloud):
    """
    Solves the overdetermined system to find p_t and p_pivot using the equation:
    R_j * p_t + p_j = p_pivot

    Parameters:
    point_cloud (dict): Dictionary containing frame data with rotation and translation.
                                Each value in the dictionary should have 'rotation' and 'translation' attributes.

    Returns:
    tuple: (p_t, p_pivot) where both are 3x1 numpy arrays.
    """
    A = []
    b = []

    # Construct A and b based on each frame's rotation and translation
    for frame_num, frame in point_cloud.items():
        # Extract rotation matrix and translation vector
        R_j = frame.rotation
        p_j = frame.translation.reshape(3, 1)  # Ensure p_j is a 3x1 vector

        # Append the values to construct the matrix A and vector b
        # A_j = [R_j | -I], where I is the identity matrix
        A_j = np.hstack((R_j, -np.eye(3)))

        A.append(A_j)
        b.append(-p_j)


    # Stack A and b vertically to create the final system of equations
    A = np.vstack(A)
    b = np.vstack(b)


    # Solve the least squares problem to find p_t and p_pivot
    # A * x = b, where x = [p_t, p_pivot]
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Extract p_t and p_pivot from the solution vector
    p_t = x[:3].reshape(3, 1)
    p_pivot = x[3:].reshape(3, 1)

    return p_t, p_pivot