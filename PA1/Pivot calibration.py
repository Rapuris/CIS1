import numpy as np

def pivot_calibration(F_G, G_0):
    """
    Perform pivot calibration to find the tip position (t_G) and the pivot point (p_pivot).

    Parameters:
    F_G (numpy.ndarray): An array of shape (N_frames, 4, 4) containing the transformation matrices from EM tracker to pointer frame.
    G_0 (numpy.ndarray): An array of shape (N_frames, 3) containing the origin of the pointer frame with respect to the EM tracker frame.

    Returns:
    t_G (numpy.ndarray): A 3-element array representing the tip of the pointer in its own frame.
    p_pivot (numpy.ndarray): A 3-element array representing the position of the pivot point in the EM tracker frame.
    """
    # Number of frames
    N_frames = F_G.shape[0]

    # Initialize lists to store A and b matrices
    A_list = []
    b_list = []

    for k in range(N_frames):
        # Extract rotation and translation from F_G[k]
        R_k = F_G[k][:3, :3]
        t_k = F_G[k][:3, 3]

        # Construct A_k and b_k
        A_k = np.hstack((R_k, -np.eye(3)))
        b_k = -t_k.reshape(3, 1)

        # Append to the list
        A_list.append(A_k)
        b_list.append(b_k)

    # Stack all A_k and b_k
    A = np.vstack(A_list)
    b = np.vstack(b_list)

    # Solve the linear system using least squares
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Extract t_G and p_pivot
    t_G = x[:3].flatten()
    p_pivot = x[3:].flatten()

    return t_G, p_pivot

# Example usage (assuming F_G and G_0 are numpy arrays of correct dimensions)
F_G = np.array([...])  # Shape: (N_frames, 4, 4)
G_0 = np.array([...])  # Shape: (N_frames, 3)
t_G, p_pivot = pivot_calibration(F_G, G_0)
print("t_G:", t_G)
print("p_pivot:", p_pivot)