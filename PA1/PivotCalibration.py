import numpy as np

def pivot_calibration(F_G):
    """
    Pivot calibration to find the tip position (t_G) and the pivot point (p_pivot) when tip is in the dimple.

    Parameters:
    F_G : N_frames transformations from EM tracker to pointer frame.
    G_0 : N_frames vectors for origin of the pointer frame wrt the EM tracker frame.

    Returns:
    t_G : 3x1 vector of the tip of pointer wrt pointer frame.
    p_pivot: 3x1 vector of the tip of pointer wrt EM tracker frame.
    """
    N_frames = F_G.shape[0]

    if N_frames < 2:
        raise ValueError("Underdetemined system. Need at least 2 frames.")

    A_list = []
    b_list = []

    for k in range(N_frames):
        R_k = F_G[k][:3, :3]
        t_k = F_G[k][:3, 3]

        A_k = np.hstack((R_k, -np.eye(3)))
        b_k = -t_k.reshape(3, 1)

        A_list.append(A_k)
        b_list.append(b_k)

    A = np.vstack(A_list)
    b = np.vstack(b_list)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    t_G = x[:3].flatten()
    p_pivot = x[3:].flatten()

    return t_G, p_pivot


def solve_for_pointer_and_dimple(point_cloud):
    """
    Pivot calibration to find the tip position (t_H) and the pivot point (p_pivot) when tip is in the dimple.

    Parameters:
    point_cloud: rotation and translation data

    Returns:
    p_t, p_pivot 3x1 vectors
    """
    A = []
    b = []

    N_frames = point_cloud.shape[0]
    if N_frames < 2:
        raise ValueError("Underdetemined system. Need at least 2 frames.")

    for frame_num, frame in point_cloud.items():
        R_j = frame.rotation
        p_j = frame.translation.reshape(3, 1) 

        # A_j = [R_j | -I]
        A_j = np.hstack((R_j, -np.eye(3)))

        A.append(A_j)
        b.append(-p_j)

    A = np.vstack(A)
    b = np.vstack(b)

 
    #x = [p_t, p_pivot]
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    p_t = x[:3].reshape(3, 1)
    p_pivot = x[3:].reshape(3, 1)

    return p_t, p_pivot