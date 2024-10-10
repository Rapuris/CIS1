import numpy as np
import matplotlib.pyplot as plt
import DataIO as io
import LinAlg as LA
import PivotCalibration as PC
import glob
import regex as re

'''
Created on Oct 5, 2024

@author: Sampath Rapuri and William Li

@summary: This module is to answer and generate the appropriate output files for programming assignment 1.
'''

def problem_4(calbody_filepath, calreadings_filepath):
    """
    Computes the expected C vectors for each frame using calibration data.

    Parameters:
    calbody_filepath (str): Path to the calbody file containing d, a, and c vectors.
    calreadings_filepath (str): Path to the calreadings file containing frame data.

    Returns:
    dict: A dictionary where keys are frame numbers and values are lists of C_expected vectors.
    """
    d_vectors, a_vectors, c_vectors, N_D, N_A, N_C = io.read_calbody_file(calbody_filepath)
    frames_data = io.read_calreadings_file(calreadings_filepath)

    F_A_point_cloud = LA.perform_calibration_registration(frames_data, a_vectors, vector_type='A')
    F_D_point_cloud = LA.perform_calibration_registration(frames_data, d_vectors, vector_type='D')

    C_expected_vectors = {}
    transformed_c_vectors = []

    for count in range(1, len(frames_data) + 1):
        for c_vec in c_vectors:
            transformed_c_vectors.append(F_D_point_cloud[count].inv() @ F_A_point_cloud[count] @ c_vec)
        C_expected_vectors[count] = transformed_c_vectors 
        transformed_c_vectors = []

    return C_expected_vectors

def problem_5(empivot_filepath):
    """
    Performs pivot calibration for EM tracker data to find the pivot position.

    Parameters:
    empivot_filepath (str): Path to the empivot file containing EM tracker data.

    Returns:
    np.ndarray: 3x1 vector representing the pivot position with respect to the EM tracker frame.
    """
    em_frames_data, N_G, N_frames = io.read_empivot_file(empivot_filepath)
    G0_vectors = LA.compute_centroid_vectors(em_frames_data, vector_type='G')
    g_i_vectors = LA.compute_local_marker_vectors(em_frames_data, vector_type='G')
    F_G_frames = {}

    for frame_num in em_frames_data:
        em_frames_data_frame = {frame_num: em_frames_data[frame_num]}
        result = LA.perform_calibration_registration(em_frames_data_frame, g_i_vectors, vector_type='G')
        F_G_frames.update(result)

    F_G = np.array([np.array(frame) for frame in F_G_frames.values()])
    t_G, p_pivot = PC.pivot_calibration(F_G)
    return p_pivot

def problem_6(optpivot_filepath, calbody_filepath):
    """
    Solves for the dimple position using optical tracker data.

    Parameters:
    optpivot_filepath (str): Path to the optpivot file containing optical tracker data.
    calbody_filepath (str): Path to the calbody file containing calibration data.

    Returns:
    np.ndarray: 3x1 vector representing the dimple position with respect to the optical tracker frame.
    """
    opt_frames_data, N_D, N_H, N_frames = io.read_optpivot_file(optpivot_filepath)
    d_vectors, a_vectors, c_vectors, N_D, N_A, N_C = io.read_calbody_file(calbody_filepath)
    H0_vectors = LA.compute_centroid_vectors(opt_frames_data, vector_type='H')
    hi_vectors = LA.compute_local_marker_vectors(opt_frames_data, vector_type='H')

    F_D_opt_point_cloud = LA.perform_calibration_registration(opt_frames_data, d_vectors, vector_type='D')
    F_H_opt_point_cloud = LA.perform_pivot_registration(opt_frames_data, hi_vectors, vector_type='H')

    F_D_inv = {}
    for frame_num in F_D_opt_point_cloud:
        frame_D_inv = F_D_opt_point_cloud[frame_num].inv()
        F_D_inv[frame_num] = frame_D_inv

    F_DH_opt_point_cloud = LA.combine_frames(F_D_inv, F_H_opt_point_cloud)

    t_H, p_dimple_H = PC.solve_for_pointer_and_dimple(F_DH_opt_point_cloud)
    p_dimple_H = np.array(p_dimple_H).flatten()
    return p_dimple_H

def write_data(outfile, C_expected_vectors, EM_probe_pos, OPT_probe_pos):
    """
    Write the results to the specified output file in the required format.

    Parameters:
    outfile (str): The output file name.
    C_expected_vectors (dict): Dictionary with keys as frame numbers and values as a list of expected C vectors.
    EM_probe_pos (np.ndarray): The computed EM probe positions.
    OPT_probe_pos (np.ndarray): The computed optical probe positions.
    """
    NC = len(C_expected_vectors[1])  
    Nframes = len(C_expected_vectors)  
    outfile_name = outfile.split('/')[-1]
    with open(outfile, 'w') as file:
        file.write(f"{NC}, {Nframes}, {outfile_name}\n")
        file.write(f"{EM_probe_pos[0]:.2f}, {EM_probe_pos[1]:.2f}, {EM_probe_pos[2]:.2f}\n")
        file.write(f"{OPT_probe_pos[0]:.2f}, {OPT_probe_pos[1]:.2f}, {OPT_probe_pos[2]:.2f}\n")
        for frame_num, vectors in C_expected_vectors.items():
            for vector in vectors:
                file.write(f"{vector.coords[0]:.2f}, {vector.coords[1]:.2f}, {vector.coords[2]:.2f}\n")



if __name__ == '__main__':
    calbody_list = sorted(glob.glob("./PA_1_Data/*pa1*calbody.txt"))
    calreading_list = sorted(glob.glob("./PA_1_Data/*pa1*calreadings.txt"))
    empivot_list = sorted(glob.glob("./PA_1_Data/*pa1*empivot.txt"))
    optpivot_list = sorted(glob.glob("./PA_1_Data/*pa1*optpivot.txt"))
    
    for calbody, calreadings, empivot, optpivot in zip(calbody_list, calreading_list, empivot_list, optpivot_list):
        name_pattern = r'pa1-(debug|unknown)-([a-z])-calbody.txt'
        match = re.search(name_pattern, calbody)
        prefix, letter = match.groups()
        print("Data set:", letter)

        # Problem 4: Compute C_expected
        C_expected = problem_4(calbody, calreadings)
        outfile = f"./PA1_Outputs/pa1-{prefix}-{letter}-output1.txt"

        # Problem 5: Compute EM probe position
        EM_probe_pos = problem_5(empivot)

        # Problem 6: Compute optical probe position
        OPT_probe_pos = problem_6(optpivot, calbody)#[0]

        write_data(outfile, C_expected, EM_probe_pos, OPT_probe_pos)
    print("DONE AND FINISHED WITH ASSIGNMENT 1!")




