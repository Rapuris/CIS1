import numpy as np
import matplotlib.pyplot as plt
import DataIO as io
import LinAlg as LA
import PivotCalibration as PC
import Debug
import glob
import regex as re

"""
This file serves as the complementary file to Run Dataset a.ipynb and allows the user to run and obtain the answers to all specified
questions in Assignment 1
"""


#PROBLEM 3

def problem_4(calbody_filepath, calreadings_filepath):
    d_vectors, a_vectors, c_vectors, N_D, N_A, N_C = io.read_calbody_file(calbody_filepath)
    frames_data = io.read_calreadings_file(calreadings_filepath)
    #PART A
    F_A_point_cloud = LA.perform_calibration_registration_for_frames(frames_data, a_vectors, vector_type='A')
    #PART B
    F_D_point_cloud = LA.perform_calibration_registration_for_frames(frames_data, d_vectors, vector_type='D')
    #PART C
    counter = 1
    C_expected_vectors = {}
    transformed_c_vectors = []
    for count in range(1, len(frames_data) + 1):
        for c_vec in c_vectors:
            transformed_c_vectors.append(F_D_point_cloud[count].inv() @ F_A_point_cloud[count] @ c_vec)
        C_expected_vectors[count] = transformed_c_vectors 
        transformed_c_vectors = []
    return C_expected_vectors

def problem_5(empivot_filepath):
    em_frames_data, N_G, N_frames  = io.read_empivot_file(empivot_filepath)
    G0_vectors = LA.compute_centroid_vectors(em_frames_data, vector_type='G')
    g_i_vectors = LA.compute_local_marker_vectors(em_frames_data, vector_type='G')
    F_G_frames = {}
    # Loop over the keys (frames) in em_frames_data and g_i_vectors
    for frame_num in em_frames_data:
        # Get the vectors for the current frame from em_frames_data and g_i_vectors
        em_frames_data_frame = {frame_num: em_frames_data[frame_num]}

        # Call the function with the modified arguments
        result = LA.perform_calibration_registration_for_frames(em_frames_data_frame, g_i_vectors, vector_type='G')

        # Store the result in the registration_results dictionary
        F_G_frames.update(result)
    F_G = np.array([np.array(frame) for frame in F_G_frames.values()])
    t_G, p_pivot = PC.pivot_calibration(F_G, G0_vectors)
    return p_pivot

def problem_6(optpivot_filepath, calbody_filepath):
    opt_frames_data, N_D, N_H, N_frames  = io.read_optpivot_file(optpivot_filepath)
    d_vectors, a_vectors, c_vectors, N_D, N_A, N_C = io.read_calbody_file(calbody_filepath)
    H0_vectors = LA.compute_centroid_vectors(opt_frames_data, vector_type='H')
    hi_vectors = LA.compute_local_marker_vectors(opt_frames_data, vector_type='H')
    F_D_opt_point_cloud = LA.perform_calibration_registration_for_frames(opt_frames_data, d_vectors, vector_type='D')
    F_H_opt_point_cloud = LA.perform_pivot_registration_for_frames(opt_frames_data, hi_vectors, vector_type='H')

    F_D_inv = {}
    for frame_num in F_D_opt_point_cloud:
        frame_D_inv = F_D_opt_point_cloud[frame_num].inv()
        F_D_inv[frame_num] = frame_D_inv

    F_DH_opt_point_cloud = LA.combine_point_cloud_frames(F_D_inv, F_H_opt_point_cloud)

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
    #print(C_expected_vectors)
    NC = len(C_expected_vectors[1])  # Number of C vectors (assumed to be the same for all frames)
    Nframes = len(C_expected_vectors)  # Total number of frames

    with open(outfile, 'w') as file:
        # Write NC, Nframes, and output file name on the same line
        file.write(f"{NC}, {Nframes}, {outfile}\n")
        file.write(f"{EM_probe_pos[0]}, {EM_probe_pos[1]}, {EM_probe_pos[2]}\n")

        file.write(f"{OPT_probe_pos[0]}, {OPT_probe_pos[1]}, {OPT_probe_pos[2]}\n")

        for frame_num, vectors in C_expected_vectors.items():
            for vector in vectors:
                # Output format for each coordinate set: x_i, y_i, z_i
                file.write(f"{vector.coords[0]}, {vector.coords[1]}, {vector.coords[2]}\n")

        #print(f"Data written to {outfile}")


if __name__ == '__main__':
    calbody_list = sorted(glob.glob("./PA_1_Data/*pa1*calbody.txt"))
    calreading_list = sorted(glob.glob("./PA_1_Data/*pa1*calreadings.txt"))
    empivot_list = sorted(glob.glob("./PA_1_Data/*pa1*empivot.txt"))
    optpivot_list = sorted(glob.glob("./PA_1_Data/*pa1*optpivot.txt"))
    
    for calbody, calreadings, empivot, optpivot in zip(calbody_list, calreading_list, empivot_list, optpivot_list):
        # Extract the letter from the file name
        name_pattern = r'pa1-(debug|unknown)-(.)-calbody.txt'
        res_calbody = re.search(name_pattern, calbody)
        _, letter = res_calbody.groups()
        print("Data set:", letter)

        # Problem 4: Compute C_expected
        C_expected = problem_4(calbody, calreadings)
        #print(C_expected)
        outfile = f"./PA_1_Data/pa1-{letter}-output1.txt"

        # Problem 5: Compute EM probe position
        EM_probe_pos = problem_5(empivot)

        # Problem 6: Compute optical probe position
        OPT_probe_pos = problem_6(optpivot, calbody)#[0]
        #print(OPT_probe_pos)

        # Write the results to the output file
        write_data(outfile, C_expected, EM_probe_pos, OPT_probe_pos)
    print("DONE AND FINISHED WITH ASSIGNMENT 1!")




