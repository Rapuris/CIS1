import LinAlg as LA


def read_calbody_file(file_path: str):
    """
    Reads a CALBODY.TXT file and extracts the d_i, a_i, and c_i vectors into separate lists.

    d_i = EM tracker base -> EM tracker base markers
    a_i = calibration object base -> calibration object optical markers
    c_i = EM calibration object base -> calibration object EM markers
    """
    d_vectors = []  
    a_vectors = []  
    c_vectors = []  

    with open(file_path, 'r') as file:
        lines = file.readlines()

        header = lines[0].split(',')
        N_D = int(header[0].strip()) 
        N_A = int(header[1].strip())  
        N_C = int(header[2].strip())  


        for i in range(1, N_D + 1):
            d_coords = list(map(float, lines[i].split(',')))
            d_vectors.append(LA.Vector(d_coords[0], d_coords[1], d_coords[2]))

        for i in range(N_D + 1, N_D + N_A + 1):
            a_coords = list(map(float, lines[i].split(',')))
            a_vectors.append(LA.Vector(a_coords[0], a_coords[1], a_coords[2]))

        for i in range(N_D + N_A + 1, N_D + N_A + N_C + 1):
            c_coords = list(map(float, lines[i].split(',')))
            c_vectors.append(LA.Vector(c_coords[0], c_coords[1], c_coords[2]))

    return d_vectors, a_vectors, c_vectors, N_D, N_A, N_C


def read_calreadings_file(file_path: str):
    """
    Reads a CALREADINGS.TXT file and organizes data by frame number. Each frame contains three lists of Vector objects
    for D_i, A_i, and C_i vectors.

    D_i = Opitcal tracker base -> EM tracker base markers
    A_i = Optical tracker base -> calibration object optical markers
    C_i = EM tracker base -> calibration object EM markers
    """
    frames_data = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

        header = lines[0].split(',')
        N_D = int(header[0].strip()) 
        N_A = int(header[1].strip()) 
        N_C = int(header[2].strip())  
        N_frames = int(header[3].strip())  
        filename = header[4].strip()  

        line_index = 1 

        for frame_num in range(1, N_frames + 1):
            frame_dict = {
                'D_vectors': [],
                'A_vectors': [],
                'C_vectors': []
            }

            for _ in range(N_D):
                d_coords = list(map(float, lines[line_index].split(',')))
                frame_dict['D_vectors'].append(LA.Vector(d_coords[0], d_coords[1], d_coords[2]))
                line_index += 1
            for _ in range(N_A):
                a_coords = list(map(float, lines[line_index].split(',')))
                frame_dict['A_vectors'].append(LA.Vector(a_coords[0], a_coords[1], a_coords[2]))
                line_index += 1
            for _ in range(N_C):
                c_coords = list(map(float, lines[line_index].split(',')))
                frame_dict['C_vectors'].append(LA.Vector(c_coords[0], c_coords[1], c_coords[2]))
                line_index += 1
            frames_data[frame_num] = frame_dict

    return frames_data


def read_empivot_file(file_path: str):
    """
    Reads EMPIVOT.TXT file and extracts the G_i points for each frame.

    G_i = EM tracker base -> EM tracker markers
    """
    frames = {}  

    with open(file_path, 'r') as file:
        lines = file.readlines()

        header = lines[0].split(',')
        N_G = int(header[0].strip())  
        N_frames = int(header[1].strip()) 
        file_name = header[2].strip() 

        current_line = 1
        for frame_idx in range(N_frames):
            frame_points = []
            for i in range(N_G):
                G_coords = list(map(float, lines[current_line].split(',')))
                frame_points.append(LA.Vector(*G_coords))
                current_line += 1
            frames[frame_idx + 1] = {'G_vectors': frame_points}

    return frames, N_G, N_frames


def read_optpivot_file(file_path: str):
    """
    Reads OPTPIVOT.TXT file and extracts the D_i and H_i points for each frame.

    H_i = optical tracker base -> optical pointer markers
    D_i = Optical tracker base -> EM tracker base markers
    """
    frames = {}  

    with open(file_path, 'r') as file:
        lines = file.readlines()

        header = lines[0].split(',')
        N_D = int(header[0].strip())  
        N_H = int(header[1].strip())  
        N_frames = int(header[2].strip())  
        file_name = header[3].strip()  

        current_line = 1
        for frame_idx in range(N_frames):
            D_points = []
            H_points = []

            for i in range(N_D):
                D_coords = list(map(float, lines[current_line].split(',')))
                D_points.append(LA.Vector(*D_coords))
                current_line += 1

            for i in range(N_H):
                H_coords = list(map(float, lines[current_line].split(',')))
                H_points.append(LA.Vector(*H_coords))
                current_line += 1

            frames[frame_idx + 1] = {'D_vectors': D_points, 'H_vectors': H_points}

    return frames, N_D, N_H, N_frames