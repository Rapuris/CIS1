import LinAlg as LA


def read_calbody_file(file_path: str):
    """
    Reads a CALBODY.TXT file and extracts the d_i, a_i, and c_i vectors into separate lists.

    Args:
    file_path (str): Path to the CALBODY.TXT file.

    Returns:
    tuple: Three lists containing d_i, a_i, and c_i vectors respectively.
    """
    d_vectors = []  # List to store the d_i vectors
    a_vectors = []  # List to store the a_i vectors
    c_vectors = []  # List to store the c_i vectors

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # First line contains N_D, N_A, N_C, and filename
        header = lines[0].split(',')
        N_D = int(header[0].strip())  # Number of d_i markers (base)
        N_A = int(header[1].strip())  # Number of a_i markers (calibration object)
        N_C = int(header[2].strip())  # Number of c_i markers (EM markers)

        # Extract d_i vectors from lines 2 to N_D + 1
        for i in range(1, N_D + 1):
            d_coords = list(map(float, lines[i].split(',')))
            d_vectors.append(LA.Vector(d_coords[0], d_coords[1], d_coords[2]))

        # Extract a_i vectors from lines N_D + 1 to N_D + N_A + 1
        for i in range(N_D + 1, N_D + N_A + 1):
            a_coords = list(map(float, lines[i].split(',')))
            a_vectors.append(LA.Vector(a_coords[0], a_coords[1], a_coords[2]))

        # Extract c_i vectors from lines N_D + N_A + 1 to N_D + N_A + N_C + 1
        for i in range(N_D + N_A + 1, N_D + N_A + N_C + 1):
            c_coords = list(map(float, lines[i].split(',')))
            c_vectors.append(LA.Vector(c_coords[0], c_coords[1], c_coords[2]))

    return d_vectors, a_vectors, c_vectors, N_D, N_A, N_C


def read_calreadings_file(file_path: str):
    """
    Reads a CALREADINGS.TXT file and organizes data by frame number. Each frame contains three lists of Vector objects
    for d_i, a_i, and c_i vectors.

    Args:
    file_path (str): Path to the CALREADINGS.TXT file.

    Returns:
    dict: A dictionary where the key is the frame number, and the value is a dictionary containing:
        - d_vectors: List of Vector objects for d_i coordinates
        - a_vectors: List of Vector objects for a_i coordinates
        - c_vectors: List of Vector objects for c_i coordinates
    """
    frames_data = {}  # Dictionary to store data for each frame

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # First line contains N_D, N_A, N_C, N_frames, and filename
        header = lines[0].split(',')
        N_D = int(header[0].strip())  # Number of d_i markers
        N_A = int(header[1].strip())  # Number of a_i markers
        N_C = int(header[2].strip())  # Number of c_i markers
        N_frames = int(header[3].strip())  # Number of data frames
        filename = header[4].strip()  # The filename is stored here as a string

        # Initialize the reading process
        line_index = 1  # Start reading after the header

        for frame_num in range(1, N_frames + 1):
            frame_dict = {
                'd_vectors': [],
                'a_vectors': [],
                'c_vectors': []
            }


            # Extract d_i vectors (N_D lines)
            for _ in range(N_D):
                d_coords = list(map(float, lines[line_index].split(',')))
                frame_dict['d_vectors'].append(LA.Vector(d_coords[0], d_coords[1], d_coords[2]))
                line_index += 1

            # Extract a_i vectors (N_A lines)
            for _ in range(N_A):
                a_coords = list(map(float, lines[line_index].split(',')))
                frame_dict['a_vectors'].append(LA.Vector(a_coords[0], a_coords[1], a_coords[2]))
                line_index += 1

            # Extract c_i vectors (N_C lines)
            for _ in range(N_C):
                c_coords = list(map(float, lines[line_index].split(',')))
                frame_dict['c_vectors'].append(LA.Vector(c_coords[0], c_coords[1], c_coords[2]))
                line_index += 1

            # Store the frame data in the dictionary
            frames_data[frame_num] = frame_dict

    return frames_data