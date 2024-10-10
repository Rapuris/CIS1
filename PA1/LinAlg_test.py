import unittest
import numpy as np
import LinAlg as LA
import random
import Debug
import PivotCalibration as PC

class TestVector(unittest.TestCase):
    def setUp(self):
        # Setting up some test vectors
        self.vector_a = LA.Vector(1, 2, 3)
        self.vector_b = LA.Vector(4, 5, 6)

    def test_vector_initialization(self):
        self.assertTrue(np.allclose(self.vector_a.as_array(), [1, 2, 3]))

    def test_vector_addition(self):
        result = self.vector_a + self.vector_b
        expected = np.array([5, 7, 9])
        self.assertTrue(np.allclose(result.as_array(), expected))

    def test_vector_subtraction(self):
        result = self.vector_a - self.vector_b
        expected = np.array([-3, -3, -3])
        self.assertTrue(np.allclose(result.as_array(), expected))

    def test_vector_dot_product(self):
        result = self.vector_a.dot(self.vector_b)
        expected = 1*4 + 2*5 + 3*6
        self.assertEqual(result, expected)

    def test_vector_cross_product(self):
        result = self.vector_a.cross(self.vector_b)
        expected = np.cross(self.vector_a.as_array(), self.vector_b.as_array())
        self.assertTrue(np.allclose(result.as_array(), expected))

    def test_vector_magnitude(self):
        result = self.vector_a.magnitude()
        expected = np.linalg.norm(self.vector_a.as_array())
        self.assertEqual(result, expected)

class TestFrame(unittest.TestCase):
    def setUp(self):
        self.rotation_matrix = np.eye(3)
        self.translation_vector = np.array([1, 2, 3])
        self.frame = LA.Frame(self.rotation_matrix, self.translation_vector)
        self.vector = LA.Vector(1, 1, 1)

    def test_frame_initialization(self):
        self.assertTrue(np.allclose(self.frame.rotation, self.rotation_matrix))
        self.assertTrue(np.allclose(self.frame.translation, self.translation_vector))

    def test_frame_inverse(self):
        inverse_frame = self.frame.inv()
        expected_rotation = self.rotation_matrix.T
        expected_translation = -self.rotation_matrix.T @ self.translation_vector
        self.assertTrue(np.allclose(inverse_frame.rotation, expected_rotation))
        self.assertTrue(np.allclose(inverse_frame.translation, expected_translation))

    def test_homogeneous_transformation(self):
        homogeneous_matrix = np.array(self.frame)
        expected_matrix = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        self.assertTrue(np.allclose(homogeneous_matrix, expected_matrix))

    def test_frame_vector_multiplication(self):
        transformed_vector = self.frame @ self.vector
        expected_coords = self.rotation_matrix @ self.vector.as_array() + self.translation_vector
        self.assertTrue(np.allclose(transformed_vector.as_array(), expected_coords))

    def test_frame_frame_multiplication(self):
        another_translation = np.array([4, 5, 6])
        another_frame = LA.Frame(self.rotation_matrix, another_translation)
        combined_frame = self.frame @ another_frame
        expected_rotation = self.rotation_matrix @ self.rotation_matrix
        expected_translation = self.rotation_matrix @ another_translation + self.translation_vector
        self.assertTrue(np.allclose(combined_frame.rotation, expected_rotation))
        self.assertTrue(np.allclose(combined_frame.translation, expected_translation))

    def test_frame_numpy_array_multiplication(self):
        point = np.array([1, 1, 1])
        transformed_point = self.frame @ point
        expected_point = self.rotation_matrix @ point + self.translation_vector
        self.assertTrue(np.allclose(transformed_point, expected_point))

class TestTransformPoints(unittest.TestCase):
    def test_transform_points_random(self):
        random_points = Debug.generate_random_points(int(random.uniform(3,20)))
        theta, phi = random.uniform(0, np.pi), random.uniform(0, np.pi)
        translation = LA.Vector(random.uniform(0, 2000), random.uniform(0, 2000), random.uniform(0, 2000))
        transformation_matrix = Debug.create_transformation_matrix(theta, phi, translation)
        transformed_points = LA.transform_points(transformation_matrix, random_points)

        for original, transformed in zip(random_points, transformed_points):
            self.assertFalse(np.allclose(original.as_array(), transformed.as_array()))
    
    def test_transform_points_random_far(self):
        random_points = Debug.generate_random_points(int(random.uniform(2,20)))
        theta, phi = random.uniform(0, np.pi), random.uniform(0, np.pi)
        translation = LA.Vector(random.uniform(3000, 10000), random.uniform(3000, 10000), random.uniform(3000, 10000))
        transformation_matrix = Debug.create_transformation_matrix(theta, phi, translation)
        transformed_points = LA.transform_points(transformation_matrix, random_points)

        for original, transformed in zip(random_points, transformed_points):
            distance = np.linalg.norm(original.as_array() - transformed.as_array())
            self.assertGreaterEqual(distance, 5000)

class TestPointCloudRegistration(unittest.TestCase):
    def test_point_cloud_registration(self):
        random_points = Debug.generate_random_points(int(random.uniform(2,20)))
        theta, phi = random.uniform(0, np.pi), random.uniform(0, np.pi)
        translation = LA.Vector(random.uniform(0, 2000), random.uniform(0, 2000), random.uniform(0, 2000))
        transformation_matrix = Debug.create_transformation_matrix(theta, phi, translation)
        transformed_points = LA.transform_points(transformation_matrix, random_points)

        transformed_points_np = Debug.vectors_to_numpy(transformed_points)
        random_points_np = Debug.vectors_to_numpy(random_points)

        R, t = LA.point_cloud_registration(transformed_points_np, random_points_np)
        ground_truth_R, ground_truth_t = transformation_matrix.rotation, transformation_matrix.translation

        self.assertTrue(np.allclose(R, ground_truth_R, atol=1e-2))
        self.assertTrue(np.allclose(t, ground_truth_t, atol=1e-2))
    
    def test_point_cloud_registration_far(self):
        random_points = Debug.generate_random_points(int(random.uniform(2,20)))
        theta, phi = random.uniform(0, np.pi), random.uniform(0, np.pi)
        translation = LA.Vector(random.uniform(3000, 10000), random.uniform(3000, 10000), random.uniform(3000, 10000))
        transformation_matrix = Debug.create_transformation_matrix(theta, phi, translation)
        transformed_points = LA.transform_points(transformation_matrix, random_points)

        transformed_points_np = Debug.vectors_to_numpy(transformed_points)
        random_points_np = Debug.vectors_to_numpy(random_points)

        R, t = LA.point_cloud_registration(transformed_points_np, random_points_np)
        ground_truth_R, ground_truth_t = transformation_matrix.rotation, transformation_matrix.translation

        self.assertTrue(np.allclose(R, ground_truth_R, atol=1e-2))
        self.assertTrue(np.allclose(t, ground_truth_t, atol=1e-2))

    def test_point_cloud_registration_noisy(self):
        random_points = Debug.generate_random_points(int(random.uniform(4,20)), noise = 2)
        theta, phi = random.uniform(0, np.pi), random.uniform(0, np.pi)
        translation = LA.Vector(random.uniform(3000, 10000), random.uniform(3000, 10000), random.uniform(3000, 10000))
        transformation_matrix = Debug.create_transformation_matrix(theta, phi, translation)
        transformed_points = LA.transform_points(transformation_matrix, random_points)

        transformed_points_np = Debug.vectors_to_numpy(transformed_points)
        random_points_np = Debug.vectors_to_numpy(random_points)

        R, t = LA.point_cloud_registration(transformed_points_np, random_points_np)
        ground_truth_R, ground_truth_t = transformation_matrix.rotation, transformation_matrix.translation

        self.assertTrue(np.allclose(R, ground_truth_R, atol=1))
        self.assertTrue(np.allclose(t, ground_truth_t, atol=1))

    def test_point_cloud_registration_vs_real(self):
        random_points = Debug.generate_random_points(int(random.uniform(2,20)))
        theta, phi = random.uniform(0, np.pi), random.uniform(0, np.pi)
        translation = LA.Vector(random.uniform(0, 2000), random.uniform(0, 2000), random.uniform(0, 2000))
        transformation_matrix = Debug.create_transformation_matrix(theta, phi, translation)
        transformed_points = LA.transform_points(transformation_matrix, random_points)

        transformed_points_np = Debug.vectors_to_numpy(transformed_points)
        random_points_np = Debug.vectors_to_numpy(random_points)

        R, t = LA.point_cloud_registration(transformed_points_np, random_points_np)
        print("Rotation matrix:", R)
        print("Translation vector:", t, "\n")
        print("Ground truth: \n", transformation_matrix)

        ground_truth_R = transformation_matrix.rotation
        ground_truth_t = transformation_matrix.translation

        self.assertTrue(np.allclose(R, ground_truth_R, atol=1e-1))
        self.assertTrue(np.allclose(t, ground_truth_t, atol=1e-1))

        proposed_transformation_matrix = LA.Frame(R, t)
        proposed_transformed_points = LA.transform_points(proposed_transformation_matrix, random_points)

        for proposed, truth in zip(proposed_transformed_points, transformed_points):
            self.assertTrue(np.allclose(proposed.as_array(), truth.as_array(), atol=1e-1))


class TestCExpectedCalculation(unittest.TestCase):
    def setUp(self):
        self.c_vectors = [
            LA.Vector(1.0, 2.0, 3.0),
            LA.Vector(4.0, 5.0, 6.0),
            LA.Vector(7.0, 8.0, 9.0)
        ]
        
        self.F_D_dict = {
            1: LA.Frame(np.eye(3), np.array([0.0, 0.0, 0.0])),
            2: LA.Frame(np.eye(3), np.array([1.0, 1.0, 1.0]))
        }
        
        self.F_A_dict = {
            1: LA.Frame(np.eye(3), np.array([1.0, 1.0, 1.0])),
            2: LA.Frame(np.eye(3), np.array([2.0, 2.0, 2.0]))
        }

    def test_c_expected_calculation(self):
        C_expected_results = LA.compute_C_expected(self.F_D_dict, self.F_A_dict, self.c_vectors)
        expected_C_frame_1 = [
            LA.Vector(2.0, 3.0, 4.0),
            LA.Vector(5.0, 6.0, 7.0),
            LA.Vector(8.0, 9.0, 10.0)
        ]

        expected_C_frame_2 = [
            LA.Vector(2.0, 3.0, 4.0),
            LA.Vector(5.0, 6.0, 7.0),
            LA.Vector(8.0, 9.0, 10.0)
        ]


        for frame_num, expected_vectors in zip([1, 2], [expected_C_frame_1, expected_C_frame_2]):
            computed_vectors = C_expected_results[frame_num]
            for i, (computed, expected) in enumerate(zip(computed_vectors, expected_vectors)):
                self.assertTrue(np.allclose(computed.as_array(), expected.as_array(), atol=1e-1), 
                                f"Mismatch in frame {frame_num}, vector {i}: Computed {computed.as_array()}, Expected {expected.as_array()}")

class TestCalibrationRegistrationForFrames(unittest.TestCase):
    def setUp(self):
        self.calbody_vectors = [
            LA.Vector(1.0, 2.0, 3.0),
            LA.Vector(4.0, 5.0, 6.0),
            LA.Vector(7.0, 8.0, 9.0)
        ]

        self.frames_data = {
            1: {
                'A_vectors': [
                    LA.Vector(1.1, 2.1, 3.1),
                    LA.Vector(4.1, 5.1, 6.1),
                    LA.Vector(7.1, 8.1, 9.1)
                ],
                'D_vectors': [
                    LA.Vector(0.9, 1.9, 2.9),
                    LA.Vector(3.9, 4.9, 5.9),
                    LA.Vector(6.9, 7.9, 8.9)
                ]
            }
        }

    def test_calibration_registration_for_frames(self):
        registration_results = LA.perform_calibration_registration(self.frames_data, self.calbody_vectors, vector_type='A')
        self.assertEqual(len(registration_results), len(self.frames_data), "Number of frames in registration results does not match input.")

        for frame_num, frame in registration_results.items():
            self.assertIsInstance(frame, LA.Frame, f"Registration result for frame {frame_num} is not a Frame object.")
            
            
            expected_rotation = np.eye(3)
            expected_translation = np.array([0.100, 0.100, 0.100])
            self.assertTrue(np.allclose(frame.rotation, expected_rotation, atol=1e-1), 
                            f"Rotation matrix for frame {frame_num} does not match expected identity matrix.")
            self.assertTrue(np.allclose(frame.translation, expected_translation, atol=1e-1),
                            f"Translation vector for frame {frame_num} is not close to zero.")
             
class TestPivotCalibration(unittest.TestCase):
    def setUp(self):
        self.F_G = np.array([
            [[1, 0, 0, 1],   # Frame 1: Rotation is identity, translation is [1, 0, 0]
             [0, 1, 0, 2],
             [0, 0, 1, 3],
             [0, 0, 0, 1]],
            
            [[1, 0, 0, 2],   # Frame 2: Rotation is identity, translation is [2, 1, 0]
             [0, 1, 0, 1],
             [0, 0, 1, 4],
             [0, 0, 0, 1]]
        ])

    def test_pivot_calibration(self):
        t_G, p_pivot = PC.pivot_calibration(self.F_G)

        expected_t_G = np.array([-0.75, -0.75, -1.75]) 
        expected_p_pivot = np.array([0.75, 0.75, 1.75])

        # Verify the results
        print(t_G, p_pivot)
        self.assertTrue(np.allclose(t_G, expected_t_G, atol=1e-1), 
                        f"Tip position (t_G) is not as expected: {t_G}")
        self.assertTrue(np.allclose(p_pivot, expected_p_pivot, atol=1e-1), 
                        f"Pivot position (p_pivot) is not as expected: {p_pivot}")


class TestCentroidAndLocalMarkerVectors(unittest.TestCase):
    def setUp(self):
        self.frames_data = {
            1: {'H_vectors': [LA.Vector(1, 2, 3), LA.Vector(4, 5, 6)], 'G_vectors': [LA.Vector(7, 8, 9), LA.Vector(10, 11, 12)]},
            2: {'H_vectors': [LA.Vector(2, 3, 4), LA.Vector(5, 6, 7)], 'G_vectors': [LA.Vector(8, 9, 10), LA.Vector(11, 12, 13)]}
        }

    def test_compute_centroid_vectors(self):
        """Test computing centroid vectors."""
        centroid_vectors = LA.compute_centroid_vectors(self.frames_data, 'H')
        expected_centroid_1 = LA.Vector(2.5, 3.5, 4.5)
        expected_centroid_2 = LA.Vector(3.5, 4.5, 5.5)
        self.assertTrue(np.allclose(centroid_vectors[0].as_array(), expected_centroid_1.as_array()))
        self.assertTrue(np.allclose(centroid_vectors[1].as_array(), expected_centroid_2.as_array()))

    def test_compute_local_marker_vectors(self):
        """Test computing local marker vectors."""
        local_vectors = LA.compute_local_marker_vectors(self.frames_data, 'G')
        expected_local_vectors = [LA.Vector(-1.5, -1.5, -1.5), LA.Vector(1.5, 1.5, 1.5)]
        self.assertTrue(np.allclose(local_vectors[0].as_array(), expected_local_vectors[0].as_array()))
        self.assertTrue(np.allclose(local_vectors[1].as_array(), expected_local_vectors[1].as_array()))


    def test_single_frame_data(self):
        """Test compute_centroid_vectors and compute_local_marker_vectors with a single frame."""
        single_frame_data = {
            1: {'H_vectors': [LA.Vector(1, 2, 3), LA.Vector(4, 5, 6)], 'G_vectors': [LA.Vector(7, 8, 9), LA.Vector(10, 11, 12)]}
        }
        centroid_vectors = LA.compute_centroid_vectors(single_frame_data, 'H')
        expected_centroid = LA.Vector(2.5, 3.5, 4.5)
        self.assertTrue(np.allclose(centroid_vectors[0].as_array(), expected_centroid.as_array()))

        local_vectors = LA.compute_local_marker_vectors(single_frame_data, 'G')
        expected_local_vectors = [LA.Vector(-1.5, -1.5, -1.5), LA.Vector(1.5, 1.5, 1.5)]
        self.assertTrue(np.allclose(local_vectors[0].as_array(), expected_local_vectors[0].as_array()))
        self.assertTrue(np.allclose(local_vectors[1].as_array(), expected_local_vectors[1].as_array()))

    def test_multiple_vector_types(self):
        """Test compute_centroid_vectors with different vector types ('H' and 'G')."""
        centroid_vectors_H = LA.compute_centroid_vectors(self.frames_data, 'H')
        expected_centroid_H_1 = LA.Vector(2.5, 3.5, 4.5)
        self.assertTrue(np.allclose(centroid_vectors_H[0].as_array(), expected_centroid_H_1.as_array()))

        centroid_vectors_G = LA.compute_centroid_vectors(self.frames_data, 'G')
        expected_centroid_G_1 = LA.Vector(8.5, 9.5, 10.5)
        self.assertTrue(np.allclose(centroid_vectors_G[0].as_array(), expected_centroid_G_1.as_array()))

if __name__ == '__main__':
    unittest.main()