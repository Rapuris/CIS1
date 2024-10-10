import unittest
import numpy as np
import LinAlg as LA
import random
import Debug

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
        random_points = Debug.generate_random_points(int(random.uniform(4,20)), noise = 5)
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

        self.assertTrue(np.allclose(R, ground_truth_R, atol=1e-2))
        self.assertTrue(np.allclose(t, ground_truth_t, atol=1e-2))

        proposed_transformation_matrix = LA.Frame(R, t)
        proposed_transformed_points = LA.transform_points(proposed_transformation_matrix, random_points)

        for proposed, truth in zip(proposed_transformed_points, transformed_points):
            self.assertTrue(np.allclose(proposed.as_array(), truth.as_array(), atol=1e-2))

if __name__ == '__main__':
    unittest.main()