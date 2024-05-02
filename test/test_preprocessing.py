import unittest

from monai.transforms import Compose

from src.preprocessing import get_transforms


class TestGetTransforms(unittest.TestCase):
    def setUp(self):
        self.transform = get_transforms()

    def test_return_type(self):
        self.assertIsInstance(self.transform, Compose)

    def test_transform_length(self):
        self.assertTrue(len(self.transform.transforms) > 0)

    def test_transform_keys(self):
        expected_keys = ["image", "seg"]
        for transform in self.transform.transforms:
            self.assertEqual(transform.keys, expected_keys)


if __name__ == "__main__":
    unittest.main()
