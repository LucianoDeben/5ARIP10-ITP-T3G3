import unittest

import monai
import numpy as np
import torch
from monai.data import MetaTensor

# Assuming the function is in a file named utils.py
from src.utils import get_iso_center, get_spacing


class TestGetSpacing(unittest.TestCase):
    def setUp(self):
        # Create a dummy CT image with known volume and spacing
        self.ct_image = MetaTensor(
            torch.randn(10, 10, 10), meta={"spacing": torch.tensor([[2.0, 2.0, 2.0]])}
        )

    def test_get_spacing(self):
        volume, spacing = get_spacing(self.ct_image)
        self.assertIsInstance(volume, np.ndarray)
        self.assertIsInstance(spacing, list)
        self.assertEqual(spacing, [2.0, 2.0, 2.0])
        self.assertEqual(volume.shape, (10, 10, 10))

    def test_invalid_input(self):
        with self.assertRaises(AttributeError):
            get_spacing(None)


class TestGetIsoCenter(unittest.TestCase):
    def setUp(self):
        # Create a dummy CT image with known volume and spacing
        self.volume = np.random.rand(10, 10, 10)
        self.spacing = [4.0, 4.0, 4.0]

    def test_get_iso_center(self):
        iso_center = get_iso_center(self.volume, self.spacing)
        self.assertIsInstance(iso_center, tuple)
        self.assertEqual(len(iso_center), 3)
        self.assertEqual(iso_center, (20.0, 20.0, 20.0))

    def test_invalid_volume(self):
        with self.assertRaises(TypeError):
            get_iso_center(None, self.spacing)

    def test_invalid_spacing(self):
        with self.assertRaises(TypeError):
            get_iso_center(self.volume, None)


if __name__ == "__main__":
    unittest.main()
