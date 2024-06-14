import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monai.apps import TciaDataset
from monai.data import DataLoader
from monai.transforms import Compose, ResizeWithPadOrCrop

from src.preprocessing import (
    add_vessel_contrast,
    get_dataloaders,
    get_datasets,
    get_eval_transforms,
    get_transforms,
)


class TestGetTransforms(unittest.TestCase):
    def setUp(self):
        self.transform = get_transforms()

    def test_return_type(self):
        self.assertIsInstance(self.transform, Compose)

    def test_transform_length(self):
        self.assertTrue(len(self.transform.transforms) > 0)

    def test_transform_keys(self):
        expected_keys = set(["image", "seg"])
        for transform in self.transform.transforms:
            self.assertTrue(set(transform.keys).issubset(expected_keys))


class TestGetEvalTransforms(unittest.TestCase):
    def setUp(self):
        self.resize_shape = [256, 256, 48]
        self.transform = get_eval_transforms(self.resize_shape)

    def test_return_type(self):
        self.assertIsInstance(self.transform, ResizeWithPadOrCrop)

    def test_lazy(self):
        self.assertFalse(self.transform.lazy)


class TestAddVesselContrast(unittest.TestCase):
    def setUp(self):
        self.image = torch.zeros((1, 1, 512, 512, 96))
        self.seg = torch.zeros((1, 4, 512, 512, 96))
        self.seg[:, 3] = 1  # Vessels are at index 3
        self.contrast_value = 1000

    def test_return_type(self):
        result = add_vessel_contrast(self.image, self.seg, self.contrast_value)
        self.assertIsInstance(result, torch.Tensor)

    def test_return_shape(self):
        result = add_vessel_contrast(self.image, self.seg, self.contrast_value)
        self.assertEqual(result.shape, self.image.shape)


if __name__ == "__main__":
    unittest.main()
