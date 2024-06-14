import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest

import torch

from diffdrr.data import load_example_ct
from src.drr import create_drr


class TestCreateDRR(unittest.TestCase):
    def setUp(self):
        self.subject = load_example_ct()
        self.sdd = 1020
        self.height = 200
        self.width = 200
        self.delx = 2.0
        self.dely = 2.0
        self.x0 = 0
        self.y0 = 0
        self.p_subsample = None
        self.reshape = True
        self.reverse_x_axis = True
        self.patch_size = None
        self.renderer = "siddon"
        self.rotations = torch.tensor([[0.0, 0.0, 0.0]])
        self.rotations_degrees = True
        self.translations = torch.tensor([[0.0, 850.0, 0.0]])
        self.mask_to_channels = True
        self.device = "cpu"

    def test_create_drr_return_type(self):
        drr_img = create_drr(
            self.subject,
            self.sdd,
            self.height,
            self.width,
            self.delx,
            self.dely,
            self.x0,
            self.y0,
            self.p_subsample,
            self.reshape,
            self.reverse_x_axis,
            self.patch_size,
            self.renderer,
            self.rotations,
            self.rotations_degrees,
            self.translations,
            self.mask_to_channels,
            self.device,
        )
        self.assertIsInstance(drr_img, torch.Tensor)

    def test_create_drr_return_shape(self):
        drr_img = create_drr(
            self.subject,
            self.sdd,
            self.height,
            self.width,
            self.delx,
            self.dely,
            self.x0,
            self.y0,
            self.p_subsample,
            self.reshape,
            self.reverse_x_axis,
            self.patch_size,
            self.renderer,
            self.rotations,
            self.rotations_degrees,
            self.translations,
            self.mask_to_channels,
            self.device,
        )
        self.assertEqual(drr_img.shape, (1, 119, self.height, self.width))

    def test_create_drr_invalid_renderer(self):
        with self.assertRaises(ValueError):
            create_drr(
                self.subject,
                self.sdd,
                self.height,
                self.width,
                self.delx,
                self.dely,
                self.x0,
                self.y0,
                self.p_subsample,
                self.reshape,
                self.reverse_x_axis,
                self.patch_size,
                "invalid_renderer",
                self.rotations,
                self.rotations_degrees,
                self.translations,
                self.mask_to_channels,
                self.device,
            )

    # Add more tests as needed


if __name__ == "__main__":
    unittest.main()
