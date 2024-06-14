import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.model import Conv3DTo2D, ConvNet, TACEnet


class TestConv3DTo2D(unittest.TestCase):
    def setUp(self):
        self.model = Conv3DTo2D()
        self.input_tensor = torch.rand((1, 1, 512, 512, 96))

    def test_forward_return_type(self):
        output = self.model.forward(self.input_tensor)
        self.assertIsInstance(output, torch.Tensor)

    def test_forward_return_shape(self):
        output = self.model.forward(self.input_tensor)
        self.assertEqual(output.shape, (1, 1, 256, 256))

    def test_forward_no_nan(self):
        output = self.model.forward(self.input_tensor)
        self.assertFalse(torch.isnan(output).any())


class TestConvNet(unittest.TestCase):
    def setUp(self):
        self.model = ConvNet()
        self.input_tensor = torch.rand((1, 2, 256, 256))

    def test_forward_return_type(self):
        output = self.model.forward(self.input_tensor)
        self.assertIsInstance(output, torch.Tensor)

    def test_forward_return_shape(self):
        output = self.model.forward(self.input_tensor)
        self.assertEqual(output.shape, (1, 1, 256, 256))

    def test_forward_no_nan(self):
        output = self.model.forward(self.input_tensor)
        self.assertFalse(torch.isnan(output).any())


class TestTACEnet(unittest.TestCase):
    def setUp(self):
        self.model = TACEnet()
        self.VesselVolume = torch.rand((1, 1, 512, 512, 96))
        self.DRR = torch.rand((1, 1, 256, 256))

    def test_forward_return_type(self):
        output = self.model.forward(self.VesselVolume, self.DRR)
        self.assertIsInstance(output, tuple)

    def test_forward_return_length(self):
        output = self.model.forward(self.VesselVolume, self.DRR)
        self.assertEqual(len(output), 2)

    def test_forward_return_types(self):
        output = self.model.forward(self.VesselVolume, self.DRR)
        self.assertIsInstance(output[0], torch.Tensor)
        self.assertIsInstance(output[1], torch.Tensor)

    def test_forward_no_nan(self):
        output = self.model.forward(self.VesselVolume, self.DRR)
        self.assertFalse(torch.isnan(output[0]).any())
        self.assertFalse(torch.isnan(output[1]).any())


if __name__ == "__main__":
    unittest.main()
