import copy
import sys
import unittest

sys.path.append("..")

import torch

from src.custom_transforms import (
    AddBackgroundChannel,
    AddVesselContrast,
    RemoveNecrosisChannel,
    UndoOneHotEncoding,
)


class TestAddBackgroundChannel(unittest.TestCase):
    def setUp(self):
        self.keys = ["key1", "key2"]
        self.transform = AddBackgroundChannel(self.keys)

    def test_init(self):
        self.assertEqual(self.transform.keys, self.keys)

    def test_call(self):
        data = {
            "key1": torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32),
            "key2": torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32),
            "key3": torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32),
        }
        transformed_data = self.transform(data)
        self.assertTrue(
            torch.equal(
                transformed_data["key1"],
                torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                transformed_data["key2"],
                torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                transformed_data["key3"],
                torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32),
            )
        )


class TestRemoveNecrosisChannel(unittest.TestCase):
    def setUp(self):
        self.keys = ["key1", "key2"]
        self.transform = RemoveNecrosisChannel(self.keys)

    def test_init(self):
        self.assertEqual(list(self.transform.keys), self.keys)

    def test_call(self):
        data = {
            "key1": torch.randn(5, 2, 2),  # 5 channels
            "key2": torch.randn(4, 2, 2),  # 4 channels
        }
        data_copy = copy.deepcopy(data)
        transformed_data = self.transform(data_copy)
        exp_result = torch.cat((data["key1"][:2], data["key1"][3:]), axis=0)
        # Check if the 3rd channel is removed for key1
        self.assertEqual(transformed_data["key1"].shape[0], 4)
        self.assertTrue(torch.allclose(transformed_data["key1"], exp_result))

        # Check if the shape is unchanged for key2
        self.assertEqual(transformed_data["key2"].shape[0], 4)
        self.assertTrue(torch.equal(transformed_data["key2"], data["key2"]))


class TestAddVesselContrast(unittest.TestCase):
    def setUp(self):
        self.keys = ["image", "seg"]
        self.contrast_value = 1
        self.transform = AddVesselContrast(self.keys, self.contrast_value)

    def test_init(self):
        self.assertEqual(list(self.transform.keys), self.keys)
        self.assertEqual(self.transform.contrast_value, self.contrast_value)

    def test_call(self):
        data = {
            "image": torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32),
            "seg": torch.tensor(
                [[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [1, 0]]],
                dtype=torch.float32,
            ),
        }
        data_copy = copy.deepcopy(data)
        transformed_data = self.transform(data_copy)
        vessel_mask = data["seg"][2].unsqueeze(0)
        expected_image = data["image"].clone()
        expected_image[vessel_mask == 1] += self.contrast_value
        self.assertTrue(torch.allclose(transformed_data["image"], expected_image))

    def test_key_not_found_exception(self):
        data = {
            "image": torch.randn(1, 2, 2),
        }
        with self.assertRaises(Exception) as context:
            self.transform(data)
        self.assertTrue("Key seg not found in data" in str(context.exception))

    def test_image_seg_not_provided_exception(self):
        data = {
            "image": None,
            "seg": torch.tensor(
                [[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [1, 0]]],
                dtype=torch.float32,
            ),
        }
        with self.assertRaises(Exception) as context:
            self.transform(data)
        self.assertTrue(
            "Both 'image' and 'segmentation' must be provided" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
