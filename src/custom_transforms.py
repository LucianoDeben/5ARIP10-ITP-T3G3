import os
from typing import Hashable, Optional

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, SaveImaged


class UndoOneHotEncoding(MapTransform):
    """
    Undo the one-hot encoding of the segmentation

    Args:
        keys (list): The keys to which the transform should be applied

    Returns:
        data (dict): The data dictionary with the segmentation as a single channel
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            data[key] = data[key].argmax(dim=0).unsqueeze(0)
        return data


class AddBackgroundChannel(MapTransform):
    """
    Add a background channel to the segmentation

    Args:
        keys (list): The keys to which the transform should be applied

    Returns:
        data (dict): The data dictionary with the background channel added
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            # Get the current segmentation
            seg = data[key]

            # Create a new channel that is 1 where all other channels are 0
            background = (seg.sum(dim=0) == 0).float()

            # Add the new channel to the segmentation
            data[key] = torch.cat([background.unsqueeze(0), seg], dim=0)

        return data


class ConvertToSingleChannel(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            # Stack along the channel dimension
            data[key] = np.argmax(data[key], axis=0).astype(np.int32)

            # Unsqueeze the channel dimension
            data[key] = torch.tensor(data[key]).unsqueeze(0)

        return data


class RemoveNecrosisChannel(MapTransform):
    """
    Remove the necrosis channel from the segmentation

    Args:
        keys (list): The keys to which the transform should be applied

    Returns:
        data (dict): The data dictionary with the necrosis channel removed
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            # Check if the segmentation has more then 4 channels
            if data[key].shape[0] > 4:
                # Remove the 3rd channel
                data[key] = torch.cat((data[key][:2], data[key][3:]), axis=0)
        return data


class ConvertToHU(MapTransform):
    """
    Convert the pixel values to Hounsfield Units

    Args:
        keys (list): The keys to which the transform should be applied

    Returns:
        data (dict): The data dictionary with the pixel values converted to Hounsfield Units
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:

            if key == "seg":
                raise ValueError("The key 'seg' is not a valid key for ConvertToHU")

            # Get the pixel array
            image = data[key]

            # Get the RescaleSlope and RescaleIntercept from the metadata
            rescale_slope = image.meta["RescaleSlope"]
            rescale_intercept = image.meta["RescaleIntercept"]

            # Convert to Hounsfield Units
            hu_array = image * rescale_slope + rescale_intercept

            # Replace the pixel array with the HU array
            data[key] = torch.tensor(hu_array).unsqueeze(0)

        return data


class AddVesselContrast(MapTransform):

    def __init__(self, keys, contrast_value):
        super().__init__(keys)
        self.contrast_value = contrast_value

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                raise Exception(f"Key {key} not found in data")

        # Get the current segmentation and image
        image = d.get("image")
        seg = d.get("seg")

        if image is None or seg is None:
            raise Exception("Both 'image' and 'segmentation' must be provided")

        # Add the vessel contrast to the image
        vessel_mask = seg[2]
        vessel_mask = vessel_mask.unsqueeze(0)
        image[vessel_mask == 1] += self.contrast_value
        return d


class RemoveDualImage(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        if data["image"].size()[3] > data["seg"].size()[3]:
            data["image"] = data["image"][:, :, :, : (data["seg"].size()[3])]
        return data


class IsolateArteries(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):

        data["seg"] = data["seg"][2, :, :, :]
        # print(data['seg'].size())
        return data
