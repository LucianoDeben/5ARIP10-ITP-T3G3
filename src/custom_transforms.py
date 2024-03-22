import torch
from monai.transforms import MapTransform


class UndoOneHotEncoding(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            data[key] = data[key].argmax(dim=0).unsqueeze(0)
        return data
    
class AddBackgroundChannel(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            # Calculate the background channel
            background = 1 - data[key].sum(dim=0, keepdim=True)
            
            # Add the background channel to the segmentation
            data[key] = torch.cat([data[key], background], dim=0)
        return data
    
class AddNecrosisChannel(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            # Check if the segmentation has less than 5 channels
            if data[key].shape[0] < 5:
                # Add an extra channel of zeros at the 3rd position
                zeros = torch.zeros((1, data[key].shape[1], data[key].shape[2], data[key].shape[3]))
                data[key] = torch.cat((data[key][:2], zeros, data[key][2:]), axis=0)
        return data