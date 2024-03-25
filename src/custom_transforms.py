import torch
from monai.transforms import MapTransform


class UndoOneHotEncoding(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            data[key] = data[key].argmax(dim=0).unsqueeze(0)
        return data

class AddBackgroundChannel:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            # Get the current segmentation
            seg = data[key]

            # Create a new channel that is 1 where all other channels are 0
            background = (seg.sum(dim=0) == 0).float()

            # Add the new channel to the segmentation
            data[key] = torch.cat([seg, background.unsqueeze(0)], dim=0)

        return data
    
class RemoveNecrosisChannel(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            # Check if the segmentation has 4 or more channels
            #print(data[key].shape)
            if data[key].shape[0] >= 4:
                # Remove the 3rd channel
                data[key] = torch.cat((data[key][:2], data[key][3:]), axis=0)
        return data
    
class RemoveDualImage(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        print(data['image'].size())
        if data['image'].size()[3] > data['seg'].size()[3]:
            data['image'] = data['image'][:, :, :, :(data['seg'].size()[3])]
        return data