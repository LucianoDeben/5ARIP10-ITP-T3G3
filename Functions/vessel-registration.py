import numpy as np
import warnings




def simpleVesselRegistration(DRR: np.ndarray, Segmentation: np.ndarray)->np.ndarray:
    """
    The most basic version of vessel registration where the vessels are directly mapped to its corresponding location in the 2D front view and the third axis is mapped to intensity.

    Params: 
    - DRR: np.ndarray
        The Digitally Reconstructed Radiograph in 2D with shape (512,96)
    - Segmentation: np.ndarray
        The vessel segmentation in 3D with shape (512,512,96)

    Returns:
    - A 2D segmentation overlay with shape (512,96)
    """

    #Check the input shapes
    if not (np.shape(DRR) == (512,96)):
        warnings.warn("DRR shape does not match expectation")

    if not (np.shape(Segmentation) == (512,512,96)):
        warnings.warn("Segmentation shape does not match expectation")

    #Sum over depth axis
    flatSegmentation = np.sum(Segmentation, axis=1)

    #MinMax Normalization
    normalizedFlatSegmentation = flatSegmentation / (np.max(flatSegmentation))

    #Assert shape
    assert (np.shape(normalizedFlatSegmentation)) == (np.shape(DRR)), "Shape of the registration does not match the shape of the DRR"

    return normalizedFlatSegmentation






    