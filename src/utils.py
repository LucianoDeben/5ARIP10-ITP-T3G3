import os
import re

import dicom2nifti
import nibabel as nib
import numpy as np
import pydicom
import torch


def dicom_nifti(data_root):
    """
    Convert DICOM images and segmentations folder to NIfTI format

    Args:
        data_root (str): The root directory of the dataset

    Returns:
        None
    """

    # Regular expression to match the patient directories
    patient_dir_pattern = re.compile(r"HCC_\d{3}")

    # Iterate over all directories in the dataset
    for directory in os.listdir(data_root):
        # If the directory is a patient directory
        if patient_dir_pattern.match(directory):

            patient_image_dir = os.path.join(data_root, directory, "300", "image")
            patient_seg_dir = os.path.join(data_root, directory, "300", "seg")

            # Convert the DICOM images to NIfTI
            dicom2nifti.convert_directory(
                patient_image_dir, patient_image_dir, compression=True, reorient=True
            )

            # Delete the original DICOM images
            for filename in os.listdir(patient_image_dir):
                if filename.endswith(".dcm"):
                    os.remove(os.path.join(patient_image_dir, filename))

            # Rename the NIfTI files to match the expected file names
            for filename in os.listdir(patient_image_dir):
                if filename.endswith(".nii.gz"):
                    output_file = os.path.join(patient_image_dir, filename)
                    new_file_name = os.path.join(
                        patient_image_dir, directory + ".nii.gz"
                    )

                    # If the destination file already exists, delete it
                    if os.path.exists(new_file_name):
                        os.remove(new_file_name)

                    os.rename(output_file, new_file_name)

            # Convert the DICOM segmentations to NIfTI
            for filename in os.listdir(patient_seg_dir):
                if filename.endswith(".dcm"):
                    dicom_file_path = os.path.join(patient_seg_dir, filename)
                    dicom_data = pydicom.dcmread(dicom_file_path)
                    array_data = dicom_data.pixel_array
                    array_data = np.transpose(
                        array_data, (2, 0, 1)
                    )  # Depending on the orientation of your DICOM data, you might need to transpose the array
                    nifti_image = nib.Nifti1Image(array_data, np.eye(4))
                    nib.save(
                        nifti_image,
                        os.path.join(patient_seg_dir, directory + "_seg.nii.gz"),
                    )

                    # Delete the original DICOM segmentations
                    os.remove(dicom_file_path)


def get_spacing(ct_image):
    try:
        # Get the CT volume and spacing from the image metadata
        volume = ct_image
        spacing = ct_image.meta["spacing"]

        # Convert the spacing to a list
        spacing = spacing.tolist()[0]

        # Convert the volume to a numpy array
        volume = volume.squeeze().numpy()

        return volume, spacing
    except AttributeError:
        raise AttributeError(
            "Invalid input: ct_image must be a valid CT image object with 'meta' attribute containing 'spacing'."
        )
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")


def get_iso_center(volume, spacing):

    # Check that volume and spacing have the correct types
    if not isinstance(volume, np.ndarray):
        raise TypeError(
            f"Invalid input: volume must be a numpy array. Got {type(volume)}."
        )
    if not isinstance(spacing, list):
        raise TypeError(f"Invalid input: spacing must be a list. Got {type(spacing)}.")

    # Calculate the isocenter
    iso = np.array(volume.shape[-3:]) * np.array(spacing) / 2

    # Return the isocenter coordinates as a tuple
    return tuple(iso.tolist())
