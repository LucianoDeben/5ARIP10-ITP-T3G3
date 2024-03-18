import os
import re

import dicom2nifti
import nibabel as nib
import numpy as np
import pydicom


def dicom2nifti(data_root):

    # Regular expression to match the patient directories
    patient_dir_pattern = re.compile(r"HCC_\d{3}")

    # Iterate over all directories in the dataset
    for directory in os.listdir(data_root):
        # If the directory is a patient directory
        if patient_dir_pattern.match(directory):
            
            patient_image_dir = os.path.join(data_root, directory, "300", "image")
            patient_seg_dir = os.path.join(data_root, directory, "300", "seg")

            # Convert the DICOM images to NIfTI
            dicom2nifti.convert_directory(patient_image_dir, patient_image_dir, compression=True, reorient=True)
            
            # Delete the original DICOM images
            for filename in os.listdir(patient_image_dir):
                if filename.endswith(".dcm"):
                    os.remove(os.path.join(patient_image_dir, filename))
                    
            # Rename the NIfTI files to match the expected file names
            for filename in os.listdir(patient_image_dir):
                if filename.endswith(".nii.gz"):
                    output_file = os.path.join(patient_image_dir, filename)
                    new_file_name = os.path.join(patient_image_dir, directory + '.nii.gz')
                
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
                    array_data = np.transpose(array_data, (2, 0, 1))  # Depending on the orientation of your DICOM data, you might need to transpose the array
                    nifti_image = nib.Nifti1Image(array_data, np.eye(4))
                    nib.save(nifti_image, os.path.join(patient_seg_dir, directory + '_seg.nii.gz'))

                    # Delete the original DICOM segmentations
                    os.remove(dicom_file_path)