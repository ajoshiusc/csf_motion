
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

# Input and output directories
input_dir = "/home/ajoshi/Desktop/Clio_CSF_Motion/ForAnand_TimeSTAMP/DATA/"
output_dir = "/home/ajoshi/Desktop/Clio_CSF_Motion/ForAnand_TimeSTAMP/REGISTERED_out/"
os.makedirs(output_dir, exist_ok=True)

# Find all NIfTI files in the input directory
nifti_files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]

for nifti_file in tqdm(nifti_files, desc="Files"):
    nifti_file_in = os.path.join(input_dir, nifti_file)
    nifti_file_out = os.path.join(output_dir, nifti_file.replace(".nii.gz", "_reg.nii.gz"))

    img = nib.load(nifti_file_in)
    data = img.get_fdata()

    registered_data = np.zeros_like(data)
    registered_data[:, :, :, 0] = data[:, :, :, 0]

    # Set up a ROBUST Registration Method for each file
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.5)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.001,
        numberOfIterations=500,
        relaxationFactor=0.5
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    print(f"Performing 2D slice-by-slice registration for {nifti_file}...")
    for t in tqdm(range(1, data.shape[3]), desc="Time Points", leave=False):
        for z in range(data.shape[2]):
            fixed_slice = sitk.GetImageFromArray(data[:, :, z, 0])
            moving_slice = sitk.GetImageFromArray(data[:, :, z, t])

            fixed_slice = sitk.Cast(fixed_slice, sitk.sitkFloat64)
            moving_slice = sitk.Cast(moving_slice, sitk.sitkFloat64)

            initial_transform = sitk.CenteredTransformInitializer(
                fixed_slice,
                moving_slice,
                sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            registration_method.SetInitialTransform(initial_transform)

            final_transform = registration_method.Execute(fixed_slice, moving_slice)

            resampled_slice = sitk.Resample(
                moving_slice,
                fixed_slice,
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_slice.GetPixelID(),
            )

            registered_data[:, :, z, t] = sitk.GetArrayFromImage(resampled_slice)

    print(f"Saving registered image to: {nifti_file_out}")
    registered_img = nib.Nifti1Image(registered_data, img.affine, img.header)
    nib.save(registered_img, nifti_file_out)
    print("Done! 👍")

