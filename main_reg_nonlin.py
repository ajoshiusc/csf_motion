import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

## 1. Load Data
nifti_file_in = "/home/ajoshi/Desktop/Clio_CSF_Motion/ForAnand_TimeSTAMP/DATA/CSF022_S2T08.nii.gz"
nifti_file_out = "/home/ajoshi/Desktop/Clio_CSF_Motion/ForAnand_TimeSTAMP/DATA/CSF022_S2T08_registered_2D_nonlinear.nii.gz"

img = nib.load(nifti_file_in)
data = img.get_fdata()

# Create an empty numpy array to store the registered data
registered_data = np.zeros_like(data)

# The first time volume (t=0) is our reference
registered_data[:, :, :, 0] = data[:, :, :, 0]

## 2. Set up NONLINEAR Registration Method
# This object is configured once and reused for efficiency.
registration_method = sitk.ImageRegistrationMethod()

# -- Similarity Metric --
# Mattes Mutual Information remains a good choice.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.5)

# -- Interpolator --
registration_method.SetInterpolator(sitk.sitkLinear)

# -- Optimizer for Nonlinear Registration --
# **FIX 1: Switch to an optimizer better suited for high-dimensional problems.**
# L-BFGS-B is a quasi-Newton method that is effective for B-spline registration.
registration_method.SetOptimizerAsLBFGSB(
    gradientConvergenceTolerance=1e-5,
    numberOfIterations=100, # Iterations per pyramid level
    maximumNumberOfCorrections=5,
    maximumNumberOfFunctionEvaluations=1024,
    costFunctionConvergenceFactor=1e+7
)

# -- Multi-Resolution Framework --
# This is crucial for robust nonlinear registration.
registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

## 3. Run Slice-by-Slice NONLINEAR Registration
print("Performing 2D slice-by-slice NONLINEAR registration...")

# Iterate over each time point (starting from the second one)
for t in tqdm(range(1, data.shape[3]), desc="Time Points"):
    # Iterate over each slice in the z-dimension
    for z in range(data.shape[2]):
        # Extract the same 2D slice (z) from the reference (t=0) and moving volumes
        fixed_slice = sitk.GetImageFromArray(data[:, :, z, 0])
        moving_slice = sitk.GetImageFromArray(data[:, :, z, t])

        # Cast to a float type required for registration
        fixed_slice = sitk.Cast(fixed_slice, sitk.sitkFloat64)
        moving_slice = sitk.Cast(moving_slice, sitk.sitkFloat64)
        
        # **FIX 2: Define the B-spline transform for nonlinear warping.**
        # The mesh size acts as the regularizer. A larger mesh size (e.g., 50mm)
        # results in a smoother, more regularized deformation.
        transform_domain_mesh_size = [10] * fixed_slice.GetDimension()
        initial_transform = sitk.BSplineTransformInitializer(
            image1=fixed_slice,
            transformDomainMeshSize=transform_domain_mesh_size,
            order=3 # Use cubic B-splines
        )
        
        # Set the B-spline transform as the initial transform
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        # Execute the registration
        final_transform = registration_method.Execute(fixed_slice, moving_slice)

        # Resample the moving slice to align it with the fixed slice
        resampled_slice = sitk.Resample(
            moving_slice,
            fixed_slice,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving_slice.GetPixelID(),
        )

        # Place the registered 2D slice back into the correct 4D position
        registered_data[:, :, z, t] = sitk.GetArrayFromImage(resampled_slice)

## 4. Save the Result
print(f"Saving registered image to: {nifti_file_out}")
registered_img = nib.Nifti1Image(registered_data, img.affine, img.header)
nib.save(registered_img, nifti_file_out)
print("Done! 👍")