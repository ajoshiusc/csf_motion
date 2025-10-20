import os
import argparse
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice-wise 2D registration across time for 4D NIfTI volumes")
    parser.add_argument("--input", required=True, help="Input NIfTI file or directory containing .nii.gz files")
    parser.add_argument("--output", required=True, help="Output file (if input is file) or output directory")
    parser.add_argument("--mode", choices=["rigid", "nonlin"], default="rigid", help="Registration mode")
    parser.add_argument("--metric", choices=["mi", "ms", "corr"], default="ms", help="Similarity metric")
    parser.add_argument("--sampling", type=float, default=0.5, help="Metric sampling percentage (0..1]")
    parser.add_argument("--pyramid", default="4,2,1", help="Shrink factors per level, e.g. 4,2,1")
    parser.add_argument("--sigmas", default="2,1,0", help="Smoothing sigmas per level, e.g. 2,1,0")
    parser.add_argument("--bspline-mesh", type=int, default=10, help="B-spline control points per dimension (2D slice)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for metric sampling")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for slice processing")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="Working dtype for processing")
    parser.add_argument("--suffix", default="_reg", help="Suffix to append to output filenames when input is a directory")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-slice logging")
    return parser.parse_args()


def parse_int_list(csv: str) -> List[int]:
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


def configure_registration(
    mode: str,
    metric: str,
    sampling: float,
    shrink_factors: List[int],
    smoothing_sigmas: List[int],
    seed: int,
) -> sitk.ImageRegistrationMethod:
    reg = sitk.ImageRegistrationMethod()

    # Metric
    if metric == "ms":
        reg.SetMetricAsMeanSquares()
    elif metric == "corr":
        reg.SetMetricAsCorrelation()
    else:
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling)
    try:
        reg.SetMetricSamplingSeed(seed)
    except Exception:
        pass

    reg.SetInterpolator(sitk.sitkLinear)

    # Optimizers
    if mode == "rigid":
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=0.001,
            numberOfIterations=500,
            relaxationFactor=0.5,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
    else:
        reg.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=100,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1024,
            costFunctionConvergenceFactor=1e7,
        )

    # Multi-resolution
    reg.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return reg


def get_file_list(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        return [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".nii.gz")]
    return [input_path]


def ensure_output_path(output: str, input_is_dir: bool) -> None:
    if input_is_dir:
        os.makedirs(output, exist_ok=True)
    else:
        out_dir = os.path.dirname(os.path.abspath(output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)


def register_slice(
    fixed_slice: sitk.Image,
    moving_slice: sitk.Image,
    mode: str,
    metric: str,
    sampling: float,
    shrink_factors: List[int],
    smoothing_sigmas: List[int],
    seed: int,
    bspline_mesh: int,
    quiet: bool,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    try:
        # Create a fresh, fully configured registration object per slice (thread-safe)
        reg = configure_registration(
            mode=mode,
            metric=metric,
            sampling=sampling,
            shrink_factors=shrink_factors,
            smoothing_sigmas=smoothing_sigmas,
            seed=seed,
        )

        if mode == "rigid":
            init = sitk.CenteredTransformInitializer(
                fixed_slice,
                moving_slice,
                sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            reg.SetInitialTransform(init, inPlace=False)
        else:
            mesh = [bspline_mesh] * fixed_slice.GetDimension()
            xform = sitk.BSplineTransformInitializer(
                image1=fixed_slice,
                transformDomainMeshSize=mesh,
                order=3,
            )
            reg.SetInitialTransform(xform, inPlace=False)

        final_transform = reg.Execute(fixed_slice, moving_slice)

        resampled = sitk.Resample(
            moving_slice,
            fixed_slice,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving_slice.GetPixelID(),
        )
        metric_value = None
        try:
            metric_value = reg.GetMetricValue()
        except Exception:
            metric_value = None
        return sitk.GetArrayFromImage(resampled), metric_value
    except Exception as e:
        if not quiet:
            print(f"Slice registration failed: {e}")
        return None, None


def process_file(
    in_file: str,
    out_path: str,
    is_dir_mode: bool,
    args: argparse.Namespace,
) -> None:
    img = nib.load(in_file)
    data = img.get_fdata(dtype=np.float32 if args.dtype == "float32" else np.float64)

    registered = np.zeros_like(data)
    registered[:, :, :, 0] = data[:, :, :, 0]

    shrink = parse_int_list(args.pyramid)
    sigmas = parse_int_list(args.sigmas)

    t_iter = tqdm(range(1, data.shape[3]), desc=os.path.basename(in_file) + " [t]", leave=False)
    for t in t_iter:
        Z = data.shape[2]
        if args.threads > 1:
            with ThreadPoolExecutor(max_workers=args.threads) as ex:
                futures = {}
                for z in range(Z):
                    fixed_slice = sitk.GetImageFromArray(data[:, :, z, 0])
                    moving_slice = sitk.GetImageFromArray(data[:, :, z, t])
                    pixel_type = sitk.sitkFloat32 if args.dtype == "float32" else sitk.sitkFloat64
                    fixed_slice = sitk.Cast(fixed_slice, pixel_type)
                    moving_slice = sitk.Cast(moving_slice, pixel_type)
                    fut = ex.submit(
                        register_slice,
                        fixed_slice,
                        moving_slice,
                        args.mode,
                        args.metric,
                        args.sampling,
                        shrink,
                        sigmas,
                        args.seed,
                        args.bspline_mesh,
                        args.quiet,
                    )
                    futures[fut] = z
                for fut in as_completed(futures):
                    z = futures[fut]
                    result, metric_value = fut.result()
                    if result is None:
                        registered[:, :, z, t] = data[:, :, z, t]
                    else:
                        registered[:, :, z, t] = result
                        if not args.quiet and metric_value is not None and z == 0:
                            # Log one slice metric per time-point to avoid spam
                            print(f"t={t}: metric={metric_value:.6f}")
        else:
            for z in range(Z):
                fixed_slice = sitk.GetImageFromArray(data[:, :, z, 0])
                moving_slice = sitk.GetImageFromArray(data[:, :, z, t])
                pixel_type = sitk.sitkFloat32 if args.dtype == "float32" else sitk.sitkFloat64
                fixed_slice = sitk.Cast(fixed_slice, pixel_type)
                moving_slice = sitk.Cast(moving_slice, pixel_type)
                result, metric_value = register_slice(
                    fixed_slice=fixed_slice,
                    moving_slice=moving_slice,
                    mode=args.mode,
                    metric=args.metric,
                    sampling=args.sampling,
                    shrink_factors=shrink,
                    smoothing_sigmas=sigmas,
                    seed=args.seed,
                    bspline_mesh=args.bspline_mesh,
                    quiet=args.quiet,
                )
                if result is None:
                    registered[:, :, z, t] = data[:, :, z, t]
                else:
                    registered[:, :, z, t] = result
                    if not args.quiet and metric_value is not None and z == 0:
                        print(f"t={t}: metric={metric_value:.6f}")

    if is_dir_mode:
        base = os.path.basename(in_file)
        if base.endswith(".nii.gz"):
            base = base[:-7]
        out_file = os.path.join(out_path, f"{base}{args.suffix}.nii.gz")
    else:
        out_file = out_path

    nib.save(nib.Nifti1Image(registered, img.affine, img.header), out_file)


def main() -> None:
    args = parse_args()
    files = get_file_list(args.input)
    is_dir_mode = os.path.isdir(args.input)
    ensure_output_path(args.output, input_is_dir=is_dir_mode)

    for f in tqdm(files, desc="Files"):
        process_file(f, args.output, is_dir_mode, args)


if __name__ == "__main__":
    main()


