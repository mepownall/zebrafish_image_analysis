import os
import glob
import json
import numpy as np
from skimage import exposure
from skimage import io, measure
from scipy.ndimage import median_filter
from tifffile import TiffFile, imwrite
import tifffile as tiff
import subprocess
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from datetime import datetime
import pandas as pd
from skimage.measure import regionprops, label
from aicsimageio import AICSImage
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import re
import cv2
from scipy.stats import pearsonr
import math
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from scipy.signal import correlate
from typing import Optional, List, Dict          
from pathlib import Path                  
import shutil                             
import csv                                
from skimage.morphology import convex_hull_image  




def load_config(config_path):
    """
    Load pipeline configuration from a JSON file.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: Configuration dictionary containing pipeline parameters.
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def show_image_napari(image, pixel_sizes=None, title="Image Viewer"):
    """
    Display an image in Napari for visual inspection.

    Parameters
    ----------
    image : np.ndarray
        Image data to display.
    pixel_sizes : tuple, optional
        Physical voxel size as (Z, Y, X).
    title : str
        Viewer window title.
    """
    import napari

    viewer = napari.Viewer(title=title)

    if pixel_sizes and len(pixel_sizes) == 3:
        viewer.add_image(
            image,
            name="Image",
            scale=pixel_sizes,
        )
    else:
        viewer.add_image(image, name="Image")

    napari.run()

def log(message, log_file=None):
    """
    Log a message to the console and optionally to a file.

    Args:
        message (str): Message to log.
        log_file (str): Path to a log file where the message will also be saved (optional).
    """
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {message}")
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"{timestamp} {message}\n")

# Global dictionary for metadata caching
metadata_cache = {}
"""Cache image metadata to avoid repeated file reads."""

def get_cached_subfolder_name(mask_file):
    """
    Extract the subfolder name from a mask file path, ensuring it matches the metadata cache key.

    Args:
        mask_file (str): Path to the mask file.

    Returns:
        str: Cleaned subfolder name matching the metadata cache key.
    """
    # Extract the basename of the file
    basename = os.path.basename(mask_file)
    # Remove file extension first
    basename = os.path.splitext(basename)[0]
    # Remove suffixes like "_cp-masks"
    if "_cp_masks" in basename:
        basename = basename.rsplit("_cp_masks", 1)[0]
    # Remove prefixes like "C4-"
    if basename.startswith("C"):
        basename = basename.split("-", 1)[-1]
    return basename


def universal_read_tiff(file_path, log_file=None):
    """
    Universal function to read an OME-TIFF file, extract metadata, and load all channels.

    Args:
        file_path (str): Path to the OME-TIFF file.
        log_file (str, optional): Path to a log file where messages are recorded.

    Returns:
        tuple: A tuple containing the AICSImage object, the image stack (np.ndarray), and metadata.
    """
    log(f"Reading file: {file_path}", log_file)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found or invalid: {file_path}")
    try:
        # Read the file using AICSImageIO
        img = AICSImage(file_path)

        # Load the full image stack including all channels
        # Shape order: (C, Z, Y, X)
        image_stack = img.get_image_data("CZYX")
        metadata = {
            "physical_pixel_sizes": img.physical_pixel_sizes,
            "channel_names": img.channel_names,
        }
        print(metadata)
        log(f"Loaded image stack shape: {image_stack.shape} (C, Z, Y, X)", log_file)

        return img, image_stack, metadata
    except Exception as e:
        log(f"Error reading file {file_path}: {e}", log_file)
        raise

def universal_read_tiff_multi(file_paths, log_file=None):
    """
    Universal function to read multiple OME-TIFF files as a single series.

    Args:
        file_paths (list of str): List of paths to TIFF files in the same subfolder.
        log_file (str, optional): Path to a log file where messages are recorded.

    Returns:
        tuple: A tuple containing the combined image stack (np.ndarray) and metadata (dict).
    """
    log(f"Reading files as a single multi-series image: {file_paths}", log_file)

    try:
        # Use the first file to represent the series (AICSImage can infer the full set)
        representative_file = file_paths[0]
        img = AICSImage(representative_file)

        # Read the entire series as one logical stack
        stack = img.get_image_data("CZYX")  # Shape: (C, Z, Y, X)
        metadata = {
            "physical_pixel_sizes": img.physical_pixel_sizes,
            "channel_names": img.channel_names,
        }

        # Log the loaded stack details
        log(f"Loaded combined image stack shape: {stack.shape} (C, Z, Y, X)", log_file)

        return stack, metadata

    except Exception as e:
        log(f"Error reading files as a single multi-series image: {e}", log_file)
        raise

def combined_reading(input_dir, log_file=None):
    """
    Read TIFF files from subfolders and determine whether to use single or multi-file reading logic.

    Args:
        input_dir (str): Path to the directory containing subfolders with TIFF files.
        log_file (str, optional): Path to a log file for recording progress and errors.

    Yields:
        dict: Contains the subfolder name, combined stack, and metadata.
              Format: {"subfolder": subfolder, "stack": combined_stack, "metadata": metadata}
    """
    # Find all subfolders in the input directory
    subfolders = [
        subfolder for subfolder in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, subfolder))
    ]

    # Iterate over subfolders to read TIFF files
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_dir, subfolder)
        tiff_files = sorted(set(
            glob.glob(os.path.join(subfolder_path, "*.ome.tif")) +
            glob.glob(os.path.join(subfolder_path, "*.tif")) +
            glob.glob(os.path.join(subfolder_path, "*.tiff"))
        ))

        if not tiff_files:
            log(f"No OME-TIFF files found in subfolder: {subfolder_path}", log_file)
            continue

        try:
            # Read and combine multiple TIFF files if needed
            if len(tiff_files) > 1:
                log(f"Multiple TIFF files found in {subfolder_path}. Combining files.", log_file)
                combined_stack, metadata = universal_read_tiff_multi(tiff_files, log_file)
            else:
                # Single file case
                img, combined_stack, metadata = universal_read_tiff(tiff_files[0], log_file)

            # Yield the results for this subfolder
            yield {
                "subfolder": subfolder,
                "stack": combined_stack,
                "metadata": metadata,
            }

        except Exception as e:
            log(f"Error processing subfolder {subfolder}: {e}", log_file)

def save_as_ome_tiff(output_stack, output_file, metadata=None):
    """
    Save a modified NumPy array as an OME-TIFF file with metadata, ensuring correct axis interpretation.

    Args:
        output_stack (np.ndarray): The modified image data to save.
        output_file (str): Path to save the new OME-TIFF file.
        metadata (dict): Metadata dictionary containing relevant information.
    """
    # Persistent variable to store the last metadata used
    if not hasattr(save_as_ome_tiff, "_last_metadata"):
        save_as_ome_tiff._last_metadata = None  # Initialize if not already set

    # Use provided metadata, or fall back to the last metadata if available
    if metadata is not None:
        save_as_ome_tiff._last_metadata = metadata
    elif save_as_ome_tiff._last_metadata is None:
        raise ValueError("Metadata must be provided for the first call.")

    current_metadata = save_as_ome_tiff._last_metadata    

    try:
        # Extract voxel sizes (physical pixel sizes)
        physical_pixel_sizes = current_metadata.get("physical_pixel_sizes", (1.0, 1.0, 1.0))
        pixel_size_x = physical_pixel_sizes[1]  # X spacing
        pixel_size_y = physical_pixel_sizes[2]  # Y spacing
        pixel_size_z = physical_pixel_sizes[0]  # Z spacing

        # Calculate resolution in dots per micrometer for X and Y
        resolution_x = 1 / pixel_size_x  # In microns
        resolution_y = 1 / pixel_size_y  # In microns

        # Handle dimensionality
        if output_stack.ndim == 4:  # Multi-channel stack (C, Z, Y, X)
            # Reorder to (Z, C, Y, X) for ImageJ compatibility
            output_stack = np.transpose(output_stack, (1, 0, 2, 3))
            # Define LUTs for multi-channel images
            true_colors = [
                [0, 255, 255],  # Cyan for channel 1
                [255, 0, 255],  # Magenta for channel 2
                [255, 255, 0],  # Yellow for channel 3
                [128, 128, 128],  # Gray for channel 4
            ]
            num_channels = output_stack.shape[1]
            luts = []
            for i in range(num_channels):
                r, g, b = true_colors[i % len(true_colors)]
                lut = np.array([
                    np.linspace(0, r, 256),  # Red gradient
                    np.linspace(0, g, 256),  # Green gradient
                    np.linspace(0, b, 256),  # Blue gradient
                ], dtype=np.uint8)
                luts.append(lut)
            # ImageJ-specific metadata for multi-channel images
            imagej_metadata = {
                "spacing": pixel_size_z,  # Z voxel size
                "unit": "micron",         # Unit for all dimensions
                "mode": "composite",      # Open in composite mode
                "LUTs": luts,             # Assign LUTs to channels
            }

        elif output_stack.ndim == 3:  # Single-channel stack (Z, Y, X)
            output_stack = np.transpose(output_stack, (0, 1, 2))
            imagej_metadata = {
                "spacing": pixel_size_z,
                "unit": "micron",
                "channels": 0,  # Explicitly state that the image has one channel
                "slices": output_stack.shape[0],  # Number of Z slices
                "frames": 1,  # Default to 1 frame for simplicity
                "axes": "ZYX",
            }
           
        else:
            raise ValueError(f"Unexpected output stack dimensions: {output_stack.ndim}")
        # Write the TIFF file with proper ImageJ metadata
        tiff.imwrite(
            output_file,
            output_stack,
            photometric="minisblack",
            resolution=(resolution_x, resolution_y),  # Resolution for X and Y
            metadata=imagej_metadata,                # Include Z-spacing, composite mode, and LUTs
            imagej=True                              # Ensure compatibility with ImageJ
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save OME-TIFF file {output_file}: {e}")



def split_channels(image_stack, metadata, channels, output_dir, subfolder, log_file):
    """
    Split channels from a multichannel stack and save each channel with voxel calibration.

    Args:
        image_stack (np.ndarray): Multichannel image stack to process.
        metadata (dict): Metadata containing voxel size information.
        channels (dict): Dictionary mapping channel names to indices (1-based).
        output_dir (str): Path to save split channel images.
        subfolder (str): Subfolder name used to construct output filenames.
        log_file (str): Path to a log file where the process is logged.
    """
    global metadata_cache  # Access the global metadata cache

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if metadata is already cached for this subfolder
    if subfolder in metadata_cache:
        log(f"Using cached metadata for subfolder: {subfolder}", log_file)
        cached_metadata = metadata_cache[subfolder]
    else:
        log(f"Caching metadata for subfolder: {subfolder}", log_file)
        metadata_cache[subfolder] = metadata
        cached_metadata = metadata

    for channel_name, channel_index in channels.items():
        # Create a subfolder for each channel
        channel_dir = os.path.join(output_dir, channel_name)
        os.makedirs(channel_dir, exist_ok=True)

        # Construct the output path
        output_path = os.path.join(channel_dir, f"C{channel_index}-{subfolder}.tif")

        if not isinstance(output_path, str):
            raise ValueError(f"Invalid output path: {output_path}")

        # Skip if output already exists
        if os.path.exists(output_path):
            log(f"Skipping split for {channel_name} in {subfolder} (output exists).", log_file)
            continue

        # Extract and save the specified channel
        try:
            channel_image = image_stack[channel_index - 1]
            save_as_ome_tiff(channel_image, output_path, cached_metadata)
            log(f"Saved {channel_name} to {output_path} with voxel calibration.", log_file)
        except Exception as e:
            log(f"Error saving {channel_name} for {subfolder}: {e}", log_file)

def subtract_background(split_channel_image, dark_image, channels, channel_name, output_dir, subfolder, metadata, log_file):
    """
    Subtract the background (dark image) from each z-slice of a split channel image and save the result,
    skipping if the output file already exists.
    If dark_image shape != slice shape, subtract the scalar mean(dark_image) instead.

    Args:
        split_channel_image (str or np.ndarray): Path to the 3D split channel image file (OME-TIFF)
            or a 3D NumPy array (z, y, x) representing the split channel image stack.
        dark_image (np.ndarray): 2D NumPy array (y, x) representing the background image to subtract.
        channels (dict): Dictionary mapping channel names (str) to their numerical indices (int).
        channel_name (str): Name of the channel being processed (e.g., "DNA").
        output_dir (str): Path to the base output directory where background-subtracted images will be saved.
        subfolder (str): Subfolder name used to construct output filenames.
        metadata (dict): Metadata containing voxel calibration information.
        log_file (str): Path to a log file where messages will be recorded.

    Returns:
        None: The background-subtracted image is saved as an OME-TIFF file.
    """
    try:
        # Prepare the output directory and path
        bg_subtracted_dir = os.path.join(output_dir, f"{channel_name}")
        os.makedirs(bg_subtracted_dir, exist_ok=True)
        output_path = os.path.join(bg_subtracted_dir, f"C{channels[channel_name]}-{subfolder}.tif")

        # Skip processing if the output already exists
        if os.path.exists(output_path):
            log(f"Skipping background subtraction for {channel_name} in {subfolder} (output exists).", log_file)
            return

        # If split_channel_image is a file path, read it
        if isinstance(split_channel_image, str):
            log(f"split_channel_image is a file path. Reading image: {split_channel_image}", log_file)
            _, image_stack, _ = universal_read_tiff(split_channel_image, log_file=log_file)
            split_channel_image = image_stack[0, :, :, :]  # Assuming single channel, 3D stack
            log(f"Loaded split_channel_image from file. Shape: {split_channel_image.shape}", log_file)

        # Decide subtraction mode: per-pixel vs scalar mean
        use_scalar_bg = False
        if dark_image.shape != split_channel_image.shape[1:]:
            # Mismatch → use scalar mean of dark image
            dark_scalar = float(np.mean(dark_image))
            use_scalar_bg = True
            log(
                f"Dimension mismatch: dark_image {dark_image.shape} vs split_channel_image slices {split_channel_image.shape[1:]}. "
                f"Using scalar mean background subtraction (mean={dark_scalar:.3f}).",
                log_file
            )
        else:
            log(
                f"Validated dimensions: dark_image {dark_image.shape}, split_channel_image {split_channel_image.shape}",
                log_file
            )

        # Define a function to process a single Z-slice
        def process_slice(z):
            """
            Subtracts the background (per-pixel or scalar) from a single Z-slice and clips the result.
            """
            slice_int32 = split_channel_image[z].astype(np.int32, copy=False)
            if use_scalar_bg:
                bg_subtracted_slice = slice_int32 - int(round(dark_scalar))
            else:
                dark_int32 = dark_image.astype(np.int32, copy=False)
                bg_subtracted_slice = slice_int32 - dark_int32
            return np.clip(bg_subtracted_slice, 0, 65535).astype(np.uint16, copy=False)

        # Process all Z-slices in parallel
        log("Starting multithreaded background subtraction...", log_file)
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            bg_subtracted_stack = list(executor.map(process_slice, range(split_channel_image.shape[0])))

        # Convert to a NumPy array (already uint16 from process_slice)
        bg_subtracted_stack = np.asarray(bg_subtracted_stack, dtype=np.uint16)

        # Save the resulting background-subtracted stack
        log(f"Saving background-subtracted image to: {output_path}", log_file)
        save_as_ome_tiff(bg_subtracted_stack, output_path, metadata)
        log(f"Successfully saved background-subtracted image for {channel_name} to {output_path}.", log_file)

    except Exception as e:
        log(f"Error in background subtraction for {channel_name} in {subfolder}: {e}", log_file)
        raise
        


def run_nuclei_unet(
    nuclei_unet_script: str,
    split_channels_dir: str,
    output_dir:str,
    filtered_dna_dir: str,
    use_bg_subtraction: bool,
    log_file: Optional[str] = None,
    model_path: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
):
    """
    Run custom unet for nuclei segmentation.

    Args:
        nuclei_unet_script (str): Path to the nuclei unet inference script (e.g., 250904_segment_with_u-net_Resblock.py).
        split_channels_dir (str): Path to base directory with split channels (expects 'DNA/' inside).
        filtered_dna_dir (str): Path to directory with background-subtracted (or otherwise preprocessed) DNA images.
        use_bg_subtraction (bool): If True, prefer `filtered_dna_dir`; else use `split_channels/DNA`.
        log_file (str, optional): Path to a log file where the process is logged.
        model_path (str, optional): Path to the unet weights (.pth). If None, the script must have its own default.
        extra_args (List[str], optional): Additional CLI args to pass through to the unet script.
    """
    # Choose input directory (default to bg-subtracted DNA if requested)
    dna_from_split = os.path.join(split_channels_dir, "DNA")
    input_dir = filtered_dna_dir if use_bg_subtraction else dna_from_split

    def _log(msg: str):
        print(msg)
        if log_file:
            try:
                with open(log_file, "a") as log:
                    log.write(msg.rstrip() + "\n")
            except Exception:
                pass

    # Gather inputs
    dna_files = sorted(glob.glob(os.path.join(input_dir, "*.tif*")))
    if not dna_files:
        _log(f"No DNA files found for unet nuclei segmentation in {input_dir}. Skipping.")
        return

    _log(f"Running unet nuclei segmentation on {len(dna_files)} image(s) from {input_dir} ...")

    # Build command: script supports --input_dir (no --output_dir)
    # Where the unet script should write masks 
    unet_out = os.path.join(output_dir, "DNA_masks")
    os.makedirs(unet_out, exist_ok=True)

    # Fast-skip if we already have any .tif masks in the target folder
    if any(str(p).lower().endswith(".tif") for p in Path(unet_out).glob("*.tif*")):
        _log(f"unet nuclei masks already exist in {unet_out}. Skipping unet segmentation.")
        return

    # For backward-compat within this function, treat this as the mask dir
    masks_dir = unet_out

    cmd = [
        "python", nuclei_unet_script,
        "--input_dir", input_dir,
        "--output_dir", unet_out,
    ]
    if model_path:
        cmd += ["--model_path", model_path]
    if extra_args:
        cmd += list(extra_args)

    try:
        subprocess.run(cmd, check=True)
        _log("unet directory-mode inference finished.")
    except subprocess.CalledProcessError as e:
        _log(f"ERROR running nuclei unet: {e}. See script output above.")
        return


def qc_and_stage_masks(
    mask_dir: str,
    output_dir: str,
    # QC thresholds for assessing mask quality
    qc_max_coverage_frac: float = 0.35,        # “fills image” if coverage exceeds this
    qc_min_median_solidity: float = 0.85,      # low median solidity suggests many holes
    qc_max_low_solidity_slice_frac: float = 0.50,  # too many slices with solidity < 0.80
    qc_low_solidity_slice_cutoff: float = 0.80,
    qc_max_border_touch_frac: float = 0.60,    # mask heavily hugging image borders
    log_file: str = None,
):
    """
    QC pass for unet masks (no filtering). Stages masks for downstream steps and reports warnings.

    Reads *.tif masks from `mask_dir`, computes QC metrics to detect failure modes (full-image fill,
    many holes), prints warnings, writes a CSV report and an exclude list, and copies masks into
    `output_dir/segmented_nuclei/` using legacy-compatible filenames.

    Args:
        mask_dir (str): Directory containing unet-produced label masks (*.tif).
        output_dir (str): Base output directory (creates/uses 'segmented_nuclei' subdir).
        qc_* (floats): Thresholds for coverage, solidity, and border-touch heuristics.
        log_file (str, optional): Append progress/warnings here in addition to stdout.
    """
    def _log(msg: str):
        print(msg)
        if log_file:
            try:
                with open(log_file, "a") as f:
                    f.write(msg.rstrip() + "\n")
            except Exception:
                pass

    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
    if not mask_files:
        _log(f"No mask files found in {mask_dir}. Skipping QC.")
        return

    report_path = os.path.join(output_dir, "qc_report.csv")
    exclude_list_path = os.path.join(output_dir, "qc_exclude.txt")

    with open(report_path, "w", newline="") as csvfile, open(exclude_list_path, "w") as excl:
        writer = csv.writer(csvfile)
        writer.writerow([
            "image_basename",
            "n_labels",
            "coverage_frac",
            "median_slice_solidity",
            "low_solidity_slice_frac",
            "border_touch_frac",
            "qc_status",
            "qc_reasons",
        ])

        for mask_file in mask_files:
            try:
                subfolder = get_cached_subfolder_name(mask_file)
            except Exception:
                # fallback: use filename stem if cache miss
                subfolder = Path(mask_file).stem

            try:
                _, label_image, _ = universal_read_tiff(mask_file)
            except Exception as e:
                _log(f"ERROR reading {mask_file}: {e}")
                continue

            # Normalize dims: expect 3D (Z, Y, X). If 4D, squeeze singleton; error otherwise.
            if label_image.ndim == 4:
                label_image = np.squeeze(label_image)
            if label_image.ndim != 3:
                _log(f"Unexpected dimensionality for {mask_file}: {label_image.ndim}D. Skipping.")
                continue

            # Basic stats
            fg = label_image > 0
            total_vox = fg.size
            fg_vox = int(fg.sum())
            coverage = (fg_vox / total_vox) if total_vox > 0 else 0.0

            labels = np.unique(label_image)
            n_labels = int(len(labels) - (1 if 0 in labels else 0))

            # Border-touch fraction: fraction of foreground voxels that lie on any image border
            zmax, ymax, xmax = np.array(label_image.shape) - 1
            # build a border mask once
            border_mask = np.zeros_like(fg, dtype=bool)
            border_mask[0, :, :]  = True
            border_mask[zmax, :, :] = True
            border_mask[:, 0, :]  = True
            border_mask[:, ymax, :] = True
            border_mask[:, :, 0]  = True
            border_mask[:, :, xmax] = True
            border_touch_frac = (fg & border_mask).sum() / fg_vox if fg_vox > 0 else 0.0

            # Per-slice convex-hull solidity to catch "many holes"/ragged interiors
            # (robust & fast; avoids 3D convex hull complexity)
            slice_solidities = []
            for z in range(label_image.shape[0]):
                sl = fg[z]
                if not sl.any():
                    continue
                hull = convex_hull_image(sl)
                # guard against degenerate hull
                hcount = int(hull.sum())
                scount = int(sl.sum())
                if hcount > 0:
                    slice_solidities.append(scount / hcount)
            if len(slice_solidities) == 0:
                median_sol = 0.0
                low_sol_frac = 0.0
            else:
                arr = np.array(slice_solidities, dtype=float)
                median_sol = float(np.median(arr))
                low_sol_frac = float((arr < qc_low_solidity_slice_cutoff).mean())

            # Rule-based QC
            reasons = []
            if coverage >= qc_max_coverage_frac:
                reasons.append(f"High coverage ({coverage:.3f} ≥ {qc_max_coverage_frac})")
            if border_touch_frac >= qc_max_border_touch_frac:
                reasons.append(f"Border-hugging ({border_touch_frac:.3f} ≥ {qc_max_border_touch_frac})")
            if median_sol < qc_min_median_solidity:
                reasons.append(f"Low median solidity ({median_sol:.3f} < {qc_min_median_solidity})")
            if low_sol_frac > qc_max_low_solidity_slice_frac:
                reasons.append(f"Many low-solidity slices ({low_sol_frac:.3f} > {qc_max_low_solidity_slice_frac})")

            status = "FAIL" if reasons else "PASS"

            # Log warnings
            base = os.path.basename(mask_file)
            if status == "FAIL":
                _log(f"[QC WARNING] {base}: " + "; ".join(reasons))

            # Write CSV row
            writer.writerow([
                base,
                n_labels,
                f"{coverage:.6f}",
                f"{median_sol:.6f}",
                f"{low_sol_frac:.6f}",
                f"{border_touch_frac:.6f}",
                status,
                "|".join(reasons),
            ])
            if status == "FAIL":
                excl.write(Path(base).stem + "\n")

    _log(f"QC report written to: {report_path}")
    _log(f"Exclude list written to: {exclude_list_path}")


def find_matching_masks(raw_dir, mask_dir):
    """
    Match raw images with corresponding segmented masks.

    Args:
        raw_dir (str): Directory containing raw images.
        mask_dir (str): Directory containing mask images.

    Returns:
        list of tuples: Each tuple contains a raw image path and its matching mask path.
    """
    raw_images = glob.glob(os.path.join(raw_dir, "*.tif"))
    mask_images = glob.glob(os.path.join(mask_dir, "*.tif"))

    matches = []
    for raw_image in raw_images:
        # Extract the basename of the raw image (remove first 3 characters and .tif extension)
        raw_basename = os.path.basename(raw_image)[3:].replace(".tif", "")

        # Find mask images that contain this basename
        for mask_image in mask_images:
            if raw_basename in os.path.basename(mask_image):
                matches.append((raw_image, mask_image))
                break  # Stop searching once a match is found

    return matches


def process_raw_and_masks(raw_dir, mask_dir, output_dir):
    """

    Combine matched raw images and segmentation masks into a two-channel OME-TIFF stack.

    Physical pixel sizes are obtained from cached metadata or from the raw image

    file when available. If metadata cannot be determined, a default voxel size

    of (1.0, 1.0, 1.0) is used.

    """
    matches = find_matching_masks(raw_dir, mask_dir)

    if not matches:
        print("No matching raw-mask pairs found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for raw_image, mask_image in matches:
        output_file = os.path.join(
            output_dir,
            f"{os.path.basename(raw_image)}_segmentation_overlay.tif"
        )

        if os.path.exists(output_file):
            print(f"Skipping {output_file}, already exists.")
            continue

        print(f"Processing pair: {raw_image} <-> {mask_image}")

        try:
            # -----------------------------
            # Load raw and mask images
            # -----------------------------
            raw = io.imread(raw_image)
            mask = io.imread(mask_image)

            if raw.shape != mask.shape:
                print(f"Dimension mismatch: {raw_image} and {mask_image}. Skipping.")
                continue

            if raw.dtype not in (np.uint16, np.float32):
                raw = raw.astype(np.uint16)

            if mask.dtype not in (np.uint16, np.float32):
                mask = mask.astype(np.uint16)

            # -----------------------------
            # Resolve metadata
            # -----------------------------
            # Attempt to determine a cache key based on filename
            subfolder = os.path.splitext(os.path.basename(raw_image))[0]

            if subfolder in metadata_cache:
                # Preferred: reuse cached metadata
                metadata = metadata_cache[subfolder]
                print(f"Using cached metadata for {subfolder}")

            else:
                # Try to extract metadata from the raw image file
                try:
                    img = AICSImage(raw_image)
                    voxel_size = img.physical_pixel_sizes  # (Z, Y, X)
                    metadata = {"physical_pixel_sizes": voxel_size}
                    print(f"Read metadata from raw image: voxel size = {voxel_size}")

                except Exception as e:
                    # Absolute fallback: safe default
                    metadata = {"physical_pixel_sizes": (1.0, 1.0, 1.0)}
                    print(
                        f"Warning: could not read metadata from {raw_image}. "
                        f"Using default voxel size (1.0, 1.0, 1.0). Error: {e}"
                    )

                # Cache it so subsequent calls are consistent
                metadata_cache[subfolder] = metadata

            # -----------------------------
            # Combine and save
            # -----------------------------
            # Output shape: (C, Z, Y, X)
            combined = np.stack((raw, mask), axis=0)

            save_as_ome_tiff(
                combined,
                output_file,
                metadata=metadata
            )

            print(f"Saved combined image to: {output_file}")

        except Exception as e:
            print(f"Error processing {raw_image} and {mask_image}: {e}")



def process_pipeline(config, steps_to_run=None):
    """
    Process the full pipeline in the following order:
    1. Split channels.
    2. Optionally perform primary background subtraction.
    3. Run nuclei unet segmentation.
    4. QC nuclei masks 
    5. Generate segmentation overlays.

    """
    output_dir = config["output_directory"]
    log_file = os.path.join(output_dir, "pipeline_log.txt")
    os.makedirs(output_dir, exist_ok=True)

    split_channels_dir = os.path.join(output_dir, "split_channels")
    bg_subtracted_dir = os.path.join(output_dir, "bg_subtracted_channels")
    seg_overlay_dir = os.path.join(output_dir, "segmentation_overlay")
    mask_dir = os.path.join(output_dir, "DNA_masks")
    os.makedirs(mask_dir, exist_ok=True)

    use_primary_bg_subtraction = config.get("use_background_subtraction", False)
    background_image_path = config.get("background_image_path", None)

    # Load dark image if primary background subtraction is enabled
    dark_image = None
    if use_primary_bg_subtraction:
        try:
            _, dark_image_stack, _ = universal_read_tiff(background_image_path, log_file=log_file)
            dark_image = dark_image_stack[0, 0, :, :]  # 2D slice
            log(f"Loaded dark image from {background_image_path}.", log_file)
        except Exception as e:
            log(f"Error loading background image: {e}", log_file)
            raise


    # Channel name for DNA
    channel_name = next(name for name, index in config["channels"].items()
                        if index == config["channels"]["DNA"])


    steps = {
        "split_channels": lambda: [
            split_channels(
                result["stack"],
                result["metadata"],
                config["channels"],
                split_channels_dir,
                result["subfolder"],
                log_file,
            )
            for result in combined_reading(config["input_directory"], log_file)
        ],

        "background_subtraction": lambda: [
            subtract_background(
                split_channel_image=os.path.join(
                    split_channels_dir, f"{channel_name}", f"C{config['channels'][channel_name]}-{result['subfolder']}.tif"
                ),
                dark_image=dark_image,
                channels=config["channels"],
                channel_name=channel_name,
                output_dir=bg_subtracted_dir,
                subfolder=result["subfolder"],
                metadata=result["metadata"],
                log_file=log_file,
            )
            for result in combined_reading(config["input_directory"], log_file)
            for channel_name in config["channels"].keys()
        ] if use_primary_bg_subtraction else None,

        # unet segmentation 
        "nuclei_unet_segmentation": lambda: run_nuclei_unet(
            nuclei_unet_script=config["nuclei_unet_script"],
            split_channels_dir=split_channels_dir,
            output_dir=output_dir,
            filtered_dna_dir=os.path.join(bg_subtracted_dir, "DNA"),
            use_bg_subtraction=use_primary_bg_subtraction,
            log_file=log_file,
            model_path=config.get("nuclei_unet_model"),
            extra_args=config.get("nuclei_unet_extra_args", []),
        ),

        # QC + staging replaces mask filtering
        "qc_nuclei_masks": lambda: qc_and_stage_masks(
            mask_dir=mask_dir,
            output_dir=output_dir,
            log_file=log_file,
        ),

        "generate_segmentation_overlay": lambda: process_raw_and_masks(
            os.path.join(bg_subtracted_dir if use_primary_bg_subtraction else split_channels_dir, "DNA"),
            mask_dir,
            seg_overlay_dir
        ),

    }

    if steps_to_run is None:
        steps_to_run = steps.keys()

    log(f"Starting pipeline...", log_file)
    for step, func in steps.items():
        if step in steps_to_run:
            log(f"Executing step: {step}", log_file)
            try:
                func()
            except Exception as e:
                log(f"Error during step {step}: {e}", log_file)
        else:
            log(f"Skipping step: {step}", log_file)

    log("Pipeline completed successfully.", log_file)

def main_with_full_pipeline(config_path, steps=None):
    """
    Main function to run the pipeline with optional steps.

    Args:
        config_path (str): Path to the JSON configuration file.
        steps (list, optional): List of steps to execute. If None, all steps are executed.
    """
    config = load_config(config_path)
    process_pipeline(config, steps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full image processing pipeline.")
    parser.add_argument("config", help="Path to the JSON configuration file.")
    parser.add_argument(
        "--steps",
        help="Comma-separated list of steps to execute (e.g., max_projection,split_channels).",
    )
    args = parser.parse_args()

    steps_to_run = args.steps.split(",") if args.steps else None
    main_with_full_pipeline(args.config, steps=steps_to_run)