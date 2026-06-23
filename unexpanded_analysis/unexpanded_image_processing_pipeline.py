
"""
This script processes microscopy datasets of zebrafish embryos and performs
image preprocessing, nuclei segmentation, intensity quantification, statistical
analysis, and summary output generation.

The pipeline coordinates multiple analysis stages through a configuration file,
including image filtering, background subtraction, nuclei segmentation,
intensity measurements, pairwise channel correlation analysis, statistical
testing, and automated plotting. Intermediate and final results are written
to structured output directories for downstream analysis and visualization. 


Input naming convention
-----------------------
The pipeline supports either:
1. A directory containing one subdirectory per image, or
2. A flat directory containing image files directly.
In both cases, the folder name or image filename stem is used as the
unique image identifier throughout the analysis workflow. This identifier
is propagated into split-channel images, segmentation masks,
quantification CSVs, correlation outputs, and plotting results.
Folder names or image filenames should follow a consistent
underscore-delimited format and include both condition information and
a unique embryo identifier. For example:
    260616_WT_H3K9me3-ab5-Alexa647_DAPI_6hpf_60x_E_1
    260616_flavopiridol_H3K9me3-ab5-Alexa647_DAPI_6hpf_60x_E_4
The plotting and grouping logic assumes that biological conditions are
encoded consistently within folder names and that embryo replicates are
identified using the suffix E_<number>. Inconsistent naming may result in
incorrect grouping of samples or failure to match intermediate analysis
files.

"""

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
import sys
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from datetime import datetime
import pandas as pd
from skimage.measure import regionprops, label
from aicsimageio import AICSImage
import napari
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import re
import cv2
from scipy.stats import pearsonr
import math
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from scipy.signal import correlate


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
    Display an image in Napari for optional interactive inspection.
    This function is not used during standard pipeline execution.

    Args:
        image (np.ndarray): The image data to display.
        pixel_sizes (tuple or list, optional): Physical pixel sizes (Z, Y, X) for scaling. Default is None.
        title (str): The title of the Napari viewer window. Default is "Image Viewer".
    """
    import napari

    # Create a Napari viewer
    viewer = napari.Viewer(title=title)

    # Add the image to the viewer
    if pixel_sizes and len(pixel_sizes) == 3:
        viewer.add_image(
            image,
            name="Image",
            scale=pixel_sizes,  # Set physical pixel sizes for proper scaling
        )
    else:
        viewer.add_image(image, name="Image")

    # Start the Napari event loop
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



def write_verified_run_log(output_dir, config, steps_to_run=None, preset=None, log_file=None):
    """
    Write a human-readable run record to the main output directory.

    This pipeline is often used to generate data that will appear in figures and downstream
    statistical comparisons. To support reproducibility and transparency, each run writes
    a unique run log file that records:

      - Date/time
      - Script name/path
      - Selected steps (or preset)
      - The full configuration (pretty-printed JSON)

    Args:
        output_dir (str): Main output directory for the pipeline.
        config (dict): Configuration dictionary used for this run.
        steps_to_run (list or None): The steps executed (if provided).
        preset (str or None): Optional preset name used (if any).
        log_file (str or None): Optional log file for recording the manifest creation.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    manifest_path = os.path.join(output_dir, f"verified_run_log_{timestamp}.txt")

    # Attempt to record the executing script path 
    script_path = None
    try:
        script_path = os.path.abspath(__file__)
    except Exception:
        script_path = "<unknown>"

    lines = []
    lines.append("=" * 70)
    lines.append("VERIFIED PIPELINE RUN LOG")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Pipeline script:")
    lines.append(f"  Path: {script_path}")
    lines.append(f"  Name: {os.path.basename(script_path) if script_path else '<unknown>'}")
    lines.append("")
    if preset:
        lines.append(f"Preset: {preset}")
    if steps_to_run is not None:
        lines.append("Steps executed:")
        for s in steps_to_run:
            lines.append(f"  - {s}")
    lines.append("")
    lines.append("Output directory:")
    lines.append(f"  {os.path.abspath(output_dir)}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("CONFIG USED")
    lines.append("-" * 70)
    try:
        config_str = json.dumps(config, indent=2, sort_keys=True)
    except Exception:
        config_str = str(config)
    lines.append(config_str)
    lines.append("")
    lines.append("=" * 70)

    try:
        with open(manifest_path, "w") as f:
            f.write("\n".join(lines))
        log(f"Created verified run log: {manifest_path}", log_file)
    except Exception as e:
        log(f"Failed to write verified run log: {e}", log_file)

    return manifest_path

# Global dictionary for metadata caching
metadata_cache = {}
"""dict: Caches image metadata (e.g., voxel sizes) by subfolder or filename key to avoid redundant I/O.

This is used throughout the pipeline to store and retrieve metadata—such as
voxel size or channel info—from raw image files. Functions that process
images (e.g., filtering, saving, segmentation) first check this cache to
reuse metadata instead of re-reading image headers.

Keys are typically subfolder names or base filenames.
"""

# ---------------------------------------------------------------------
# Persistent on-disk metadata caching
# ---------------------------------------------------------------------
METADATA_CACHE_DIR = None
"""str or None: Directory where per-image metadata JSON files are cached.


The cache key is `image_id`, which should match the identifier used throughout the pipeline
(e.g., subfolder name in subfolder mode, or filename stem in flat mode).
"""

def _sanitize_for_json(obj):
    """Convert common metadata objects to JSON-serializable types."""
    if obj is None:
        return None
    # AICSImageIO physical_pixel_sizes may be a namedtuple-like object.
    try:
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
            return [float(x) for x in obj]
    except Exception:
        pass
    # Fall back to string representation (last resort)
    return str(obj)

def _metadata_cache_path(image_id, cache_dir):
    """Return the on-disk metadata cache path for a given image_id."""
    safe_id = str(image_id).replace(os.sep, "_")
    return os.path.join(cache_dir, f"{safe_id}_metadata.json")

def write_metadata_cache_to_disk(image_id, metadata, cache_dir, log_file=None):
    """Write metadata to disk in a small JSON file for persistent reuse."""
    if cache_dir is None:
        return
    os.makedirs(cache_dir, exist_ok=True)

    payload = {
        "image_id": str(image_id),
        "cached_at": datetime.now().isoformat(timespec="seconds"),
        "physical_pixel_sizes": _sanitize_for_json(metadata.get("physical_pixel_sizes")),
        "channel_names": metadata.get("channel_names"),
        # Store any additional fields verbatim if already JSON-friendly
        **{k: v for k, v in metadata.items() if k not in ("physical_pixel_sizes", "channel_names")},
    }

    path = _metadata_cache_path(image_id, cache_dir)
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        log(f"Wrote metadata cache: {path}", log_file)
    except Exception as e:
        log(f"WARNING: Failed to write metadata cache for {image_id}: {e}", log_file)

def read_metadata_cache_from_disk(image_id, cache_dir, log_file=None):
    """Read metadata cache JSON from disk. Returns dict or None if missing."""
    if cache_dir is None:
        return None
    path = _metadata_cache_path(image_id, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        log(f"Loaded metadata cache: {path}", log_file)
        return {
            "physical_pixel_sizes": payload.get("physical_pixel_sizes"),
            "channel_names": payload.get("channel_names"),
            **{k: v for k, v in payload.items() if k not in ("image_id", "cached_at", "physical_pixel_sizes", "channel_names")},
        }
    except Exception as e:
        log(f"WARNING: Failed to read metadata cache for {image_id}: {e}", log_file)
        return None

def read_metadata_only(file_path, log_file=None):
    """Read metadata from a raw image without loading the full image stack."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found or invalid: {file_path}")
    img = AICSImage(file_path)
    return {
        "physical_pixel_sizes": _sanitize_for_json(img.physical_pixel_sizes),
        "channel_names": img.channel_names,
    }

def get_or_create_metadata(image_id, raw_image_path, log_file=None):
    """Return metadata for an image_id, using in-memory and on-disk caches when available.

    This helper guarantees that metadata can be retrieved even in partial reruns. If the metadata
    is not in memory and no on-disk cache is present, the metadata is extracted from the raw image
    and cached to disk (if METADATA_CACHE_DIR is set).

    Args:
        image_id (str): Pipeline image identifier (subfolder name or filename stem).
        raw_image_path (str): Path to the raw image file for metadata extraction.
        log_file (str, optional): Log file path.

    Returns:
        dict: Metadata dict including `physical_pixel_sizes` and `channel_names`.
    """
    # 1) In-memory cache
    if image_id in metadata_cache:
        return metadata_cache[image_id]

    # 2) On-disk cache
    disk = read_metadata_cache_from_disk(image_id, METADATA_CACHE_DIR, log_file=log_file)
    if disk is not None:
        metadata_cache[image_id] = disk
        return disk

    # 3) Extract from raw image and cache
    extracted = read_metadata_only(raw_image_path, log_file=log_file)
    metadata_cache[image_id] = extracted
    write_metadata_cache_to_disk(image_id, extracted, METADATA_CACHE_DIR, log_file=log_file)
    return extracted


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

def combined_reading(input_dir, log_file=None, no_subfolders=False):
    """
    Read TIFF files from input_dir and determine whether to use single or multi-file reading logic.

    Modes
    -----
    1) Subfolder mode (default; no_subfolders=False):
       - input_dir contains subfolders
       - each subfolder contains one-or-more *.ome.tif files (possibly multi-file series)
       - key used for output naming + metadata caching is the subfolder name

    2) Flat mode (no_subfolders=True):
       - input_dir contains TIFFs directly (no subfolders)
       - supports: *.ome.tif, *.tif, *.tiff
       - each file is treated as one logical image item
       - key used for output naming + metadata caching is the filename stem, with a small cleanup:
         e.g. "sample.ome.tif" -> image_id "sample"

    Yields:
        dict: {"subfolder": <image_id>, "stack": combined_stack, "metadata": metadata}
            
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found or invalid: {input_dir}")

    # -------------------------
    # Flat directory mode
    # -------------------------
    if no_subfolders:
        # Accept .ome.tif, .tif, .tiff (case-insensitive-ish via multiple globs)
        patterns = ["*.ome.tif", "*.tif", "*.tiff", "*.OME.TIF", "*.TIF", "*.TIFF"]
        tiff_files = []
        for pat in patterns:
            tiff_files.extend(glob.glob(os.path.join(input_dir, pat)))

        # De-duplicate but preserve sorting
        tiff_files = sorted(set(os.path.abspath(p) for p in tiff_files))

        if not tiff_files:
            log(f"No TIFF files found in flat input directory: {input_dir}", log_file)
            return

        for file_path in tiff_files:
            # image_id is basename without extension(s)
            base = os.path.basename(file_path)

            # Handle .ome.tif specifically: splitext twice turns "x.ome.tif" -> "x.ome" -> "x"
            stem1, ext1 = os.path.splitext(base)          # removes .tif/.tiff
            stem2, ext2 = os.path.splitext(stem1)         # removes .ome if present
            image_id = stem2 if ext2.lower() == ".ome" else stem1

            # If there's any weird trailing dot, clean it
            image_id = image_id.rstrip(".")

            try:
                img, combined_stack, metadata = universal_read_tiff(file_path, log_file)
                yield {
                    "subfolder": image_id,
                    "stack": combined_stack,
                    "metadata": metadata,
                }
            except Exception as e:
                log(f"Error processing flat TIFF {file_path}: {e}", log_file)

        return

    # -------------------------
    # Subfolder mode
    # -------------------------
    subfolders = [
        subfolder for subfolder in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, subfolder))
    ]

    for subfolder in subfolders:
        subfolder_path = os.path.join(input_dir, subfolder)

        # Original behavior: only *.ome.tif in subfolders 
        tiff_files = sorted(glob.glob(os.path.join(subfolder_path, "*.ome.tif")))

        if not tiff_files:
            log(f"No OME-TIFF files found in subfolder: {subfolder_path}", log_file)
            continue

        try:
            if len(tiff_files) > 1:
                log(f"Multiple TIFF files found in {subfolder_path}. Combining files.", log_file)
                combined_stack, metadata = universal_read_tiff_multi(tiff_files, log_file)
            else:
                img, combined_stack, metadata = universal_read_tiff(tiff_files[0], log_file)

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


def save_max_projection(max_proj_stack, output_file, metadata):
    """
    Save a max-projected NumPy array (CYX) as an OME-TIFF file with metadata.

    Args:
        max_proj_stack (np.ndarray): The max-projected image data to save (CYX format).
        output_file (str): Path to save the new OME-TIFF file.
        metadata (dict): Metadata dictionary containing relevant information.
    """
    try:
        # Extract voxel sizes (physical pixel sizes)
        physical_pixel_sizes = metadata.get("physical_pixel_sizes", (1.0, 1.0, 1.0))
        pixel_size_x = physical_pixel_sizes[1]  # X spacing
        pixel_size_y = physical_pixel_sizes[2]  # Y spacing
        pixel_size_z = physical_pixel_sizes[0]  # Retain original Z spacing in metadata

        # Calculate resolution in dots per micrometer for X and Y
        resolution_x = 1 / pixel_size_x  # In microns
        resolution_y = 1 / pixel_size_y  # In microns

        # Add a singleton Z-axis for consistency with OME-TIFF format
        output_stack = np.expand_dims(max_proj_stack, axis=1)  # Add Z-axis: (C, 1, Y, X)

        # Reorder axes for ImageJ compatibility: (Z, C, Y, X)
        output_stack = np.transpose(output_stack, (1, 0, 2, 3))

        # Define true colors for LUTs
        true_colors = [
            [0, 255, 255],  # Cyan for channel 1
            [255, 0, 255],  # Magenta for channel 2
            [255, 255, 0],  # Yellow for channel 3
            [128, 128, 128],  # Gray for channel 4
        ]

        # Determine number of channels
        num_channels = output_stack.shape[1]

        # Generate LUTs as gradient arrays for each channel
        luts = []
        for i in range(num_channels):
            # For single-channel images, assign gray LUT by default
            if num_channels == 1:
                lut = np.array([
                    np.linspace(0, 255, 256),  # Red gradient for gray
                    np.linspace(0, 255, 256),  # Green gradient for gray
                    np.linspace(0, 255, 256),  # Blue gradient for gray
                ], dtype=np.uint8)
            else:
                r, g, b = true_colors[i % len(true_colors)]  # Cycle through colors if > 4 channels
                if [r, g, b] == [128, 128, 128]:  # Generate correct LUT for gray
                    lut = np.array([
                        np.linspace(0, 255, 256),  # Red gradient for gray
                        np.linspace(0, 255, 256),  # Green gradient for gray
                        np.linspace(0, 255, 256),  # Blue gradient for gray
                    ], dtype=np.uint8)
                else:
                    lut = np.array([
                        np.linspace(0, r, 256),  # Red gradient
                        np.linspace(0, g, 256),  # Green gradient
                        np.linspace(0, b, 256),  # Blue gradient
                    ], dtype=np.uint8)

            luts.append(lut)


        # ImageJ-specific metadata
        imagej_metadata = {
            "spacing": pixel_size_z,  # Retain Z voxel size for consistency
            "unit": "micron",         # Unit for all dimensions
            "mode": "composite",      # Open in composite mode
            "LUTs": luts,             # Assign LUTs to channels
        }

        # Write the TIFF file with proper ImageJ metadata
        tiff.imwrite(
            output_file,
            output_stack,
            photometric="minisblack",
            resolution=(resolution_x, resolution_y),  # Resolution for X and Y
            metadata=imagej_metadata,                # Include Z-spacing, composite mode, and LUTs
            imagej=True,                              # Ensure compatibility with ImageJ
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save max projection OME-TIFF file {output_file}: {e}")

def iter_input_items(input_dir, no_subfolders=False):
    """
    Yield (image_id, file_path) without reading any image pixels.
    """
    if no_subfolders:
        for p in sorted(Path(input_dir).glob("*.tif*")):
            if p.name.endswith(".ome.tif") or p.suffix.lower() in {".tif", ".tiff"}:
                image_id = p.stem.replace(".ome", "")
                yield image_id, str(p)
    else:
        for sub in sorted(Path(input_dir).iterdir()):
            if not sub.is_dir():
                continue
            # pick the first .ome.tif in the subfolder
            files = sorted(sub.glob("*.ome.tif"))
            if not files:
                continue
            yield sub.name, str(files[0])


def max_projection_raw_images(input_dir, output_dir, log_file, no_subfolders=False):
    """
    Perform maximum projection for all raw images in the input directory,
    combining all channels in a single output for each file/item.

    Modes
    -----
    - Subfolder mode (default): input_dir contains subfolders, each with *.ome.tif.
      The subfolder name is used as the image key (for naming + metadata cache).
    - Flat mode (--no_subfolders): input_dir contains TIFFs directly.
      Supports *.ome.tif, *.tif, *.tiff. The filename stem is used as the image key.

    Args:
        input_dir (str): Path to raw input directory.
        output_dir (str): Path to save the maximum projection images.
        log_file (str): Path to the log file for recording progress.
        no_subfolders (bool): If True, read TIFFs directly from input_dir.
    """
    global metadata_cache
    os.makedirs(output_dir, exist_ok=True)

    for result in combined_reading(input_dir, log_file=log_file, no_subfolders=no_subfolders):
        subfolder = result["subfolder"]              # key / image_id
        combined_stack = result["stack"]             # (C, Z, Y, X)
        extracted_metadata = result["metadata"]

        output_path = os.path.join(output_dir, f"{subfolder}_max_proj.tif")

        # Skip processing if the max-projection file already exists
        if os.path.exists(output_path):
            log(f"Skipping max projection for {subfolder} (output exists).", log_file)
            continue

        try:
            # Cache metadata if not already cached
            if subfolder in metadata_cache:
                log(f"Using cached metadata for subfolder: {subfolder}", log_file)
                metadata = metadata_cache[subfolder]
            else:
                metadata_cache[subfolder] = extracted_metadata
                write_metadata_cache_to_disk(subfolder, extracted_metadata, METADATA_CACHE_DIR, log_file=log_file)
                metadata = extracted_metadata
                log(f"Metadata cached for subfolder: {subfolder}", log_file)

            log(f"Projecting item '{subfolder}'", log_file)

            # Max projection along Z for all channels: (C, Z, Y, X) -> (C, Y, X)
            max_projection = np.max(combined_stack, axis=1)

            # Save as a single multichannel file
            save_max_projection(max_projection, output_path, metadata)
            log(f"Saved max projection to: {output_path}", log_file)

        except Exception as e:
            log(f"Error processing item '{subfolder}': {e}", log_file)


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
        write_metadata_cache_to_disk(subfolder, metadata, METADATA_CACHE_DIR, log_file=log_file)
        cached_metadata = metadata

    for channel_name, channel_index in channels.items():
        # Create a subfolder for each channel
        channel_dir = os.path.join(output_dir, channel_name)
        os.makedirs(channel_dir, exist_ok=True)

        # Construct the output path
        output_path = os.path.join(channel_dir, f"C{channel_index}-{subfolder}.tif")

        # Ensure output_path is a string
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

        # Validate that dark_image dimensions match the YX dimensions of the split channel
        if dark_image.shape != split_channel_image.shape[1:]:
            log(f"Dimension mismatch: dark_image {dark_image.shape}, split_channel_image {split_channel_image.shape}", log_file)
            raise ValueError(
                f"Dimension mismatch: dark_image {dark_image.shape} and split_channel_image slices {split_channel_image.shape[1:]} do not align."
            )

        # Log validation success
        log(f"Validated dimensions: dark_image {dark_image.shape}, split_channel_image {split_channel_image.shape}", log_file)

        # Define a function to process a single Z-slice
        def process_slice(z):
            """
            Subtracts the dark image from a single Z-slice and clips the result.

            Args:
                z (int): Index of the Z-slice in the 3D image stack.

            Returns:
                np.ndarray: Background-subtracted and clipped Z-slice as uint16.
            """
            slice_int32 = split_channel_image[z].astype(np.int32)
            dark_int32 = dark_image.astype(np.int32)
            bg_subtracted_slice = slice_int32 - dark_int32
            return np.clip(bg_subtracted_slice, 0, 65535)




        # Process all Z-slices in parallel
        log("Starting multithreaded background subtraction...", log_file)
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            bg_subtracted_stack = list(executor.map(process_slice, range(split_channel_image.shape[0])))

        # Convert to a NumPy array and ensure dtype is uint16
        bg_subtracted_stack = np.array(bg_subtracted_stack, dtype=np.uint16)

        # Save the resulting background-subtracted stack
        log(f"Saving background-subtracted image to: {output_path}", log_file)
        save_as_ome_tiff(bg_subtracted_stack, output_path, metadata)
        log(f"Successfully saved background-subtracted image for {channel_name} to {output_path}.", log_file)

    except Exception as e:
        log(f"Error in background subtraction for {channel_name} in {subfolder}: {e}", log_file)
        raise
        
def normalize_and_filter_dna_channel(split_channels_dir, output_dir, median_radius, gaussian_sigma, log_file):
    """
    Normalize and filter the DNA channel images with multithreading.
    Uses 3D adaptive histogram equalization after applying median and guassian filtering.
    optional for alternative segmentation strategies. with cpsam, this function is not used

    Args:
        split_channels_dir (str): Base directory containing split channels.
        output_dir (str): Directory to save the filtered DNA images.
        median_radius (int): Radius for the median filter.
        gaussian_sigma (float): Sigma for the Gaussian blur filter.
        log_file (str): Path to a log file where the process is logged.
    """

    global metadata_cache  # Access the global metadata cache
    print("Step 3: Normalizing and filtering DNA channel...")

    n_workers = multiprocessing.cpu_count()

    gaussian_sigma = float(gaussian_sigma)  # Ensures compatibility for int or float input

    # Construct DNA channel directory
    dna_dir = os.path.join(split_channels_dir, "DNA")
    os.makedirs(output_dir, exist_ok=True)

    # Identify all files in the DNA directory
    dna_files = glob.glob(os.path.join(dna_dir, "*.tif"))
    for dna_file in dna_files:
        output_file = os.path.join(output_dir, os.path.basename(dna_file))
        if os.path.exists(output_file):
            log(f"Skipping filtering for {dna_file} (output exists).", log_file)
            continue

        print(f"Processing DNA file: {dna_file}")

        # Retrieve image_id in a way consistent with metadata_cache in BOTH subfolder + flat mode
        subfolder = get_cached_subfolder_name(dna_file)

        # Retrieve cached metadata
        if subfolder in metadata_cache:
            metadata = metadata_cache[subfolder]
            log(f"Using cached metadata for subfolder: {subfolder}", log_file)
        else:
            log(f"Warning: Metadata not found for subfolder {subfolder}. Using default voxel size.", log_file)
            metadata = {"physical_pixel_sizes": (1.0, 1.0, 1.0)}  # Default voxel size as a fallback

        # Read the DNA image using tifffile
        with TiffFile(dna_file) as tif:
            dna_image = tif.asarray()

        # Apply median filter (slice-by-slice for 3D data) with multithreading
        print("Applying median filter with multithreading...")
        def process_median_slice(z_index):
            """
            Apply a median filter to the z-th slice of the DNA image.

            Args:
                z_index (int): Index of the Z-slice.

            Returns:
                np.ndarray: Median filtered 2D slice.
            """
            return median_filter(dna_image[z_index], size=median_radius)

        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            median_filtered_image = np.array(list(executor.map(process_median_slice, range(dna_image.shape[0]))))

        # Apply Gaussian blur (slice-by-slice for 3D data) with multithreading
        print("Applying Gaussian blur with multithreading...")
        def process_gaussian_slice(z_index):
            """
            Apply a gaussian filter to the z-th slice of the DNA image.

            Args:
                z_index (int): Index of the Z-slice.

            Returns:
                np.ndarray: Guassian filtered 2D slice.
            """
            return gaussian_filter(median_filtered_image[z_index], sigma=gaussian_sigma)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            gaussian_filtered_image = np.array(list(executor.map(process_gaussian_slice, range(median_filtered_image.shape[0]))))

        # Apply 3D adaptive histogram equalization
        print("Applying 3D adaptive histogram equalization...")
        normalized_image = exposure.equalize_adapthist(
            gaussian_filtered_image, kernel_size=(150, 150, 20), clip_limit=0.01
        )

        # Save the filtered image using the cached metadata
        log(f"Saving filtered DNA image to {output_file}", log_file)
        save_as_ome_tiff((normalized_image * 65535).astype(np.uint16), output_file, metadata)


def run_cellpose(cellpose_script, split_channels_dir, filtered_dna_dir, use_cpsam, log_file):
    """
    Run Cellpose for segmentation.

    Args:
        cellpose_script (str): Path to the Cellpose bash script.
        split_channels_dir (str): Path to base directory with split channels.
        filtered_dna_dir (str): Path to directory with filtered DNA images.
        use_cpsam (bool): Whether to use CPSAM inputs (skips filtering).
        log_file (str): Path to a log file where the process is logged.
    """
    # Determine input directory based on use_cpsam flag
    input_dir = os.path.join(split_channels_dir, "DNA") if use_cpsam else filtered_dna_dir
    masks_dir = os.path.join(input_dir, "masks")

    if os.path.exists(masks_dir) and os.listdir(masks_dir):
        print("Cellpose masks already exist. Skipping Cellpose segmentation.")
        if log_file:
            with open(log_file, "a") as log:
                log.write("Cellpose masks already exist. Skipping Cellpose segmentation.\n")
        return

    os.makedirs(masks_dir, exist_ok=True)

    # Identify DNA files in the chosen input directory
    dna_files = glob.glob(os.path.join(input_dir, "*.tif"))
    if not dna_files:
        print("No DNA files found for Cellpose. Skipping segmentation.")
        if log_file:
            with open(log_file, "a") as log:
                log.write("No DNA files found for Cellpose. Skipping segmentation.\n")
        return

    print("Running Cellpose for segmentation...")
    subprocess.run([cellpose_script, input_dir, masks_dir], check=True)
    print(f"Cellpose segmentation completed. Results saved in {masks_dir}.")
    if log_file:
        with open(log_file, "a") as log:
            log.write(f"Cellpose segmentation completed. Results saved in {masks_dir}.\n")



def sphericity_filter(input_stack, subfolder, output_dir, sphericity_threshold=0.7, filtering_log_file=None):
    """
    Filter labels based on sphericity and save resulting images, with label ID, sphericity logging,
    and a summary of labels processed and removed.

    Args:
        input_stack (np.ndarray): Label image (3D stack).
        subfolder (str): Subfolder name used to retrieve cached metadata.
        output_dir (str): Path to save the filtered label images.
        sphericity_threshold (float): Minimum sphericity value to retain labels. Default is 0.7.
        log_file (str, optional): Path to a log file where progress is recorded.
    """
    global metadata_cache  # Access the global metadata cache
    os.makedirs(output_dir, exist_ok=True)

    # Output file path
    output_path = os.path.join(output_dir, f"{subfolder}_sphericity_filtered.tif")

    # Check if the output already exists
    if os.path.exists(output_path):
        log(f"Skipping {subfolder} as the output already exists: {output_path}", filtering_log_file)
        print(f"Skipping {subfolder} as the output already exists: {output_path}")
        return

    # Retrieve cached metadata
    if subfolder in metadata_cache:
        metadata = metadata_cache[subfolder]
        log(f"Using cached metadata for subfolder: {subfolder}", filtering_log_file)
    else:
        raise ValueError(f"Metadata not found for subfolder: {subfolder}. Ensure metadata is cached before running this function.")

    # Extract voxel size from metadata
    voxel_size = metadata.get("physical_pixel_sizes", (1.0, 1.0, 1.0))  # Default to (1, 1, 1)
    voxel_volume = np.prod(voxel_size)

    label_image = input_stack

    # Ensure the label image is 3D
    if label_image.ndim != 3:
        raise ValueError(f"Label image must be 3D. Got shape: {label_image.shape}")

    print(f"Label image shape: {label_image.shape}, Voxel size: {voxel_size}")

    # Filter labels based on sphericity
    filtered_image = np.zeros_like(label_image, dtype=np.uint16)  # Create empty output image
    props = measure.regionprops(label_image, spacing=voxel_size)  # Include voxel size for 3D calculations

    total_labels = len(props)
    removed_labels = 0

    def process_label(prop):
        """
        Compute sphericity for a single labeled region and filter it based on a threshold.

        This function performs the following steps:
        - Extracts a 3D binary mask for the given label.
        - Crops to the bounding box of the region for efficiency.
        - Upsamples the cropped region by 2× in all dimensions.
        - Applies Gaussian smoothing to improve surface estimation.
        - Uses marching cubes to estimate the surface area.
        - Computes volume in physical units using voxel size.
        - Calculates sphericity using the formula:
            sphericity = (36 * π * volume²) / (surface_area³)
        - Logs the results and updates the output mask if the sphericity passes the threshold.

        Args:
            prop (skimage.measure._regionprops.RegionProperties): The region properties for the current label.

        Modifies:
            - `filtered_image` (outer scope): Sets the current label in the output if it passes the sphericity threshold.
            - `removed_labels` (outer scope): Increments if the label is removed.

        Raises:
            RuntimeError: If marching cubes or surface area computation fails 
        """
        nonlocal removed_labels
        label_id = prop.label
        region_mask = label_image == label_id  # Boolean mask where True corresponds to the current label

        # Focus on the bounding box of the label for efficiency
        bbox = prop.bbox
        region_cropped = region_mask[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]

        # Upsample the cropped region by 2x for finer resolution
        upsampled_region = resize(region_cropped.astype(float), 
                                  (region_cropped.shape[0] * 2, 
                                   region_cropped.shape[1] * 2, 
                                   region_cropped.shape[2] * 2), 
                                  mode='reflect', anti_aliasing=True)

        # Smooth the upsampled region with a Gaussian filter for more accurate surface area calculation
        smoothed_region = gaussian_filter(upsampled_region, sigma=0.5)

        # Apply marching cubes on the smoothed region
        verts, faces, _, _ = measure.marching_cubes(smoothed_region, spacing=np.array(voxel_size) / 2)
        surface_area = measure.mesh_surface_area(verts, faces)

        # Calculate volume based on the number of voxels
        volume = np.sum(region_cropped) * voxel_volume  # Convert to physical volume

        # Calculate sphericity: (36 * π * Volume^2) / (Surface Area^3)
        if surface_area > 0:
            sphericity = (36 * np.pi * volume**2) / (surface_area**3)
        else:
            sphericity = 0  # Invalid sphericity if no surface area

        # Log and print the label ID and sphericity
        log_message = f"Label ID {label_id}: Sphericity = {sphericity:.4f}, voxels = {np.sum(region_cropped)}, Volume = {volume}, Surface Area = {surface_area}\n"
        print(log_message.strip())
        with open(filtering_log_file, "a") as filtering_log_handle:
            filtering_log_handle.write(log_message)

        # Retain label if sphericity is above threshold
        if sphericity >= sphericity_threshold:
            filtered_image[region_mask] = label_id
        else:
            removed_labels += 1


    # Use multithreading to process labels in parallel
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(process_label, props)

    # Save the filtered image
    save_as_ome_tiff(filtered_image, output_path, metadata)
    save_message = f"Processed {subfolder}: saved to {output_path}\n"
    print(save_message.strip())
    if filtering_log_file:
        with open(filtering_log_file, "a") as log_file_handle:
            log_file_handle.write(save_message)

    # Print and log summary
    if total_labels > 0:
        percent_removed = (removed_labels / total_labels) * 100
    else:
        percent_removed = 0.0
    summary_message = (
        f"Summary for {subfolder}: {total_labels} labels processed, "
        f"{removed_labels} labels removed ({percent_removed:.2f}% of total labels).\n"
    )
    print(summary_message.strip())
    if filtering_log_file:
        with open(filtering_log_file, "a") as log_file_handle:
            log_file_handle.write(summary_message)



def filter_masks(
    mask_dir, output_dir, voxel_threshold=25000, sphericity_threshold=0.7, log_file=None
):
    """
    Filter Cellpose masks by voxel count, apply optional Z-score intensity filtering, sphericity filtering, and save them.

    Args:
        mask_dir (str): Path to the directory containing the Cellpose 'masks' subfolder.
        output_dir (str): Base output directory where the 'segmented_nuclei' folder will be created.
        voxel_threshold (int): Minimum number of voxels required to include a label in the mask.
        sphericity_threshold (float): Minimum sphericity value to retain labels.
        log_file (str, optional): Path to a log file where progress is recorded.
    """
    print("Step 6: Filtering Cellpose masks by voxel size, Z-score intensity, and sphericity...")

    # Define paths
    masks_dir = mask_dir
    segmented_nuclei_dir = os.path.join(output_dir, "segmented_nuclei")
    os.makedirs(segmented_nuclei_dir, exist_ok=True)

    # Locate all mask files in the masks directory
    mask_files = glob.glob(os.path.join(masks_dir, "*.tif"))
    if not mask_files:
        print(f"No mask files found in {masks_dir}. Skipping mask filtering step.")
        return

    print(f"Located mask files: {mask_files}")

    for mask_file in mask_files:
        # Get the subfolder name for cached metadata
        subfolder = get_cached_subfolder_name(mask_file)

        # Define the output file path for the sphericity-filtered results
        sphericity_output_path = os.path.join(segmented_nuclei_dir, f"{subfolder}_sphericity_filtered.tif")

        # Check if the sphericity-filtered output already exists
        if os.path.exists(sphericity_output_path):
            print(f"Sphericity-filtered output already exists for {mask_file}. Skipping.")
            continue

        print(f"Processing mask file: {mask_file}")

        # Read the mask
        try:
            _, label_image, _ = universal_read_tiff(mask_file)
        except Exception as e:
            print(f"Error reading mask file {mask_file}: {e}")
            continue

        # Ensure the mask is 3D
        if label_image.ndim == 4:
            print(f"4D image detected for {mask_file}, squeezing dimensions...")
            label_image = np.squeeze(label_image)
        elif label_image.ndim != 3:
            raise ValueError(f"Unexpected dimensionality ({label_image.ndim}D) for {mask_file}. Only 3D images are supported.")

        # Ensure the mask is not empty
        if label_image.max() == 0:
            print(f"No labels found in {os.path.basename(mask_file)}.")
            continue

        # Relabel connected components in 3D
        #relabeled_image, num_features = nd_label(label_image > 0, structure=np.ones((3, 3, 3)))
        relabeled_image = label_image

        # Define the filtering log file
        filtering_log_file = os.path.join(output_dir, "filtering_log.txt")

        # Create a new filtered mask with the same shape as the input
        size_filtered_mask = np.zeros_like(relabeled_image, dtype=np.uint16)

        # Use regionprops to calculate properties for each label
        props = measure.regionprops(relabeled_image)

        # Filter labels by voxel count
        for prop in props:
            label_id = prop.label
            voxel_count = prop.area  # Number of voxels in this label

            # Retain the label if it meets the threshold
            if voxel_count >= voxel_threshold:
                size_filtered_mask[relabeled_image == label_id] = label_id
            else:
                log_message = f"Label ID {label_id} excluded (voxel count = {voxel_count}).\n"
                print(log_message.strip())
                with open(filtering_log_file, "a") as filtering_log_handle:
                    filtering_log_handle.write(log_message)

        # Call the sphericity filter
        log_message = f"Calling sphericity filter for subfolder: {subfolder}\n"
        print(log_message.strip())
        with open(filtering_log_file, "a") as filtering_log_handle:
            filtering_log_handle.write(log_message)
        sphericity_filter(size_filtered_mask, subfolder, segmented_nuclei_dir, sphericity_threshold, filtering_log_file)

        log_message = f"Filtered masks saved to: {segmented_nuclei_dir}\n"
        print(log_message.strip())
        with open(filtering_log_file, "a") as filtering_log_handle:
            filtering_log_handle.write(log_message)

    print(f"Filtered masks saved to: {segmented_nuclei_dir}")


def find_two_clean_rois(seg_mid, bbox, roi_size):
    """
    Finds up to two candidate ROIs within the provided bounding box that have no nuclei (seg_mid==0),
    do not overlap each other, and whose centers are at least 100 pixels apart.
    
    Args:
        seg_mid (np.ndarray): 2D segmentation mask (midplane of segmentation).
        bbox (tuple): A tuple (y_min, y_max, x_min, x_max) specifying the allowed area.
        roi_size (tuple): The desired ROI size (height, width). The area is assumed to be roi_size[0]*roi_size[1].
    
    Returns:
        list: A list of up to two tuples representing candidate ROIs 
              as (y1, x1, y2, x2), where (y1, x1) is top-left and (y2, x2) is bottom-right.
              Only non-overlapping candidate pairs whose centers are at least 100 pixels apart are returned.
    """
    
    candidates = []
    y_min, y_max, x_min, x_max = bbox
    roi_h, roi_w = roi_size
    # Determine center of the bounding box.
    bbox_center_y = (y_min + y_max) // 2
    bbox_center_x = (x_min + x_max) // 2

    # Slide candidate windows over the bounding box with a step size (e.g., 5 pixels)
    for y in range(y_min, y_max - roi_h + 1, 5):
        for x in range(x_min, x_max - roi_w + 1, 5):
            candidate_mask = seg_mid[y:y+roi_h, x:x+roi_w]
            # Only consider windows with zero nuclear pixels.
            if np.any(candidate_mask > 0):
                continue
            candidate_rect = (y, x, y+roi_h, x+roi_w)
            # Compute candidate center.
            candidate_center_y = y + roi_h // 2
            candidate_center_x = x + roi_w // 2
            dist_to_bbox_center = math.hypot(candidate_center_y - bbox_center_y, candidate_center_x - bbox_center_x)
            # Save candidate as (rect, distance, (center_y, center_x)).
            candidates.append((candidate_rect, dist_to_bbox_center, (candidate_center_y, candidate_center_x)))
    
    # Sort candidates based on closeness to the bounding box center.
    candidates.sort(key=lambda item: item[1])
    
    def rects_overlap(rect1, rect2):
        """Return True if rect1 and rect2 overlap; rects are (y1, x1, y2, x2)."""
        y1, x1, y2, x2 = rect1
        a1, b1, a2, b2 = rect2
        return (max(y1, a1) < min(y2, a2)) and (max(x1, b1) < min(x2, b2))
    
    min_center_distance = 500  # Minimum center-to-center distance in pixels, to sample different regions of the embyro 
    chosen = []
    chosen_centers = []
    for candidate in candidates:
        rect, _, center = candidate
        if not chosen:
            chosen.append(rect)
            chosen_centers.append(center)
        else:
            # Check that this candidate does not overlap any already chosen ROI.
            if any(rects_overlap(rect, already) for already in chosen):
                continue
            # Check that the center-to-center distance is at least 100 pixels.
            if all(math.hypot(center[0] - c[0], center[1] - c[1]) >= min_center_distance for c in chosen_centers):
                chosen.append(rect)
                chosen_centers.append(center)
                if len(chosen) == 2:
                    break
    
    return chosen


def estimate_secondary_background(
    input_dir,
    cp_mask_dir,
    output_dir,
    channels,
    roi_size=(50, 50),
    pad=10,
    log_file=None,
):
    """
    Estimate a *global* secondary background value per channel using ALL images in the experiment.

    The strategy implemented here is:

      1. For each image and channel:
         - Use the corresponding segmentation mask (nuclei) to find two "clean" ROIs with zero nuclear signal
           in the midplane (Z-mid) slice.
         - Compute the mean intensity for each ROI in the midplane image.
         - Take the *median* of these two ROI means to yield a per-image secondary background estimate.

      2. For each channel:
         - Take the *median* of the per-image secondary background estimates across ALL images.
           This yields a single constant secondary background value per channel for the experiment.

    Outputs (written to the main output directory):
      - secondary_background_estimates.csv: Per-image ROI intensities and per-image background estimates.
      - secondary_background_global.json: Per-channel global background values and method metadata.
      - secondary_background_roi_overlays/: ROI overlay images for visual verification (one per image).

    Args:
        input_dir (str): Directory containing the channel images (organized by channel subfolders).
        cp_mask_dir (str): Directory containing segmentation masks.
        output_dir (str): Main pipeline output directory (used for writing CSV/JSON/overlays).
        channels (dict): Dictionary mapping channel names -> channel indices.
        roi_size (tuple): ROI dimensions (height, width).
        pad (int): Shrink the nuclei bounding box inward by 'pad' pixels on each side.
        log_file (str): Optional log file.

    Returns:
        dict: Dictionary mapping channel name -> global secondary background value (float).
    """
    log("Estimating global secondary background values across all images...", log_file)

    overlays_root = os.path.join(output_dir, "secondary_background_roi_overlays")
    os.makedirs(overlays_root, exist_ok=True)

    estimates_csv_path = os.path.join(output_dir, "secondary_background_estimates.csv")
    global_json_path = os.path.join(output_dir, "secondary_background_global.json")

    rows = []
    global_values = {}

    # Process each channel separately.
    for channel in channels.keys():
        input_channel_dir = os.path.join(input_dir, channel)
        overlay_channel_dir = os.path.join(overlays_root, channel)
        os.makedirs(overlay_channel_dir, exist_ok=True)

        files = glob.glob(os.path.join(input_channel_dir, "*.tif"))
        if not files:
            log(f"No files found in {input_channel_dir}.", log_file)
            continue

        channel_estimates = []

        for file_path in files:
            image_id = get_cached_subfolder_name(file_path)

            # Find the corresponding segmentation mask for this image_id.
            mask_pattern = os.path.join(cp_mask_dir, f"*{image_id}*.tif")
            mask_files = glob.glob(mask_pattern)
            if not mask_files:
                log(f"No segmentation mask found for image_id {image_id} in channel {channel}.", log_file)
                continue
            seg_mask_file = mask_files[0]

            try:
                seg_mask = io.imread(seg_mask_file)
            except Exception as e:
                log(f"Error reading segmentation mask {seg_mask_file}: {e}", log_file)
                continue

            # Assume the segmentation mask is 3D; extract its midplane.
            z_mid_mask = seg_mask.shape[0] // 2
            seg_mid = seg_mask[z_mid_mask]

            # Compute bounding box from seg_mid where nuclei exist.
            ys, xs = np.where(seg_mid > 0)
            if len(ys) == 0 or len(xs) == 0:
                y_min, y_max = 0, seg_mid.shape[0]
                x_min, x_max = 0, seg_mid.shape[1]
            else:
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()

            # Shrink bounding box inward by 'pad' pixels.
            y_min = y_min + pad
            y_max = y_max - pad
            x_min = x_min + pad
            x_max = x_max - pad
            if y_max - y_min < roi_size[0] or x_max - x_min < roi_size[1]:
                log(f"Bounding box for image_id {image_id} in channel {channel} is too small for ROI.", log_file)
                continue
            bbox = (y_min, y_max, x_min, x_max)

            try:
                img = io.imread(file_path)  # Expected shape: (Z, Y, X)
            except Exception as e:
                log(f"Error reading image {file_path}: {e}", log_file)
                continue

            z_mid_img = img.shape[0] // 2
            img_mid = img[z_mid_img]

            candidate_rois = find_two_clean_rois(seg_mid, bbox, roi_size)
            if len(candidate_rois) < 2:
                log(f"Not enough clean ROIs found for image_id {image_id} in channel {channel}.", log_file)
                continue

            roi1, roi2 = candidate_rois[:2]
            roi1_mean = float(np.mean(img_mid[roi1[0]:roi1[2], roi1[1]:roi1[3]]))
            roi2_mean = float(np.mean(img_mid[roi2[0]:roi2[2], roi2[1]:roi2[3]]))

            # IMPORTANT: per-image estimate uses the median of the two ROI means
            per_image_bg = float(np.median([roi1_mean, roi2_mean]))

            channel_estimates.append(per_image_bg)

            rows.append(
                {
                    "image_id": image_id,
                    "channel": channel,
                    "file_path": file_path,
                    "mask_path": seg_mask_file,
                    "z_mid": int(z_mid_img),
                    "roi1": str(roi1),
                    "roi2": str(roi2),
                    "roi1_mean": roi1_mean,
                    "roi2_mean": roi2_mean,
                    "per_image_secondary_bg": per_image_bg,
                }
            )

            # Save ROI overlay for verification.
            try:
                img_mid_overlay = img_mid.copy()
                if img_mid_overlay.dtype == np.uint16:
                    img_mid_overlay = (img_mid_overlay / 256).astype(np.uint8)
                if len(img_mid_overlay.shape) == 2:
                    img_mid_color = cv2.cvtColor(img_mid_overlay, cv2.COLOR_GRAY2BGR)
                else:
                    img_mid_color = img_mid_overlay.copy()

                cv2.rectangle(img_mid_color, (roi1[1], roi1[0]), (roi1[3] - 1, roi1[2] - 1), (0, 165, 255), 2)
                cv2.rectangle(img_mid_color, (roi2[1], roi2[0]), (roi2[3] - 1, roi2[2] - 1), (0, 165, 255), 2)

                base_name = os.path.basename(file_path)
                name, ext = os.path.splitext(base_name)
                overlay_name = f"{name}_Z{z_mid_img}_ROI_overlay{ext}"
                overlay_path = os.path.join(overlay_channel_dir, overlay_name)
                imwrite(overlay_path, img_mid_color)
            except Exception as e:
                log(f"Error saving ROI overlay for {file_path}: {e}", log_file)

        if len(channel_estimates) == 0:
            log(f"No valid secondary background estimates computed for channel {channel}.", log_file)
            continue

        # IMPORTANT: global per-channel value uses the median across ALL images.
        global_bg = float(np.median(channel_estimates))
        global_values[channel] = global_bg
        log(f"Global secondary background for channel {channel}: {global_bg:.2f}", log_file)

    # Save per-image estimates and global values.
    try:
        pd.DataFrame(rows).to_csv(estimates_csv_path, index=False)
        log(f"Saved per-image secondary background estimates: {estimates_csv_path}", log_file)
    except Exception as e:
        log(f"Error saving secondary background estimates CSV: {e}", log_file)

    try:
        with open(global_json_path, "w") as f:
            json.dump(
                {
                    "method": "secondary_bg_median_of_two_rois_then_median_across_images",
                    "roi_size": list(roi_size),
                    "pad": int(pad),
                    "global_secondary_background": global_values,
                    "generated": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        log(f"Saved global secondary background values: {global_json_path}", log_file)
    except Exception as e:
        log(f"Error saving global secondary background JSON: {e}", log_file)

    return global_values


def secondary_background_subtraction(input_dir, cp_mask_dir, output_dir, channels, roi_size=(50, 50), pad=10, log_file=None):
    """
    Apply *global* secondary background subtraction to all images, per channel.

    This function expects a global background value per channel to exist in:
      <pipeline_output_dir>/secondary_background_global.json

    If the global file does not exist, it will be computed using
    `estimate_secondary_background()`.


    Args:
        input_dir (str): Directory containing the images (organized by channel subfolders).
        cp_mask_dir (str): Directory containing segmentation masks (needed if estimating BG).
        output_dir (str): Output directory for background-subtracted images (per channel).
        channels (dict): Dictionary mapping channel names -> channel indices.
        roi_size (tuple): ROI dimensions (height, width).
        pad (int): Shrink the nuclei bounding box inward by 'pad' pixels on each side.
        log_file (str): Optional log file.
    """
    pipeline_output_dir = os.path.dirname(output_dir.rstrip(os.sep))
    global_json_path = os.path.join(pipeline_output_dir, "secondary_background_global.json")

    # If global BG values are missing, estimate them now.
    if not os.path.exists(global_json_path):
        log(
            "Global secondary background file not found. Estimating global background values now...",
            log_file,
        )
        estimate_secondary_background(
            input_dir=input_dir,
            cp_mask_dir=cp_mask_dir,
            output_dir=pipeline_output_dir,
            channels=channels,
            roi_size=roi_size,
            pad=pad,
            log_file=log_file,
        )

    try:
        with open(global_json_path, "r") as f:
            global_data = json.load(f)
        global_values = global_data.get("global_secondary_background", {})
    except Exception as e:
        raise RuntimeError(f"Failed to load global secondary background JSON: {global_json_path}. Error: {e}")

    log("Applying global secondary background subtraction...", log_file)

    for channel in channels.keys():
        if channel not in global_values:
            log(f"No global background value found for channel {channel}; skipping.", log_file)
            continue

        bg_value = float(global_values[channel])
        input_channel_dir = os.path.join(input_dir, channel)
        output_channel_dir = os.path.join(output_dir, channel)
        os.makedirs(output_channel_dir, exist_ok=True)

        files = glob.glob(os.path.join(input_channel_dir, "*.tif"))
        if not files:
            log(f"No files found in {input_channel_dir}.", log_file)
            continue

        def process_file(file_path):
            try:
                img = io.imread(file_path)
                img_sub = np.clip(img.astype(np.float32) - bg_value, 0, 65535).astype(np.uint16)

                base_name = os.path.basename(file_path)
                output_path = os.path.join(output_channel_dir, base_name)

                imwrite(output_path, img_sub)
                return file_path
            except Exception as e_proc:
                log(f"Error processing file {file_path}: {e_proc}", log_file)

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(executor.map(process_file, files))

        log(f"Completed global secondary background subtraction for channel {channel} (bg={bg_value:.2f}).", log_file)

    log("Secondary background subtraction completed.", log_file)


def find_matching_masks(raw_dir, mask_dir):
    """
    Match raw channel images with corresponding segmented masks using a shared image_id.

    Assumptions:
    - Raw channel images are named like: C{n}-{image_id}.tif  (or .tiff or .ome.tif)
    - Mask images contain {image_id} somewhere in the filename 
      e.g. {image_id}_cp_masks.tif, or {image_id}_sphericity_filtered.tif, etc.

    Supports raw extensions: .tif, .tiff, .ome.tif
    Supports mask extensions: .tif, .tiff, .ome.tif
    """

    def _image_id_from_raw_filename(fname: str) -> str:
        base = os.path.basename(fname)

        # Remove extensions (.tif/.tiff, and optionally .ome)
        stem1, ext1 = os.path.splitext(base)      # removes .tif/.tiff
        stem2, ext2 = os.path.splitext(stem1)     # removes .ome if present
        stem = stem2 if ext2.lower() == ".ome" else stem1
        stem = stem.rstrip(".")

        # Remove channel prefix "C{n}-" if present
        if stem.startswith("C") and "-" in stem:
            left, right = stem.split("-", 1)
            if left[1:].isdigit():  # "C4" etc
                stem = right

        return stem

    # Gather raw images (channel images)
    raw_patterns = ["*.ome.tif", "*.tif", "*.tiff", "*.OME.TIF", "*.TIF", "*.TIFF"]
    raw_images = []
    for pat in raw_patterns:
        raw_images.extend(glob.glob(os.path.join(raw_dir, pat)))
    raw_images = sorted(set(raw_images))

    # Gather masks
    mask_patterns = ["*.ome.tif", "*.tif", "*.tiff", "*.OME.TIF", "*.TIF", "*.TIFF"]
    mask_images = []
    for pat in mask_patterns:
        mask_images.extend(glob.glob(os.path.join(mask_dir, pat)))
    mask_images = sorted(set(mask_images))

    matches = []
    if not raw_images or not mask_images:
        return matches

    # Map mask basenames for faster matching
    mask_basenames = [(m, os.path.basename(m)) for m in mask_images]

    for raw_image in raw_images:
        image_id = _image_id_from_raw_filename(raw_image)

        # Find first mask that contains this image_id
        match_path = None
        for mpath, mbase in mask_basenames:
            if image_id in mbase:
                match_path = mpath
                break

        if match_path is not None:
            matches.append((raw_image, match_path))

    return matches


def process_raw_and_masks(raw_dir, mask_dir, output_dir):
    """
    Combine matched raw images and masks into two-channel 3D TIFF stacks.

    Args:
        raw_dir (str): Directory containing raw images.
        mask_dir (str): Directory containing mask images.
        output_dir (str): Directory to save combined outputs.
    """
    matches = find_matching_masks(raw_dir, mask_dir)

    if not matches:
        print("No matching raw-mask pairs found.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for raw_image, mask_image in matches:
        # Define the output file name
        output_file = os.path.join(output_dir, f"{os.path.basename(raw_image)}_segmentation_overlay.tif")

        # Skip if the file already exists
        if os.path.exists(output_file):
            print(f"Skipping {output_file}, already exists.")
            continue

        print(f"Processing pair: {raw_image} <-> {mask_image}")

        try:
            # Load the raw and mask images
            raw = io.imread(raw_image)
            mask = io.imread(mask_image)

            # Ensure both images have the same dimensions
            if raw.shape != mask.shape:
                print(f"Dimension mismatch: {raw_image} and {mask_image}. Skipping.")
                continue

            # Check and correct data types
            if raw.dtype not in [np.uint16, np.float32]:
                print(f"Converting raw image from {raw.dtype} to uint16.")
                raw = raw.astype(np.uint16)
            if mask.dtype not in [np.uint16, np.float32]:
                print(f"Converting mask image from {mask.dtype} to uint16.")
                mask = mask.astype(np.uint16)

            # Combine into a 2-channel image (C, Z, Y, X)
            combined = np.stack((raw, mask), axis=0)  # Shape: (C, Z, Y, X)

            # Save the combined image
            save_as_ome_tiff(combined, output_file)
            print(f"Saved combined image to: {output_file}")

        except Exception as e:
            print(f"Error processing {raw_image} and {mask_image}: {e}")

def quantify_labels_per_channel(config, input_dir, segmented_nuclei_dir, split_channels_dir, output_dir, log_file=None):
    """
    Quantify label measurements for each channel in the segmented nuclei and raw 3D images.


    Metadata / voxel size requirement
    ---------------------------------
      - subfolder mode: key is the subfolder name
      - flat mode: key is the cleaned filename stem (e.g., "sample.ome.tif" -> "sample")

    If metadata is missing, there will be an error 
    """

    def get_voxel_size(metadata_cache, image_id, raw_image_path, default_voxel_size=(1.0, 1.0, 1.0), log_file=None):
        """
        Retrieve voxel size for a given image_id.

        Args:
            metadata_cache (dict): In-memory metadata cache (global).
            image_id (str): Pipeline image identifier.
            raw_image_path (str): Path to the raw image for metadata extraction if needed.
            default_voxel_size (tuple): Fallback voxel size, used only if metadata is missing fields.
            log_file (str, optional): Log file path.

        Returns:
            tuple: (z, y, x) voxel size in physical units.
        """
        metadata = get_or_create_metadata(image_id=image_id, raw_image_path=raw_image_path, log_file=log_file)
        voxel_size = metadata.get("physical_pixel_sizes", default_voxel_size)
        log(f"Voxel size for {image_id}: {voxel_size}", log_file)
        return voxel_size

    def process_label(prop, labels, voxel_size, voxel_volume, raw):
        """
        Compute per-label measurements for a single nucleus.

        Notes:
        - `labels` and `raw` are the *full* 3D stacks (Z,Y,X).
        - We crop to the label bounding box for expensive operations (texture + sphericity surface estimation).
        - Haralick + autocorrelation features are computed via your nucleus-local function
          `compute_texture_features_for_label(region_cropped, raw_crop)`.

        Returns:
            dict or None
        """
        try:
            label_id = prop.label

            # Boolean mask of label in the *full* volume
            region_mask = (labels == label_id)

            # Crop to bounding box for efficiency
            bbox = prop.bbox  # (min_z, min_y, min_x, max_z, max_y, max_x)
            region_cropped = region_mask[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
            raw_crop = raw[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]

            # Compute texture features (masked GLCM, multi-distance contrast, 3D autocorrelation)
            texture_features = compute_texture_features_for_label(region_cropped, raw_crop)

            # ---- Sphericity / surface area estimation ----
            # Upsample the cropped binary region for smoother marching-cubes surfaces
            upsampled_region = resize(
                region_cropped.astype(float),
                (
                    region_cropped.shape[0] * 2,
                    region_cropped.shape[1] * 2,
                    region_cropped.shape[2] * 2,
                ),
                mode="reflect",
                anti_aliasing=True,
            )

            # Light smoothing makes the surface less “blocky”
            smoothed_region = gaussian_filter(upsampled_region, sigma=0.5)

            # Marching cubes in physical units (spacing is voxel_size/2 because we upsampled by 2x)
            verts, faces, _, _ = measure.marching_cubes(
                smoothed_region,
                spacing=np.array(voxel_size) / 2
            )
            surface_area = measure.mesh_surface_area(verts, faces)

            # Volume in physical units (voxel count * physical voxel volume)
            volume = float(np.sum(region_cropped) * voxel_volume)

            # Sphericity definition: (36*pi*V^2) / (A^3)
            if surface_area > 0:
                sphericity = float((36 * np.pi * volume**2) / (surface_area**3))
            else:
                sphericity = 0.0

            # ---- Intensity-based measurements ----
            # regionprops already computed intensity stats using intensity_image=raw
            min_intensity = prop.intensity_min
            max_intensity = prop.intensity_max
            mean_intensity = prop.mean_intensity
            sum_intensity = float(prop.intensity_image.sum())

            # For robust stats, pull intensities directly from raw at prop.coords
            coords = prop.coords
            intensity_values = raw[coords[:, 0], coords[:, 1], coords[:, 2]]

            median_intensity = float(np.median(intensity_values))
            std_intensity = float(np.std(intensity_values))
            cv_intensity = float(std_intensity / mean_intensity) if mean_intensity > 0 else 0.0
            mad_intensity = float(np.median(np.abs(intensity_values - median_intensity)))

            return {
                "Label ID": label_id,
                "Voxel Count": prop.area,
                "Volume": volume,
                "Surface Area": float(surface_area),
                "Sphericity": sphericity,
                "Min Intensity": float(min_intensity),
                "Max Intensity": float(max_intensity),
                "Mean Intensity": float(mean_intensity),
                "Median Intensity": median_intensity,
                "Sum Intensity": sum_intensity,
                "Std Dev Intensity": std_intensity,
                "CV Intensity": cv_intensity,
                "MAD Intensity": mad_intensity,

                # Texture features (computed inside compute_texture_features_for_label)
                "Haralick Entropy": float(texture_features.get("Haralick Entropy", 0.0)), # Auxiliary texture metrics retained for exploratory analysis
                "Haralick Contrast d=1": float(texture_features.get("Haralick Contrast d=1", 0.0)),
                "Haralick Contrast d=2": float(texture_features.get("Haralick Contrast d=2", 0.0)),
                "Haralick Contrast d=4": float(texture_features.get("Haralick Contrast d=4", 0.0)),
                "3D Autocorrelation": float(texture_features.get("3D Autocorrelation", 0.0)), # Auxiliary metrics retained for exploratory analysis
            }

        except Exception as e:
            # Keep failure local to one nucleus; continue other labels
            print(f"Error processing label {prop.label}: {e}")
            return None

    # -----------------------------
    # Main loop: channel-by-channel
    # -----------------------------
    for channel_name, channel_index in config["channels"].items():
        print(f"Processing channel: {channel_name}")

        # Output directory for this channel’s quantification CSVs
        channel_output_dir = os.path.join(output_dir, f"{channel_name}_quantification")
        os.makedirs(channel_output_dir, exist_ok=True)

        # Match raw per-channel images to segmentation masks by filename containment logic.
        # NOTE: This depends on your find_matching_masks() convention (raw basename after stripping "C#-").
        matches = find_matching_masks(
            os.path.join(split_channels_dir, channel_name),
            segmented_nuclei_dir
        )
        if not matches:
            print(f"No matching raw-mask pairs found for channel {channel_name}.")
            continue

        # Iterate over each matched (raw_channel_image, label_mask) pair directly.
        # This works in both subfolder mode and flat mode because it does not rely on os.listdir(input_dir).
        for raw_image_path, label_image_path in matches:
            # Derive the metadata_cache key from the channel filename.
            # This must match how you cached metadata earlier:
            #   - subfolder mode: key is subfolder name
            #   - flat mode: key is cleaned filename stem (e.g. sample.ome.tif -> sample)
            image_id = get_cached_subfolder_name(raw_image_path)

            # CSV name is based on the raw channel filename (keeps your current naming convention)
            raw_image_name = os.path.splitext(os.path.basename(raw_image_path))[0]
            quant_csv_path = os.path.join(channel_output_dir, f"{raw_image_name}_quantification.csv")

            # Skip if already quantified
            if os.path.exists(quant_csv_path):
                print(f"Skipping {quant_csv_path}, already exists.")
                continue

            print(f"Processing pair: {raw_image_path} <-> {label_image_path}")

            # Load 3D data
            labels = io.imread(label_image_path)  # (Z,Y,X)
            raw = io.imread(raw_image_path)       # (Z,Y,X)

            # Validate dimensionality and shape
            if labels.ndim != 3 or raw.ndim != 3:
                print(f"Skipping pair due to dimensionality: {raw_image_path}, {label_image_path}")
                continue
            if labels.shape != raw.shape:
                print(f"Skipping pair due to shape mismatch: {raw_image_path}, {label_image_path}")
                continue

            # Get voxel size from cache (critical for volume/surface computations)
            voxel_size = get_voxel_size(metadata_cache, image_id=image_id, raw_image_path=raw_image_path, log_file=log_file)
            voxel_volume = float(np.prod(voxel_size))

            # Extract region props with intensity image
            props = regionprops(labels, intensity_image=raw)
            if not props:
                print(f"No labels found in {label_image_path}.")
                continue

            # Process labels in parallel (CPU-bound-ish; OK with threads for a first pass)
            all_data = []
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                results = list(executor.map(lambda p: process_label(p, labels, voxel_size, voxel_volume, raw), props))

            # Drop failures
            all_data = [r for r in results if r is not None]

            # Save one CSV per image volume
            if all_data:
                print(f"Saving results for {raw_image_name} to {quant_csv_path}.")
                df = pd.DataFrame(all_data)
                df.to_csv(quant_csv_path, index=False)
            else:
                print(f"No data collected for {raw_image_name}.")


def compute_texture_features_for_label(
    label_mask: np.ndarray,
    raw_crop: np.ndarray,
    levels: int = 256,
    distances=(1, 2, 4),
    angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
    smooth_sigma: float = 1.0,
):
     """
    Compute texture features for a single 3D nucleus.

    The function calculates:

    - Haralick entropy (2D, averaged across z-slices)
    - Haralick contrast at multiple pixel distances (d=1, 2, and 4)
    - 3D autocorrelation between adjacent z-slices

    Haralick features are computed using within-mask pixel pairs only,
    excluding background pixels from GLCM calculations. Intensities are
    normalized once per nucleus using the masked voxel intensity range,
    quantized, and optionally smoothed prior to feature calculation.

    Returns
    -------
    dict
        Dictionary containing:

        - "Haralick Entropy"
        - "Haralick Contrast d=1"
        - "Haralick Contrast d=2"
        - "Haralick Contrast d=4"
        - "3D Autocorrelation"

    Notes
    -----
    - Haralick features are computed independently for each z-slice and
      averaged across slices.
    - Slices with insufficient mask coverage are skipped.
    - Autocorrelation is computed only from voxels present within the mask
      in adjacent z-slices.
    """

    # ----------------------------
    # Basic checks / type safety
    # ----------------------------
    if label_mask is None or raw_crop is None:
        return {
            "Haralick Entropy": 0.0,
            **{f"Haralick Contrast d={d}": 0.0 for d in distances},
            "3D Autocorrelation": 0.0,
        }

    label_mask = (label_mask > 0)
    if label_mask.shape != raw_crop.shape:
        raise ValueError(f"label_mask and raw_crop must have same shape. Got {label_mask.shape} vs {raw_crop.shape}")

    # ----------------------------
    # Nucleus-global normalization (once per 3D nucleus)
    # ----------------------------
    roi_vox = raw_crop[label_mask]
    if roi_vox.size == 0:
        return {
            "Haralick Entropy": 0.0,
            **{f"Haralick Contrast d={d}": 0.0 for d in distances},
            "3D Autocorrelation": 0.0,
        }

    # Intensity range from masked voxels
    pmin, pmax = np.percentile(roi_vox, [1, 99])
    if pmax <= pmin:
        return {
            "Haralick Entropy": 0.0,
            **{f"Haralick Contrast d={d}": 0.0 for d in distances},
            "3D Autocorrelation": 0.0,
        }

    # Normalize entire crop against nucleus-level range, then clip
    norm = (raw_crop.astype(np.float32) - float(pmin)) / (float(pmax) - float(pmin))
    norm = np.clip(norm, 0.0, 1.0)

    # Quantization helper: map [0,1] -> {0,...,levels-1}
    def quantize01_to_levels(arr01: np.ndarray, levels_: int) -> np.ndarray:
        # Use floor so 1.0 maps to levels-1 after clipping below
        out = np.floor(arr01 * (levels_ - 1)).astype(np.int32)
        out = np.clip(out, 0, levels_ - 1)
        return out

    # ----------------------------
    # Masked GLCM implementation
    # ----------------------------
    # Compute masked GLCMs independently for each z-slice.
    def masked_glcm_2d(quant_img: np.ndarray, mask2d: np.ndarray, dist: int, theta: float, levels_: int) -> np.ndarray:
        """
        Build a masked GLCM for a single 2D slice.

        Parameters
        ----------
        quant_img : ndarray
            Quantized image with values in [0, levels-1].
        mask2d : ndarray
            Binary mask defining pixels included in the calculation.
        dist : int
            Pixel offset distance.
        theta : float
            Offset angle in radians.
        Returns
        -------
        ndarray or None
            Normalized GLCM, or None if no valid pixel pairs are available.
        """
        # Compute integer pixel shift for this angle.
        dr = int(np.round(dist * np.sin(theta)))   # row shift
        dc = int(np.round(dist * np.cos(theta)))   # col shift

        if dr == 0 and dc == 0:
            return None

        # Create aligned views of "source" pixels and "neighbor" pixels
        # so that (src[r,c], nbr[r,c]) are offset by (dr,dc) in the original image.
        H, W = quant_img.shape

        # Determine overlapping region after shifting
        r0_src = max(0, -dr)
        r1_src = min(H, H - dr)  # exclusive
        c0_src = max(0, -dc)
        c1_src = min(W, W - dc)

        r0_nbr = r0_src + dr
        r1_nbr = r1_src + dr
        c0_nbr = c0_src + dc
        c1_nbr = c1_src + dc

        src = quant_img[r0_src:r1_src, c0_src:c1_src]
        nbr = quant_img[r0_nbr:r1_nbr, c0_nbr:c1_nbr]

        m_src = mask2d[r0_src:r1_src, c0_src:c1_src]
        m_nbr = mask2d[r0_nbr:r1_nbr, c0_nbr:c1_nbr]

        valid = m_src & m_nbr
        if not np.any(valid):
            return None

        i = src[valid].ravel()
        j = nbr[valid].ravel()

        # Count co-occurrences into a matrix
        glcm = np.zeros((levels_, levels_), dtype=np.float64)
        # Fast counting using numpy indexing
        np.add.at(glcm, (i, j), 1)

        # Make symmetric if desired by mirroring counts (matches symmetric=True behavior)
        glcm = glcm + glcm.T

        s = glcm.sum()
        if s <= 0:
            return None
        glcm /= s
        return glcm

    # Entropy helper (expects normalized probabilities)
    def glcm_entropy(glcm: np.ndarray) -> float:
        # Add epsilon to avoid log(0); epsilon does not materially change the value
        eps = 1e-12
        return float(-np.sum(glcm * np.log2(glcm + eps)))

    # Contrast helper (Haralick contrast definition):
    # contrast = sum_{i,j} P(i,j) * (i-j)^2
    def glcm_contrast(glcm: np.ndarray) -> float:
        idx = np.arange(glcm.shape[0], dtype=np.float64)
        diff2 = (idx[:, None] - idx[None, :]) ** 2
        return float(np.sum(glcm * diff2))

    # ----------------------------
    # Compute Haralick features slice-by-slice, but using nucleus-global normalization
    # ----------------------------
    entropy_vals = []
    contrast_vals_by_d = {d: [] for d in distances}

    Z = norm.shape[0]
    for z in range(Z):
        mask2d = label_mask[z]
        if mask2d.sum() < 10:
            # Too few pixels -> unreliable GLCM; skip this slice
            continue

        slice01 = norm[z]

        # Optional smoothing to suppress pixel noise before quantization.
        # Important: smoothing is applied in the real geometry, not on any reshaped ROI.
        if smooth_sigma is not None and smooth_sigma > 0:
            slice01 = gaussian_filter(slice01, sigma=float(smooth_sigma))

        q = quantize01_to_levels(slice01, levels)

        # For each distance, aggregate across angles by averaging GLCMs (like graycomatrix + mean props).
        # We do this explicitly so that each angle contributes equally among valid pairs.
        for d in distances:
            glcms = []
            for theta in angles:
                g = masked_glcm_2d(q, mask2d, dist=int(d), theta=float(theta), levels_=levels)
                if g is not None:
                    glcms.append(g)

            if not glcms:
                continue

            # Average probability matrices across angles (equal-angle weighting)
            gmean = np.mean(glcms, axis=0)

            # Compute entropy using the shortest offset distance.
            if d == distances[0]:
                entropy_vals.append(glcm_entropy(gmean))

            contrast_vals_by_d[d].append(glcm_contrast(gmean))

    # Aggregate per-nucleus Haralick values across slices
    har_entropy = float(np.mean(entropy_vals)) if entropy_vals else 0.0
    out = {"Haralick Entropy": har_entropy}

    for d in distances:
        vals = contrast_vals_by_d[d]
        out[f"Haralick Contrast d={d}"] = float(np.mean(vals)) if vals else 0.0

    # ----------------------------
    # 3D autocorrelation (masked-only) between adjacent Z slices
    # ----------------------------
    # Compute entropy using the shortest offset distance.
    masked_raw = raw_crop.astype(np.float32)
    corr_vals = []
    for z in range(1, Z):
        m = label_mask[z] & label_mask[z - 1]
        if not np.any(m):
            continue
        a = masked_raw[z][m].ravel()
        b = masked_raw[z - 1][m].ravel()
        if a.size < 5:
            continue
        # Guard against constant arrays (corrcoef would produce nan)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        if np.isfinite(corr):
            corr_vals.append(float(corr))

    out["3D Autocorrelation"] = float(np.mean(corr_vals)) if corr_vals else 0.0

    return out



def merge_csv_files_by_condition(config, output_dir, log_file=None):
    """
    Legacy helper for merging per-embryo quantification CSVs by condition.

    This function is retained for compatibility with older analysis workflows.

    The current plotting workflow reads per-embryo quantification CSV files directly
    and does not require merged intensity CSVs.

    This function:
    - Looks in each channel-specific output directory.
    - Groups CSVs based on a naming pattern defined by a regular expression.
    - Concatenates intensity measurements from matched CSVs with one blank row separating each file (e.g., each embryo).
    - Outputs the merged result as `*_merged_results.csv` per group.
    - Merged results combine all embryos from each condition in one file, separated by a blank (NaN) row.

    **Regex Pattern Note:**
    The regex used to group files is:
        r'(C\\d+-.*?_E)_\\d+_quantification\\.csv'

    This expects filenames like:
        C2-imageX_E_2_quantification.csv  
        C2-imageX_E_3_quantification.csv  

    It extracts `C2-imageX_E` as the group key, which includes the channel and condition.  
    Then, `_E_\\d+` is further stripped programmatically to yield a final common group like `C2-imageX`.

    **If the file format changes**, users can edit this pattern. For example:
    - For filenames like `C1-sample123_E1.csv`, you might change the regex to:
      `r'(C\\d+-sample\\d+)_E\\d+\\.csv'`
    - If there's no replicate or embryo ID, a simpler match like:
      `r'(.*)_quantification\\.csv'` may suffice.

    Args:
        config (dict): Configuration dictionary with channel names (e.g., {"DNA": 4, "H3K9me3": 2}).
        output_dir (str): Root directory containing channel subfolders with quantification results.
        log_file (str, optional): Path to log file for recording progress or errors.

    Raises:
        ValueError: If required folders are missing or no files are found to merge.

    Output:
        Saves merged CSVs in:
            <output_dir>/<channel_name>_quantification/merged_results/<group_name>_merged_results.csv

    Example file tree:
        output_dir/
        ├── DNA_quantification/
        │   ├── C4-sample1_E_1_quantification.csv
        │   ├── C4-sample1_E_2_quantification.csv
        │   └── ...
        └── H3K9me3_quantification/
            ├── C2-sample5_E_1_quantification.csv
            └── ...
    """

    # Fixed regex pattern for grouping files
    # Accepts: ..._E_01..., ..._E__01..., ..._E-01...
    regex_pattern = r'(C\d+-.*?_E)(?:_|-)+\d+_quantification\.csv'

    # Iterate over each channel in the configuration
    for channel_name in config["channels"].keys():
        print(f"Processing merging for channel: {channel_name}")
        channel_output_dir = os.path.join(output_dir, f"{channel_name}_quantification")

        # Check if the channel's quantification directory exists
        if not os.path.exists(channel_output_dir):
            log(f"Skipping channel {channel_name}, quantification directory does not exist.", log_file)
            continue

        # Directory for merged results
        merged_results_dir = os.path.join(channel_output_dir, "merged_results")
        os.makedirs(merged_results_dir, exist_ok=True)

        # Initialize a dictionary to group matching files
        file_groups = {}

        # Process each file in the channel's quantification directory
        for filename in os.listdir(channel_output_dir):
            file_path = os.path.join(channel_output_dir, filename)
            if os.path.isfile(file_path) and not filename.startswith('._'):
                match = re.match(regex_pattern, filename)
                if match:
                    # Extract the common name based on the regex
                    full_name = match.group(1)
                    common_name = re.sub(r'_E_\d+', '', full_name)
                    file_groups.setdefault(common_name, []).append(filename)
                    log(f"Grouping file {filename} under {common_name}", log_file)
                else:
                    log(f"No match for file: {filename}", log_file)

        if not file_groups:
            log(f"No files found to merge for channel {channel_name}.", log_file)
            continue

        # Process and merge the data for each group
        for common_name, files in file_groups.items():
            merged_data_path = os.path.join(merged_results_dir, f"{common_name}_merged_results.csv")
            if os.path.exists(merged_data_path):
                log(f"Skipping {merged_data_path}, already exists.", log_file)
                continue

            merged_data = []
            header = None

            for i, file in enumerate(files):
                intensity_path = os.path.join(channel_output_dir, file)

                # Load the intensity data
                try:
                    with open(intensity_path, 'r') as f:
                        if header is None:
                            header = f.readline().strip()  # Read the header line

                    intensity_data = np.genfromtxt(intensity_path, delimiter=',', skip_header=1)
                    if intensity_data.size > 0:
                        # Add an empty line before appending new data if it's not the first file
                        if i > 0:
                            merged_data.append(np.full((1, intensity_data.shape[1]), np.nan))  # Adding an empty line
                        merged_data.append(intensity_data)
                except Exception as e:
                    log(f"Error loading file {file}: {e}", log_file)
                    continue

            if merged_data:
                # Combine all data into one array
                merged_data = np.vstack(merged_data)

                # Save the merged data
                np.savetxt(merged_data_path, merged_data, delimiter=',', header=header, comments='', fmt='%.6f')
                log(f"Saved merged data to {merged_data_path}", log_file)
            else:
                log(f"No data merged for group {common_name} in channel {channel_name}.", log_file)

    print("Merging completed for all channels.")

def correlate_labels_per_channel(config, input_dir, segmented_nuclei_dir, channels_dir, output_dir, log_file=None):
    """
    Compute per-label Pearson correlation between two channels inside each segmented nucleus.

    For every label > 0, compute Pearson correlation between channel1 and channel2 voxel intensities
    inside that label. Save one CSV per image item:
        <image_id>_correlation_cX_cY.csv    
    """

    # Retrieve which two channels to correlate from config
    correlation_channels = config.get("correlation_channels")
    if not correlation_channels or "channel1" not in correlation_channels or "channel2" not in correlation_channels:
        raise ValueError("config['correlation_channels'] must define 'channel1' and 'channel2'")

    channels_config = config["channels"]
    ch1_name = correlation_channels["channel1"]
    ch2_name = correlation_channels["channel2"]

    if ch1_name not in channels_config or ch2_name not in channels_config:
        raise ValueError(
            f"Correlation channel names must exist in config['channels'].\n"
            f"Got channel1={ch1_name}, channel2={ch2_name}, channels={list(channels_config.keys())}"
        )

    ch1_index = channels_config[ch1_name]
    ch2_index = channels_config[ch2_name]

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------
    # Iterate over segmentation masks on disk
    # --------------------------------------------

    mask_files = sorted(glob.glob(os.path.join(segmented_nuclei_dir, "*.tif")))
    if not mask_files:
        log(f"No segmentation mask files found in {segmented_nuclei_dir}. Skipping correlation analysis.", log_file)
        return

    for seg_mask_path in mask_files:
        # Parse image_id in a way consistent with metadata_cache naming
        image_id = get_cached_subfolder_name(seg_mask_path)
        image_id_for_channels = re.sub(r'_sphericity_filtered$', '', image_id)

        print(f"Processing item: {image_id} for correlation analysis")

        # Construct expected channel file paths
        # Assumes naming: channels_dir/<ChannelName>/C{channel_index}-{image_id}.tif
        ch1_path = os.path.join(channels_dir, ch1_name, f"C{ch1_index}-{image_id_for_channels}.tif")
        ch2_path = os.path.join(channels_dir, ch2_name, f"C{ch2_index}-{image_id_for_channels}.tif")
        
        if not os.path.exists(ch1_path) or not os.path.exists(ch2_path):
            log(
                f"Skipping {image_id}: missing channel image(s). "
                f"Expected:\n  {ch1_path}\n  {ch2_path}",
                log_file
            )
            continue

        # Output CSV path (per item)
        subfolder_csv = os.path.join(output_dir, f"{image_id}_correlation_c{ch1_index}_c{ch2_index}.csv")

        # Skip if already done
        if os.path.exists(subfolder_csv):
            log(f"Skipping {image_id}: correlation CSV already exists: {subfolder_csv}", log_file)
            continue

        # Load 3D images + mask
        try:
            channel1_image = io.imread(ch1_path)
            channel2_image = io.imread(ch2_path)
            segmentation_mask = io.imread(seg_mask_path)
        except Exception as e:
            log(f"Error loading images for {image_id}: {e}", log_file)
            continue

        # Validate expected shapes
        if channel1_image.ndim != 3 or channel2_image.ndim != 3 or segmentation_mask.ndim != 3:
            log(f"Skipping {image_id}: one or more images are not 3D.", log_file)
            continue

        if channel1_image.shape != channel2_image.shape or channel1_image.shape != segmentation_mask.shape:
            log(
                f"Skipping {image_id}: shape mismatch.\n"
                f"  ch1: {channel1_image.shape}\n"
                f"  ch2: {channel2_image.shape}\n"
                f"  mask:{segmentation_mask.shape}",
                log_file
            )
            continue

        # Unique labels 
        unique_labels = np.unique(segmentation_mask)

        def process_single_label(label_val):
            """
            Compute pearson correlation for one label inside the segmentation mask.

            Returns:
                dict or None (None for background)
            """
            if label_val == 0:
                return None  # background

            mask = (segmentation_mask == label_val)
            vals1 = channel1_image[mask]
            vals2 = channel2_image[mask]

            # Require enough voxels to make correlation meaningful
            if vals1.size < 10:
                corr = None
            else:
                try:
                    corr, _ = pearsonr(vals1, vals2)
                except Exception:
                    corr = None

            return {"Image": image_id, "Label": int(label_val), "Correlation": corr}

        # Process labels in parallel
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(executor.map(process_single_label, unique_labels))
        results = [r for r in results if r is not None]

        # Save CSV
        if results:
            df = pd.DataFrame(results)
            df.to_csv(subfolder_csv, index=False)
            log(f"Saved correlation results for {image_id} to {subfolder_csv}.", log_file)
            print(f"Saved correlation results for {image_id} to {subfolder_csv}.")
        else:
            log(f"No correlation data for {image_id} (no valid labels).", log_file)
            print(f"No correlation data for {image_id}.")

def merge_correlation_csv_files(config, correlations_dir, log_file=None):
    """
    Merges correlation CSV files by experimental condition, aggregating replicate-level data into one file per condition.

    This function:
    - Looks in the `correlations_dir` for files following a naming convention that includes a condition, optional replicate number, and channel pairing.
    - Uses a regular expression to extract and group files by condition name (replicate suffix is ignored).
    - Concatenates correlation matrices across replicates, inserting a blank row (filled with NaNs) to separate each embryo/replicate.
    - Outputs the merged result as `<condition>_merged_correlations.csv` per group in a `merged_results` subdirectory.

    **Regex Pattern Note:**
    The default regex used to group files is:
        r'^(.*)_E_\\d+_correlation_c\\d+_c\\d+\\.csv$'

    This expects filenames like:
        ConditionA_E_1_correlation_c1_c2.csv  
        ConditionA_E_2_correlation_c1_c2.csv  

    The grouping key extracted is `ConditionA`, ignoring the `_E_1`, `_E_2`, etc. suffixes.

    Args:
        config (dict): Configuration dictionary, used to determine relevant channels and structure.
        correlations_dir (str): Directory containing per-embryo correlation CSV files for each condition.
        log_file (str, optional): Path to a log file for recording messages or errors.

    Raises:
        ValueError: If the correlation directory does not exist or no matching files are found.

    Output:
        Saves merged CSVs in:
            correlations_dir/merged_results/<condition>_merged_correlations.csv

    Example file tree:
        correlations_dir/
        ├── ConditionA_E_1_correlation_c1_c2.csv
        ├── ConditionA_E_2_correlation_c1_c2.csv
        ├── ConditionB_E_1_correlation_c1_c2.csv
        └── merged_results/
            ├── ConditionA_merged_correlations.csv
            └── ConditionB_merged_correlations.csv
    """
    regex_pattern = r'^(.*)_E(?:_+|-)\d+(?:_sphericity_filtered)?_correlation_c\d+_c\d+\.csv$'
    
    # Directory for merged results inside correlations_dir
    merged_results_dir = os.path.join(correlations_dir, "merged_results")
    os.makedirs(merged_results_dir, exist_ok=True)
    
    # Dictionary to group files by condition
    file_groups = {}
    
    # Iterate over each file in correlations_dir
    for filename in os.listdir(correlations_dir):
        file_path = os.path.join(correlations_dir, filename)
        if os.path.isfile(file_path) and not filename.startswith("._"):
            match = re.match(regex_pattern, filename)
            if match:
                condition = match.group(1)
                file_groups.setdefault(condition, []).append(filename)
                log(f"Grouping file {filename} under condition '{condition}'.")
            else:
                log(f"File {filename} did not match the expected pattern.")
    
    if not file_groups:
        log("No correlation CSV files found to merge.")
        return
    
    # For each condition group, merge the CSV files
    for condition, files in file_groups.items():
        merged_csv_path = os.path.join(merged_results_dir, f"{condition}_merged_correlation.csv")
        if os.path.exists(merged_csv_path):
            log(f"Skipping merging for {merged_csv_path}, already exists.")
            continue
        
        merged_data = []
        header = None
        
        for i, file in enumerate(files):
            file_full_path = os.path.join(correlations_dir, file)
            try:
                with open(file_full_path, 'r') as f:
                    if header is None:
                        header = f.readline().strip()  # Read header once
                data = np.genfromtxt(file_full_path, delimiter=',', skip_header=1)
                if data.size > 0:
                    # Optionally add a blank line between files if not the first file
                    if i > 0:
                        merged_data.append(np.full((1, data.shape[1]), np.nan))
                    merged_data.append(data)
            except Exception as e:
                log(f"Error loading file {file}: {e}")
                continue
        
        if merged_data:
            merged_data = np.vstack(merged_data)
            # Save merged data to CSV; include header and no additional comments
            np.savetxt(merged_csv_path, merged_data, delimiter=',', header=header, comments='', fmt='%.6f')
            log(f"Saved merged correlation data for condition '{condition}' to {merged_csv_path}.")
        else:
            log(f"No data merged for condition '{condition}'.")
    
    log("Merging of correlation CSV files completed.")


def run_plotting(config, output_dir, plot_dir, log_file):
    """
    Run the external plotting script on per-embryo quantification CSVs.

    Args:
        config (dict): Pipeline configuration dictionary.
        output_dir (str): Base output directory containing `<channel>_quantification/` folders.
        plot_dir (str): Directory where plot outputs should be written.
        log_file (str): Log file path.
    """
    channels = config.get("channels", {})
    plotting_script = config.get("plotting_script", None)

    if not channels:
        raise ValueError("Channel mappings are missing or empty in the configuration.")
    if not plotting_script:
        raise ValueError(
            "Plotting script path is missing in the configuration. "
            "Set `plotting_script` in the JSON config to the path of your plotting script."
        )

    os.makedirs(plot_dir, exist_ok=True)

    # Optional: allow the main pipeline config to specify a control identifier substring
    # that the plotting script can use to identify the control condition for specific stats/annotations.
    control_substring = config.get("control_condition_substring", None)

    log("Starting plot generation for each channel from per-embryo CSVs...", log_file)

    for channel_name in channels.keys():
        data_dir = os.path.join(output_dir, f"{channel_name}_quantification")
        if not os.path.isdir(data_dir):
            log(f"WARNING: Skipping plotting for {channel_name}: missing directory {data_dir}", log_file)
            continue

        # Each channel gets its own plot output folder
        channel_plot_dir = os.path.join(plot_dir, channel_name)
        os.makedirs(channel_plot_dir, exist_ok=True)

        cmd = [
            sys.executable,
            plotting_script,
            "--data_dir", data_dir,
            "--output_dir", channel_plot_dir,
        ]
        if control_substring:
            cmd += ["--control", str(control_substring)]

        log(f"Running plotting for channel '{channel_name}' via: {' '.join(cmd)}", log_file)

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            log(f"ERROR: Plotting failed for channel '{channel_name}': {e}", log_file)
            raise

    log("Plot generation completed.", log_file)


def run_correlation_plotting(config, correlations_dir, plot_dir, log_file):
    """
    Run the correlation plotting script using merged correlation CSV files as input.

    This function calls the standalone correlation plotting script (specified in config)

    Args:
        config (dict): Must contain "correlation_plotting_script" path.
        correlations_dir (str): Directory containing merged correlation CSV files.
        plot_dir (str): Directory to save *all* generated plots.
        log_file (str): Path to the log file (optional).
    """

    correlation_plotting_script = config.get("correlation_plotting_script")
    if not correlation_plotting_script:
        raise ValueError("Missing 'correlation_plotting_script' in config")

    # Make one clean output folder
    os.makedirs(plot_dir, exist_ok=True)
    log(f"Running correlation plots on {correlations_dir}, output: {plot_dir}")

    # Single subprocess call:
    subprocess.run([
        "python3", correlation_plotting_script,
        "--data_dir", correlations_dir,
        "--output_dir", plot_dir
    ], check=True)

    #check for completion
    merged_csv_files = sorted(glob.glob(os.path.join(correlations_dir, "*.csv")))
    if not merged_csv_files:
        log(f"No CSVs found in {correlations_dir}, skipping.")
        return
        
    bases = [os.path.splitext(os.path.basename(p))[0] for p in merged_csv_files]

    missing = []

    # per-CSV histograms
    for b in bases:
        for ext in (".tif", ".svg"):
            path = os.path.join(plot_dir, f"{b}_histogram{ext}")
            if not os.path.exists(path):
                missing.append(f"{b}_histogram{ext}")

    # combined plots
    combined = [
        "all_conditions_bootstrap_kde",
        "all_conditions_plain_kde",
    ]



    for name in combined:
        for ext in (".tif", ".svg"):
            path = os.path.join(plot_dir, f"{name}{ext}")
            if not os.path.exists(path):
                missing.append(f"{name}{ext}")

    if missing:
        log("ERROR: Missing plot files:")
        for m in missing:
            log(f"  • {m}")
        raise RuntimeError(f"Plot generation incomplete: {len(missing)} files missing.")
    else:
        log("Correlation plot generation completed successfully — all files present.")

def process_pipeline(config, steps_to_run=None, no_subfolders=False, preset=None):
    """
    Process the full pipeline in the following order:
    1. Perform max projection.
    2. Split channels.
    3. Optionally perform primary background subtraction.
    4. Optionally normalize and filter the DNA channel if not using cpsam.
    5. Run Cellpose segmentation.
    6. Filter Cellpose masks based on size and sphericity.
    7. Generate segmentation overlays.
    8. Optionally perform secondary background subtraction.
    9. Quantify labels in each channel.
    10. Generate plots.
    11. Merge CSV files by condition.
    12. Run correlation analysis.
    13. Merge correlation CSV files.
    14. Run correlation plotting.
    
    
    Args:
        config (dict): Configuration dictionary containing pipeline parameters.
        steps_to_run (list, optional): List of steps to execute. If None, all steps are executed.
    """
    output_dir = config["output_directory"]
    log_file = os.path.join(output_dir, "pipeline_log.txt")
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # Persistent metadata cache (disk)
    # -----------------------------------------------------------------
    global METADATA_CACHE_DIR
    METADATA_CACHE_DIR = os.path.join(output_dir, "metadata_cache")
    os.makedirs(METADATA_CACHE_DIR, exist_ok=True)
    log(f"Metadata cache directory: {METADATA_CACHE_DIR}", log_file)

    max_proj_dir = os.path.join(output_dir, "max_projection")
    split_channels_dir = os.path.join(output_dir, "split_channels")
    bg_subtracted_dir = os.path.join(output_dir, "bg_subtracted_channels")
    dna_filtered_dir = os.path.join(output_dir, "filtered_dna")
    masks_dir = os.path.join(dna_filtered_dir, "masks")
    seg_overlay_dir = os.path.join(output_dir, "segmentation_overlay")
    plot_dir = os.path.join(output_dir, "plots")
    correlations_dir = os.path.join(output_dir, "correlations")
    os.makedirs(correlations_dir, exist_ok=True)
    
    # Determine which folder to use as input for secondary bg subtraction:
    # If primary background subtraction is enabled, use that; otherwise, use split_channels.
    use_primary_bg_subtraction = config.get("use_background_subtraction", False)
    
    background_image_path = config.get("background_image_path", None)

    # Load the dark image once if background subtraction is enabled
    dark_image = None
    if use_primary_bg_subtraction:
        try:
            _, dark_image_stack, dark_metadata = universal_read_tiff(background_image_path, log_file=log_file)
            dark_image = dark_image_stack[0, 0, :, :]  # Extract the single 2D slice
            log(f"Loaded dark image from {background_image_path}.", log_file)
        except Exception as e:
            log(f"Error loading background image: {e}", log_file)
            raise

    sec_bg_input_dir = bg_subtracted_dir if use_primary_bg_subtraction else split_channels_dir
    # Folder for secondary (measured) background subtracted images.
    measured_bg_dir = os.path.join(output_dir, "measured_background_subtracted_images")

    # Now decide what folder to feed into the quantification step:
    # If secondary bg subtraction is used, use measured_bg_dir;
    # else if primary bg subtraction is enabled, use bg_subtracted_dir;
    # otherwise, use split_channels_dir.
    if config.get("use_secondary_background_subtraction", False):
        input_for_quant = measured_bg_dir
    else:
        input_for_quant = bg_subtracted_dir if config.get("use_background_subtraction", False) else split_channels_dir

    #decide whether using cpsam or filtering dna to generate masks for subsequent mask directory location:
    mask_base_dir = (
        split_channels_dir
    )

    # Find the channel name corresponding to the DNA channel index
    channel_name = next(name for name, index in config["channels"].items() if index == config["channels"]["DNA"])

    cp_mask_dir = (
        os.path.join(mask_base_dir, channel_name, "masks")
        if config.get("use_cpsam", False)
        else os.path.join(dna_filtered_dir, "masks")
    )
    write_verified_run_log(output_dir, config, steps_to_run=steps_to_run, preset=preset, log_file=log_file)
    steps = {
        "max_projection": lambda: max_projection_raw_images(
            config["input_directory"], max_proj_dir, log_file, no_subfolders=no_subfolders
        ),
        "split_channels": lambda: [
            split_channels(
                result["stack"],
                result["metadata"],
                config["channels"],
                split_channels_dir,
                result["subfolder"],
                log_file,
            )
            for result in combined_reading(config["input_directory"], log_file, no_subfolders=no_subfolders)
        ],
        "background_subtraction": lambda: [
            subtract_background(
                split_channel_image=os.path.join(split_channels_dir, f"{channel_name}", f"C{config['channels'][channel_name]}-{result['subfolder']}.tif"),
                dark_image=dark_image,  # assume dark_image is previously loaded
                channels=config["channels"],
                channel_name=channel_name,
                output_dir=bg_subtracted_dir,
                subfolder=result["subfolder"],
                metadata=result["metadata"],
                log_file=log_file,
            )
            for result in combined_reading(config["input_directory"], log_file, no_subfolders=no_subfolders)
            for channel_name in config["channels"].keys()
        ] if use_primary_bg_subtraction else None,
        "normalize_filter_dna": lambda: (
            None if config.get("use_cpsam", False)
            else normalize_and_filter_dna_channel(
                bg_subtracted_dir if use_primary_bg_subtraction else split_channels_dir,
                dna_filtered_dir,
                config["median_filter_radius"],
                config["gaussian_filter_radius"],
                log_file,
            )
        ),
        "cellpose_segmentation": lambda: run_cellpose(
            config["cellpose_bash_script"],
            split_channels_dir,
            dna_filtered_dir,
            config.get("use_cpsam", False),
            log_file
        ),
        "filter_masks": lambda: filter_masks(
            mask_dir=(cp_mask_dir),
            output_dir=config["output_directory"],
            voxel_threshold=config.get("voxel_threshold"),
            sphericity_threshold=config.get("sphericity_threshold"),
            log_file=log_file
        ),
        "generate_segmentation_overlay": lambda: process_raw_and_masks(
            os.path.join(bg_subtracted_dir if use_primary_bg_subtraction else split_channels_dir, "DNA"),
            os.path.join(output_dir, "segmented_nuclei"),
            seg_overlay_dir
        ),
        # Secondary background subtraction (global per-channel value).
        # Step 1: Estimate per-image secondary background values and compute a global median per channel.
        "estimate_secondary_background": lambda: (
            estimate_secondary_background(
                input_dir=sec_bg_input_dir,
                cp_mask_dir=cp_mask_dir,
                output_dir=output_dir,
                channels=config["channels"],
                roi_size=(50, 50),
                pad=10,
                log_file=log_file
            ) if config.get("use_secondary_background_subtraction", False) else None
        ),
        # Step 2: Apply the global secondary background value per channel and write bg-subtracted images.
        "secondary_background_subtraction": lambda: (
            secondary_background_subtraction(
                input_dir=sec_bg_input_dir,
                cp_mask_dir=cp_mask_dir,
                output_dir=measured_bg_dir,
                channels=config["channels"],
                roi_size=(50, 50),
                pad=10,
                log_file=log_file
            ) if config.get("use_secondary_background_subtraction", False) else None
        ),
        "quantify_labels": lambda: quantify_labels_per_channel(
            config,
            config["input_directory"],
            os.path.join(output_dir, "segmented_nuclei"),
            input_for_quant,
            output_dir,
            log_file,
        ),

        # ----------------------------------
        # Analysis pre-sets for re-running 
        # ----------------------------------
        "analyze_quant": lambda: (
            quantify_labels_per_channel(
                config,
                config["input_directory"],
                os.path.join(output_dir, "segmented_nuclei"),
                input_for_quant,
                output_dir,
                log_file,
            ),
            run_plotting(
                config=config,
                output_dir=output_dir,
                plot_dir=plot_dir,
                log_file=log_file
            )
        ),

        "correlation_analysis": lambda: correlate_labels_per_channel(
            config,
            config["input_directory"],
            os.path.join(output_dir, "segmented_nuclei"),
            (input_for_quant),
            correlations_dir,
            log_file,
        ),

        "analyze_correlation": lambda: (
            correlate_labels_per_channel(
                config,
                config["input_directory"],
                os.path.join(output_dir, "segmented_nuclei"),
                (input_for_quant),
                correlations_dir,
                log_file,
            ),
            merge_correlation_csv_files(config, correlations_dir, log_file),
            run_correlation_plotting(
                config,
                os.path.join(output_dir, "correlations", "merged_results"),
                os.path.join(output_dir, "correlation_plots"),
                log_file
            )
        )

    }
    
    if steps_to_run is None:
        # Default pipeline run order 
        steps_to_run = [
            "max_projection",
            "split_channels",
            "background_subtraction",
            "normalize_filter_dna",
            "cellpose_segmentation",
            "filter_masks",
            "generate_segmentation_overlay",
            "estimate_secondary_background",
            "secondary_background_subtraction",
            "analyze_quant",
            "analyze_correlation",
        ]
    # Write the manifest ONCE per pipeline run
    write_verified_run_log(
        output_dir,
        config,
        steps_to_run=steps_to_run,
        preset=preset,
        log_file=log_file
    )
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

def main_with_full_pipeline(config_path, steps=None, no_subfolders=False, preset=None):
    """
    Main function to run the pipeline with optional steps.

    Args:
        config_path (str): Path to the JSON configuration file.
        steps (list, optional): List of steps to execute. If None, the pipeline uses its default run order.
        no_subfolders (bool): If True, treat input_directory as a flat folder of TIFFs rather than subfolders.
        preset (str, optional): Optional pipeline preset. If provided (and steps is None), this overrides
            the default run order with a small set of commonly used stage steps.
    """
    config = load_config(config_path)

    # Presets are designed to be ergonomic shortcuts for common partial reruns.
    # They intentionally map to *analysis stage* steps (e.g., analyze_quant), not micro-steps.
    if steps is None and preset:
        preset_map = {
            "quant_only": ["analyze_quant"],
            "corr_only": ["analyze_correlation"],
            "bg_only": ["estimate_secondary_background", "secondary_background_subtraction"],
            "analysis": ["analyze_quant", "analyze_correlation"],
        }
        if preset not in preset_map:
            raise ValueError(f"Unknown preset '{preset}'. Valid presets: {sorted(preset_map.keys())}")
        steps = preset_map[preset]

    process_pipeline(config, steps, no_subfolders=no_subfolders, preset=preset)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full image processing pipeline.")
    parser.add_argument("config", help="Path to the JSON configuration file.")
    parser.add_argument(
        "--steps",
        help="Comma-separated list of steps to execute (e.g., max_projection,split_channels).",
    )
    parser.add_argument(
        "--preset",
        help="Optional pipeline preset to run (e.g., quant_only, bg_only). If provided, this overrides --steps.",
    )
    parser.add_argument(
        "--no_subfolders",
        action="store_true",
        help="Treat input_directory as a flat folder of TIFFs (supports .ome.tif, .tif, .tiff) instead of subfolders."
    )
    args = parser.parse_args()

    steps_to_run = args.steps.split(",") if args.steps else None
    main_with_full_pipeline(args.config, steps=steps_to_run, no_subfolders=args.no_subfolders, preset=args.preset)