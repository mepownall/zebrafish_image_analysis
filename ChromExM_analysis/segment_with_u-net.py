"""
Prediction script for running a trained U-Net model on 3D microscopy TIFF volumes.

The script performs slice-wise inference on volumetric image data using a
2.5D U-Net architecture with sliding-window prediction and overlap-based
blending to minimize edge artifacts.

Workflow

1. Load a trained model and associated weights (.pth).
2. For each 3D image volume:
    * Normalize image intensities.
    * Run sliding-window inference on each z-slice.
    * Combine overlapping predictions using weighted blending.
    * Generate probability maps and binary segmentation masks.
    * Apply postprocessing to remove artifacts and improve mask quality.
3. Save segmentation outputs.

Outputs:
By default, the script saves:
* Postprocessed binary masks

Optional outputs include:
* RGB segmentation overlays
* Raw segmentation masks
* Probability maps
* Intermediate feature-map visualizations for troubleshooting

The script supports GPU acceleration, mixed-precision inference, and
optional model compilation for improved performance on compatible hardware.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter, binary_fill_holes, label

from customunet_ResASPP_25D import UNetResASPP

# ------------------------------
# Performance settings
# ------------------------------

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
except Exception:
    pass

# ------------------------------
# Utilities
# ------------------------------


def normalize(img: np.ndarray) -> np.ndarray:
    """

    Normalize image intensities to the range [0, 1] using percentile clipping.

    Intensities are clipped to the 1st and 99th percentiles before scaling.

    """
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    return ((img - p1) / (p99 - p1 + 1e-5)).astype(np.float32)

def slice_contrast_p95_p5(img: np.ndarray) -> float:

    """

    Compute a robust contrast metric for a 2D image slice.

    Contrast is defined as the difference between the 95th and 80th
    intensity percentiles.

    """

    p80, p95 = np.percentile(img, (80, 95))

    return float(p95 - p80)


def postprocess_mask_stack(mask_stack: np.ndarray) -> np.ndarray:
    """
    Clean up a binary mask stack (3D volume) after model prediction.

    Pipeline:
    1. Fill holes in each 2D slice (e.g., donut-shaped nuclei).
    2. Label connected components in 3D.
    3. Keep only the largest connected component (drop speckles).

    Args:
        mask_stack (ndarray): 3D uint8 mask [Z,H,W], values {0,1}.

    Returns:
        ndarray: Cleaned 3D uint8 mask [Z, H, W].
    """
    # Step 1: Fill holes per slice
    filled_stack = np.stack([binary_fill_holes(slice_) for slice_ in mask_stack])

    # Step 2: Connected component labeling in 3D
    labeled_3d, num_features = label(filled_stack)

    if num_features == 0:
        return np.zeros_like(mask_stack)

    # Step 3: Keep only the largest connected component
    sizes = np.bincount(labeled_3d.ravel())
    sizes[0] = 0  # ignore background
    largest_label = sizes.argmax()
    cleaned_stack = (labeled_3d == largest_label).astype(np.uint8)

    return cleaned_stack


# ---- Sliding-window cache ----

class _TileCache:
    """Cache Hann windows and tile anchor positions."""

    def __init__(self) -> None:
        self._hann = {}
        self._anchors = {}

    def anchors(self, L: int, win: int, stride: int) -> list[int]:
        key = (L, win, stride)
        xs = self._anchors.get(key)
        if xs is not None:
            return xs
        if L <= win:
            xs = [0]
        else:
            xs = list(range(0, L - win + 1, stride))
            if xs[-1] != L - win:
                xs.append(L - win)
        self._anchors[key] = xs
        return xs

    def hann2d(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, device.type)
        win = self._hann.get(key)
        if win is not None and win.device == device:
            return win
        wy = torch.hann_window(h, periodic=False, dtype=torch.float32, device=device)
        wx = torch.hann_window(w, periodic=False, dtype=torch.float32, device=device)
        w2d = torch.outer(wy, wx)
        w2d /= (w2d.max() + 1e-8)
        win = w2d.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        self._hann[key] = win
        return win


_TILE_CACHE = _TileCache()


# ------------------------------
# Prediction
# ------------------------------

def predict_slice(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    *,
    patch_size: int = 1024,
    stride: int = 512,
    threshold: float = 0.5,
    smooth_sigma: float = 1.0,
    return_prob: bool = True,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Run sliding-window prediction on a single 2D slice using overlapping patches
    with Hann weighting to avoid seams.

    Args:
        model: trained UNetResASPP.
        tensor: [1,1,H,W] normalized slice on *device*.
        patch_size: window size for inference (e.g., 1024).
        stride: step between windows (e.g., 512).
        threshold: cutoff for binary mask (after sigmoid).
        smooth_sigma: optional Gaussian smoothing of prob map.
        return_prob: if True, return (mask, prob_map); else mask only.
        device: torch.device; if None, use tensor.device.

    Returns:
        mask (ndarray uint8) [H,W], and prob_map [H,W] in [0,1] if requested.
    """
    device = device or tensor.device
    model = model.to(device)

    _, C, H, W = tensor.shape
    ph = pw = patch_size

    pred_sum = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)
    weight_map = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)

    window_full = _TILE_CACHE.hann2d(ph, pw, device)

    ys = _TILE_CACHE.anchors(H, ph, stride)
    xs = _TILE_CACHE.anchors(W, pw, stride)

    with torch.inference_mode():
        use_cuda = (device.type == "cuda")
        amp_enabled = use_cuda
        # Prefer bf16 on modern GPUs; fall back to fp16; otherwise fp32
        if use_cuda:
            try:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except Exception:
                dtype = torch.float16
            ctx = torch.amp.autocast("cuda", enabled=amp_enabled, dtype=dtype)
        else:
            ctx = torch.amp.autocast("cpu", enabled=False)
        with ctx:
            for i in ys:
                for j in xs:
                    patch = tensor[:, :, i : i + ph, j : j + pw]
                    out = model(patch)  # expects sigmoid output [1,1,ph,pw]
                    pred_sum[:, :, i : i + ph, j : j + pw] += out * window_full
                    weight_map[:, :, i : i + ph, j : j + pw] += window_full

    pred_avg = pred_sum / (weight_map + 1e-8)
    pred_avg = torch.clamp(pred_avg, 0.0, 1.0)

    pred_np = pred_avg.squeeze().detach().cpu().numpy()
    if smooth_sigma and smooth_sigma > 0:
        pred_np = gaussian_filter(pred_np, sigma=smooth_sigma)

    binary = (pred_np > threshold).astype(np.uint8)
    return (binary, pred_np) if return_prob else binary


def overlay_mask(raw_slice: np.ndarray, mask: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Create an RGB overlay of a binary mask on top of a grayscale raw slice."""
    m = raw_slice.max()
    raw_norm = (raw_slice / (m + 1e-8) * 255).astype(np.uint8)
    rgb = np.stack([raw_norm] * 3, axis=-1)
    overlay = rgb.copy()
    overlay[mask == 1] = [255, 0, 0]  # red mask
    return (rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)


# ------------------------------
# Feature map utilities
# ------------------------------

def save_feature_map(tensor: torch.Tensor, save_path: Path, max_channels: int = 8, cmap: str = "viridis") -> None:
    """Save selected feature-map channels as a PNG image."""
    import matplotlib.pyplot as plt

    fmap = tensor[0].detach().cpu().numpy()  # [C,H,W]
    C = fmap.shape[0]
    n_show = min(C, max_channels)

    fig, axs = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1:
        axs = [axs]
    for i in range(n_show):
        ch = fmap[i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)  # normalize 0–1
        axs[i].imshow(ch, cmap=cmap)
        axs[i].set_title(f"ch {i}")
        axs[i].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)


def save_feature_overlays(
    tensor: torch.Tensor,
    raw_slice: np.ndarray,
    save_path: Path,
    k: int = 4,
    cmap: str = "magma",
) -> None:
    """
    Overlay selected high-activation feature channels onto the corresponding raw slice.

    Feature channels are ranked by mean activation. Channels are resized to match
    the raw image dimensions when needed.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    fmap = tensor[0].detach().cpu().numpy()  # [C, h, w]
    C, h, w = fmap.shape

    H, W = raw_slice.shape
    means = fmap.reshape(C, -1).mean(axis=1)
    top_idx = np.argsort(means)[-k:][::-1]

    raw_norm = (raw_slice / (raw_slice.max() + 1e-8)).astype(np.float32)

    fig, axs = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axs = [axs]

    for j, c in enumerate(top_idx):
        ch = fmap[c]
        if ch.shape != (H, W):
            sy = H / ch.shape[0]
            sx = W / ch.shape[1]
            ch = zoom(ch, (sy, sx), order=1)
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        overlay = 0.6 * raw_norm + 0.4 * ch
        axs[j].imshow(overlay, cmap=cmap, vmin=0, vmax=1)
        axs[j].set_title(f"top{j+1} ch{int(c)} (mean={means[c]:.3f})")
        axs[j].axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)


def dump_feature_maps_for_slice(
    model: torch.nn.Module,
    vol: np.ndarray,
    z_index: int,
    output_dir: Path,
    *,
    max_channels: int = 8,
    k_overlay: int = 4,
    vis_scale: float = 0.5,
    force_cpu: bool = False,
) -> None:
    """
    Save intermediate feature-map visualizations for a single z-slice.

    Forward hooks are registered on selected model layers, activations are copied
    to CPU, and feature-map grids and overlays are written to disk. Hooks are
    removed before the function returns.
    """
    import torch.nn.functional as F

    output_dir.mkdir(parents=True, exist_ok=True)
    if not (0 <= z_index < vol.shape[0]):
        print(f"[feature dump] z_index {z_index} out of bounds (0..{vol.shape[0]-1})")
        return

    device_orig = next(model.parameters()).device
    use_cpu = force_cpu or (device_orig.type != "cuda")

    feats: dict[str, torch.Tensor] = {}

    def hook_down1(_, __, out):
        t = out.detach()
        if vis_scale and vis_scale < 1.0:
            t = F.interpolate(t, scale_factor=vis_scale, mode="area")
        feats["down1"] = t.float().cpu()

    def hook_aspp(_, __, out):
        t = out.detach()
        if vis_scale and vis_scale < 1.0:
            t = F.interpolate(t, scale_factor=vis_scale, mode="area")
        feats["aspp"] = t.float().cpu()

    h1 = model.down1.register_forward_hook(hook_down1)
    hA = model.aspp.register_forward_hook(hook_aspp)

    raw_full = vol[z_index]
    raw = raw_full
    if vis_scale and vis_scale < 1.0:
        r = torch.from_numpy(raw_full).unsqueeze(0).unsqueeze(0).to(torch.float32)
        r = F.interpolate(r, scale_factor=vis_scale, mode="area").squeeze(0).squeeze(0).numpy()
        raw = r

    if use_cpu and device_orig.type == "cuda":
        model_cpu = model.to("cpu")
        run_device = torch.device("cpu")
    else:
        model_cpu = model
        run_device = device_orig

    x = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(torch.float32).to(run_device)

    with torch.inference_mode():
        if run_device.type == "cuda":
            try:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except Exception:
                dtype = torch.float16
            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                _ = model_cpu(x)
        else:
            _ = model_cpu(x)

    if "down1" in feats:
        save_feature_map(
            feats["down1"], output_dir / f"down1_slice{z_index}.png", max_channels=max_channels
        )
        save_feature_overlays(
            feats["down1"], raw_full, output_dir / f"down1_slice{z_index}_overlays.png", k=k_overlay
        )
    if "aspp" in feats:
        save_feature_map(
            feats["aspp"], output_dir / f"aspp_slice{z_index}.png", max_channels=max_channels
        )
        save_feature_overlays(
            feats["aspp"], raw_full, output_dir / f"aspp_slice{z_index}_overlays.png", k=k_overlay
        )

    h1.remove(); hA.remove()

    if use_cpu and device_orig.type == "cuda":
        model_cpu.to(device_orig)

    if device_orig.type == "cuda":
        torch.cuda.empty_cache()

def get_25d_slice(vol_norm: np.ndarray, z: int) -> np.ndarray:
    """
    Construct a 2.5D input centered on one z-slice.

    The output contains three channels corresponding to z-1, z, and z+1,
    with edge slices clamped at the volume boundaries.
    """
    Z = vol_norm.shape[0]
    z0 = max(0, z - 1)
    z1 = z
    z2 = min(Z - 1, z + 1)
    return np.stack([vol_norm[z0], vol_norm[z1], vol_norm[z2]], axis=0).astype(np.float32)

# ------------------------------
# Volume runner
# ------------------------------

def run_on_volume(
    img_path: Path,
    model: torch.nn.Module,
    patch_size: int,
    stride_frac: float,
    threshold: float,
    device: torch.device,
    *,
    skip_overlays: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """
    Run segmentation prediction on a 3D TIFF volume.

    Each z-slice is processed using a 2.5D input constructed from the
    neighboring slices. Probability maps are thresholded to generate binary
    masks, and optional overlays can be generated for visual inspection.

    Returns
    -------
    tuple
        mask_stack, overlay_stack, probability_stack, normalized_volume
    """

    vol_raw = imread(img_path)  # [Z,H,W]

    # Normalize the *entire 3D volume* once so all z-slices share the same scaling.
    vol = normalize(vol_raw)  # [Z,H,W] float32 in [0,1]

    Z = vol.shape[0]
    mask_stack, prob_stack = [], []
    overlay_stack = [] if not skip_overlays else None

    stride = max(1, int(patch_size * stride_frac))

    for z in range(Z):
        # For overlays/metrics, keep using the CENTER slice
        raw_norm = vol[z]       # [H,W] normalized slice in [0,1]
        raw_orig = vol_raw[z]   # [H,W] original raw slice

        # Model input is 2.5D: [3,H,W] = (z-1, z, z+1)
        x_np = get_25d_slice(vol, z)  # [3,H,W]

        # Build tensor for model input: [1,3,H,W]
        tensor = (
            torch.from_numpy(x_np)
            .unsqueeze(0)  # [1,3,H,W]
            .to(torch.float32)
            .pin_memory()
            .to(device, non_blocking=True)
        )

        _, prob_map = predict_slice(
            model,
            tensor,
            patch_size=patch_size,
            stride=stride,
            threshold=0.5,      
            smooth_sigma=1.0,
            return_prob=True,
            device=device,
        )

        mask = (prob_map > threshold).astype(np.uint8)
        mask_stack.append(mask)
        prob_stack.append(prob_map)

        if not skip_overlays:
            overlay_stack.append(overlay_mask(raw_norm, mask))

    mask_stack = np.stack(mask_stack)
    prob_stack = np.stack(prob_stack)
    prob_stack_16bit = (np.clip(prob_stack, 0, 1) * 65535).astype(np.uint16)

    if not skip_overlays:
        overlay_stack = np.stack(overlay_stack)
    else:
        overlay_stack = None

    return mask_stack, overlay_stack, prob_stack_16bit, vol

# ------------------------------
# Main entry point
# ------------------------------

def main(
    input_dir: Path,
    model_path: Path,
    patch_size: int = 1024,
    threshold: float = 0.5,
    *,
    output_dir: Path | None = None,  
    stride_frac: float = 0.75,
    save_feature_debug: bool = False,
    feature_slice: int = 100,
    feature_max_channels: int = 8,
    feature_k_overlay: int = 4,
    feature_vis_scale: float = 0.5,
    feature_force_cpu: bool = False,
    compile_model: bool = False,
    skip_overlays: bool = True,
    save_all: bool = False,
) -> None:
    """
    Run full inference pipeline on all .tif stacks in a folder.

    Args:
        input_dir: folder with raw .tif stacks.
        model_path: path to trained .pth file.
        patch_size: sliding window size (default 1024).
        threshold: mask cutoff (default 0.5).
        stride_frac: fraction of patch size to use as stride (default 0.75).
        save_feature_debug: if True, dump feature maps/overlays for `feature_slice`.
        feature_*: controls for feature debug outputs.
        compile_model: if True and PyTorch>=2, compile the model for speed.
        skip_overlays: if True, do not save RGB overlays (faster I/O).
        save_all: if True, also save raw mask/overlay/probability like previous behavior.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetResASPP(in_channels=3, out_channels=1)

    if compile_model:
        # Safe guard: only available on torch>=2
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("[info] model compiled with torch.compile")
        except Exception as e:
            print(f"[warn] torch.compile unavailable or failed: {e}")

    model = model.to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    input_dir = Path(input_dir)

    # Default behavior preserved if output_dir is not provided
    if output_dir is None:
        output_dir = input_dir / "nucleus_inference_output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for img_path in sorted(input_dir.glob("*.tif*")):
            print(f"Processing {img_path.name}")
            # If not saving all outputs, skip expensive per-slice raw overlays
            effective_skip_overlays = skip_overlays or (not save_all)
            mask, overlay, heatmap, vol = run_on_volume(
                img_path,
                model,
                patch_size,
                stride_frac,
                threshold,
                device,
                skip_overlays=effective_skip_overlays,
            )

            base = img_path.stem
            # By default, do NOT save raw mask/overlay/probability; restore with --save_all
            if save_all:
                imwrite(output_dir / f"{base}_mask.tif", (mask.astype(np.uint8) * 255))
                if overlay is not None:
                    imwrite(output_dir / f"{base}_overlay.tif", overlay.astype(np.uint8))
                imwrite(output_dir / f"{base}_probability.tif", heatmap)

            # Post-process and save
            mask_post = postprocess_mask_stack(mask)
            # optional generate post-processed overlay generation
            imwrite(output_dir / f"{base}_mask_postprocessed.tif", (mask_post.astype(np.uint8) * 255))

            if not skip_overlays:
                overlay_post = np.stack(
                    [overlay_mask(raw_slice, mask_slice) for raw_slice, mask_slice in zip(vol, mask_post)]
                )
                imwrite(output_dir / f"{base}_overlay_postprocessed.tif", overlay_post.astype(np.uint8))

            print(f"Saved outputs for {base}")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder with raw 3D .tif images")
    parser.add_argument("--model_path", type=Path, default=Path("unet_2d_model.pth"), help="Trained model path (.pth)")
    parser.add_argument("--patch_size", type=int, default=1024, help="Sliding window size for inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability cutoff for binary mask")

    # New: speed/perf controls
    parser.add_argument("--stride_frac", type=float, default=0.75, help="Stride as a fraction of patch size (e.g. 0.5–0.9)")
    parser.add_argument("--compile_model", action="store_true", help="Compile the model graph for speed (PyTorch 2.x)")
    parser.add_argument("--save_overlays", action="store_true", help="Save RGB overlay images in addition to segmentation masks.")

    # New: feature-debug (OFF by default)
    parser.add_argument("--save_feature_debug", action="store_true", help="If set, dump feature maps/overlays for a slice")
    parser.add_argument("--feature_slice", type=int, default=100, help="Z index (slice) for feature debug outputs")
    parser.add_argument("--feature_max_channels", type=int, default=8, help="Max feature channels to grid-plot")
    parser.add_argument("--feature_k_overlay", type=int, default=4, help="Top-k active channels to overlay on raw slice")
    parser.add_argument("--feature_vis_scale", type=float, default=0.5, help="Downsample factor for debug pass (0.25–1.0)")
    parser.add_argument("--feature_force_cpu", action="store_true", help="Run debug forward on CPU to avoid GPU use")


    # New: keep legacy writeouts only if explicitly requested
    parser.add_argument("--save_all", action="store_true", help="Also save raw mask/overlay/probability like before")

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: <input_dir>/nucleus_inference_output",
    )

    args = parser.parse_args()

    main(
        args.input_dir,
        args.model_path,
        args.patch_size,
        args.threshold,
        output_dir=args.output_dir,
        stride_frac=args.stride_frac,
        save_feature_debug=args.save_feature_debug,
        feature_slice=args.feature_slice,
        feature_max_channels=args.feature_max_channels,
        feature_k_overlay=args.feature_k_overlay,
        feature_vis_scale=args.feature_vis_scale,
        feature_force_cpu=args.feature_force_cpu,
        compile_model=args.compile_model,
        skip_overlays=not args.save_overlays,
        save_all=args.save_all,
    )