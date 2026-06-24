#!/usr/bin/env python3
"""
Masked radial autocorrelation analysis with condition-level aggregation.
This script computes radial autocorrelation curves from masked image volumes,
exports per-image autocorrelation profiles, and generates condition-level
summary plots and statistics.

Features
--------
- Computes radial autocorrelation within segmentation masks
- Supports parameter sweeps across analysis settings
- Exports per-image autocorrelation curves
- Generates condition-level summary plots
- Optional permutation control that preserves intensity distributions while
  disrupting spatial organization

Default manuscript setting
--------------------------
window_size = (2, 6, 6)
normalization = "mask"
occupancy_threshold = 0.1

Outputs
-------
OUT_DIR/
  sweep__<param_tag>/
    per_image_curves/
      <image_id>__<condition>.csv
    autocorr_by_condition_full.png
    autocorr_by_condition_full.pdf
    autocorr_by_condition_zoom.png
    autocorr_by_condition_zoom.pdf
    condition_summary.csv
    run_log.txt

"""

import os
import re
import glob
import argparse
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd
import tifffile as tiff

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Logging
# -----------------------------
def log(msg, log_file=None):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


# -----------------------------
# Matching helpers
# -----------------------------
def find_matching_masks(raw_dir, mask_dir):
    def _image_id_from_raw_filename(fname: str) -> str:
        base = os.path.basename(fname)
        stem1, _ = os.path.splitext(base)
        stem2, ext2 = os.path.splitext(stem1)
        stem = stem2 if ext2.lower() == ".ome" else stem1
        stem = stem.rstrip(".")
        if stem.startswith("C") and "-" in stem:
            left, right = stem.split("-", 1)
            if left[1:].isdigit():
                stem = right
        return stem

    raw_patterns = ["*.ome.tif", "*.tif", "*.tiff", "*.OME.TIF", "*.TIF", "*.TIFF"]
    raw_images = []
    for pat in raw_patterns:
        raw_images.extend(glob.glob(os.path.join(raw_dir, pat)))
    raw_images = sorted(set(raw_images))

    mask_patterns = ["*.ome.tif", "*.tif", "*.tiff", "*.OME.TIF", "*.TIF", "*.TIFF"]
    mask_images = []
    for pat in mask_patterns:
        mask_images.extend(glob.glob(os.path.join(mask_dir, pat)))
    mask_images = sorted(set(mask_images))

    matches = []
    if not raw_images or not mask_images:
        return matches

    mask_basenames = [(m, os.path.basename(m)) for m in mask_images]

    for raw_image in raw_images:
        image_id = _image_id_from_raw_filename(raw_image)
        match_path = None
        for mpath, mbase in mask_basenames:
            if image_id in mbase:
                match_path = mpath
                break
        if match_path is not None:
            matches.append((raw_image, match_path))

    return matches


def infer_condition_from_image_id(image_id: str) -> str:
    """
    Infer the experimental condition from an image identifier.

    The condition is inferred from the token preceding the embryo identifier
    (for example E1, E2, n1, or n2), while skipping common metadata tokens.
    """
    s = re.sub(r"_sphericity_filtered$", "", image_id)
    toks = s.split("_")

    anchor = None
    for i, t in enumerate(toks):
        if re.fullmatch(r"E\d+", t, flags=re.IGNORECASE):
            anchor = i
            break

    if anchor is None:
        for i, t in enumerate(toks):
            if re.fullmatch(r"n\d+", t, flags=re.IGNORECASE):
                anchor = i
                break

    if anchor is None or anchor == 0:
        return toks[-1]

    ignore_tokens = {
        "stitched",
    }

    j = anchor - 1
    while j >= 0:
        t = toks[j]
        t_norm = t.strip().lower()

        if t_norm in ignore_tokens:
            j -= 1
            continue

        if re.fullmatch(r"(rep|r)\d+", t, flags=re.IGNORECASE):
            j -= 1
            continue

        if re.fullmatch(r"\d+(um|nm|mm|x|pct|percent)", t, flags=re.IGNORECASE):
            j -= 1
            continue

        return t

    return toks[-1]

def parse_image_id_from_raw_path(raw_path: str) -> str:
    base = os.path.basename(raw_path)
    stem1, _ = os.path.splitext(base)
    stem2, ext2 = os.path.splitext(stem1)
    stem = stem2 if ext2.lower() == ".ome" else stem1
    stem = stem.rstrip(".")
    if stem.startswith("C") and "-" in stem:
        left, right = stem.split("-", 1)
        if left[1:].isdigit():
            stem = right
    return stem


def _read_tiff_as_zyx(path: str) -> np.ndarray:
    """
    Read a TIFF volume and return a 3D array with shape (Z, Y, X).

    The function first attempts to load the volume directly with tifffile.
    If a valid 3D volume cannot be obtained, TIFF pages are stacked
    explicitly as Z planes.

    Returns
    -------
    np.ndarray
        Volume with shape (Z, Y, X).

    Raises
    ------
    RuntimeError
        If a valid 3D volume cannot be constructed.

    """
    # Fast path
    arr = tiff.imread(path)

    # Already a 3D volume
    if isinstance(arr, np.ndarray) and arr.ndim == 3:
        return arr

    # Remove singleton dimensions if present
    if isinstance(arr, np.ndarray) and arr.ndim == 4:
        # Squeeze singleton axes conservatively
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 3:
            return squeezed

    # Fall back to page-wise stacking
    with tiff.TiffFile(path) as tf:
        pages = tf.pages
        if len(pages) == 0:
            raise RuntimeError(f"No TIFF pages found in: {path}")

        # Read pages as individual Z planes
        planes = []
        for p in pages:
            a = p.asarray()
            if a.ndim != 2:
                raise RuntimeError(f"Non-2D TIFF page encountered in {path}: page shape={a.shape}")
            planes.append(a)

        vol = np.stack(planes, axis=0)  # (Z, Y, X)
        return vol


def read_raw_and_mask_with_z_fix(
    raw_path: str,
    mask_path: str,
    *,
    log_file: str | None = None,
    allow_z_crop: bool = True
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Read raw and mask volumes and resolve minor Z-dimension mismatches.

    Volumes must have matching XY dimensions. If the Z dimensions differ and
    allow_z_crop is True, both volumes are cropped to their overlapping Z range.

    Returns
    -------
    (tuple[np.ndarray, np.ndarray]) or (None, None)
        Raw and mask volumes with matching dimensions, or None on failure.

    """
    try:
        raw = _read_tiff_as_zyx(raw_path)
    except Exception as e:
        log(f"FAILED reading raw (robust): {raw_path} ({e})", log_file)
        return None, None

    try:
        mask = _read_tiff_as_zyx(mask_path)
    except Exception as e:
        log(f"FAILED reading mask (robust): {mask_path} ({e})", log_file)
        return None, None

    if raw.ndim != 3 or mask.ndim != 3:
        log(f"SKIP (not 3D after robust read): raw.ndim={raw.ndim} mask.ndim={mask.ndim} id={raw_path}", log_file)
        return None, None

    # Check XY dimensions
    if raw.shape[1:] != mask.shape[1:]:
        log(f"SKIP (XY mismatch): raw={raw.shape} mask={mask.shape} raw={raw_path} mask={mask_path}", log_file)
        return None, None

    # Crop to overlapping Z range
    if raw.shape[0] != mask.shape[0]:
        if not allow_z_crop:
            log(f"SKIP (Z mismatch, crop disabled): raw={raw.shape} mask={mask.shape} raw={raw_path} mask={mask_path}",
                log_file)
            return None, None

        zmin = min(raw.shape[0], mask.shape[0])
        log(f"WARNING Z mismatch: cropping to overlap Z={zmin} "
            f"(rawZ={raw.shape[0]} maskZ={mask.shape[0]}) id={os.path.basename(raw_path)}", log_file)

        raw = raw[:zmin, :, :]
        mask = mask[:zmin, :, :]

    return raw, mask

# -----------------------------
# Normalization
# -----------------------------
def normalize_volume_once(raw_zyx: np.ndarray,
                          mask_zyx: np.ndarray | None,
                          p_low: float = 1.0,
                          p_high: float = 99.0):
    raw_f = raw_zyx.astype(np.float32)
    if mask_zyx is not None:
        m = mask_zyx.astype(bool)
        v = raw_f[m]
    else:
        v = raw_f.reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(raw_f, dtype=np.float32), 0.0, 1.0
    lo, hi = np.percentile(v, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(raw_f, dtype=np.float32), float(lo), float(hi)
    out = (raw_f - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out, float(lo), float(hi)


# -----------------------------
# FFT autocorrelation + radial averaging
# -----------------------------
def autocorr2d_fft(img2d: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(img2d)
    ac = np.fft.ifft2(f * np.conj(f)).real
    return ac


def radial_average_centered(ac2d: np.ndarray, max_r: int) -> tuple[np.ndarray, np.ndarray]:
    ac = np.fft.fftshift(ac2d)
    h, w = ac.shape
    cy = h // 2
    cx = w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)

    mask = rr <= max_r
    r_flat = rr[mask].ravel()
    ac_flat = ac[mask].ravel()

    sums = np.bincount(r_flat, weights=ac_flat, minlength=max_r + 1)
    counts = np.bincount(r_flat, minlength=max_r + 1)

    prof = sums / np.maximum(counts, 1)
    r = np.arange(max_r + 1, dtype=np.int32)
    return r, prof.astype(np.float32)


# -----------------------------
# Permutation control
# -----------------------------
def permute_within_mask(img2d: np.ndarray, mask2d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img2d.copy()
    vals = out[mask2d]
    if vals.size <= 1:
        return out
    rng.shuffle(vals)
    out[mask2d] = vals
    return out


# -----------------------------
# Slice curve
# -----------------------------
def compute_slice_curve(norm_slice: np.ndarray,
                        mask_slice: np.ndarray | None,
                        *,
                        mean_mode: str,
                        vol_mean: float | None,
                        use_mask_correction: bool,
                        max_r_px: int,
                        threshold_quantile: float | None,
                        permute: bool,
                        rng: np.random.Generator | None):
    """
    Compute the radial autocorrelation curve for one 2D slice.

    Returns radius values, normalized autocorrelation values, and radial weights.
    """
    img = norm_slice.astype(np.float32)

    if threshold_quantile is not None:
        finite = img[np.isfinite(img)]
        if finite.size:
            thr = float(np.quantile(finite, threshold_quantile))
            img = np.where(img >= thr, img, 0.0).astype(np.float32)

    if mask_slice is None:
        if mean_mode == "slice":
            finite = img[np.isfinite(img)]
            mu = float(np.mean(finite)) if finite.size else 0.0
        else:
            mu = float(vol_mean) if vol_mean is not None else 0.0
        img0 = (img - mu).astype(np.float32)

        ac_num = autocorr2d_fft(img0)
        r, prof = radial_average_centered(ac_num, max_r=max_r_px)

        ones = np.ones_like(img0, dtype=np.float32)
        ac_den = autocorr2d_fft(ones)
        _, den_prof = radial_average_centered(ac_den, max_r=max_r_px)
        weights = den_prof.astype(np.float32)

        if prof[0] != 0:
            prof = prof / prof[0]
        return r, prof.astype(np.float32), weights

    m = mask_slice.astype(bool)
    if not np.any(m):
        return None, None, None

    if permute:
        if rng is None:
            raise ValueError("permute=True requires rng")
        img = permute_within_mask(img, m, rng)

    if mean_mode == "slice":
        mu = float(np.mean(img[m])) if np.any(m) else 0.0
    else:
        mu = float(vol_mean) if vol_mean is not None else (float(np.mean(img[m])) if np.any(m) else 0.0)

    img0 = (img - mu).astype(np.float32)

    if use_mask_correction:
        s = (img0 * m.astype(np.float32)).astype(np.float32)
        ac_num = autocorr2d_fft(s)
        ac_den = autocorr2d_fft(m.astype(np.float32))

        r, num_prof = radial_average_centered(ac_num, max_r=max_r_px)
        _, den_prof = radial_average_centered(ac_den, max_r=max_r_px)
        weights = den_prof.astype(np.float32)

        with np.errstate(divide="ignore", invalid="ignore"):
            prof = np.where(den_prof > 0, num_prof / den_prof, 0.0).astype(np.float32)

        if prof[0] != 0:
            prof = prof / prof[0]
        return r, prof.astype(np.float32), weights

    # Masked autocorrelation without mask correction
    s = (img0 * m.astype(np.float32)).astype(np.float32)
    ac_num = autocorr2d_fft(s)
    r, prof = radial_average_centered(ac_num, max_r=max_r_px)

    ac_den = autocorr2d_fft(m.astype(np.float32))
    _, den_prof = radial_average_centered(ac_den, max_r=max_r_px)
    weights = den_prof.astype(np.float32)

    if prof[0] != 0:
        prof = prof / prof[0]
    return r, prof.astype(np.float32), weights


def aggregate_slices(curves):
    """
    Aggregate per-slice autocorrelation curves using radial weights.
    """
    if not curves:
        return None, None
    r0 = curves[0][0]
    L = len(r0)
    num = np.zeros(L, dtype=np.float64)
    den = np.zeros(L, dtype=np.float64)
    for r, c, w in curves:
        if r is None or c is None or w is None:
            continue
        if len(r) != L:
            continue
        num += c.astype(np.float64) * w.astype(np.float64)
        den += w.astype(np.float64)
    c_agg = np.where(den > 0, num / den, np.nan)
    return r0, c_agg.astype(np.float32)


# -----------------------------
# Sweep parameter definition
# -----------------------------
@dataclass(frozen=True)
class ParamSet:
    mean_mode: str                  # "volume" or "slice"
    threshold_quantile: float | None
    ignore_mask: bool
    mask_correction: bool
    permute_control: bool

    def tag(self) -> str:
        tq = "none" if self.threshold_quantile is None else f"q{self.threshold_quantile:.2f}"
        mm = self.mean_mode
        im = "ignoreMask" if self.ignore_mask else "useMask"
        mc = "maskCorr" if self.mask_correction else "noMaskCorr"
        pc = "permute" if self.permute_control else "real"
        return f"{pc}__{im}__{mc}__{mm}__thr-{tq}"


# -----------------------------
# Plotting
# -----------------------------
def mean_and_sem(curves):
    A = np.vstack(curves).astype(np.float64)
    mean = np.nanmean(A, axis=0)
    n = np.sum(~np.isnan(A), axis=0).astype(np.float64)
    std = np.nanstd(A, axis=0, ddof=1)
    sem = np.where(n > 0, std / np.sqrt(n), np.nan)
    return mean.astype(np.float32), sem.astype(np.float32)


def plot_conditions(out_png, out_pdf, r, cond_to_curves, title,
                    *,
                    xlim_max=None,
                    cond_to_curves_control=None):
    fig, ax = plt.subplots(figsize=(8.8, 6.8))

    for cond in sorted(cond_to_curves.keys()):
        curves = cond_to_curves[cond]
        mu, sem = mean_and_sem(curves)
        ax.plot(r, mu, label=f"{cond} (n={len(curves)})")
        ax.fill_between(r, mu - sem, mu + sem, alpha=0.2)

        if cond_to_curves_control is not None and cond in cond_to_curves_control and len(cond_to_curves_control[cond]) > 0:
            mu_c, _ = mean_and_sem(cond_to_curves_control[cond])
            ax.plot(r, mu_c, linestyle="--", linewidth=1.5, alpha=0.8, label=f"{cond} permute")

    ax.set_xlabel("Distance r (pixels)")
    ax.set_ylabel("Normalized autocorrelation C(r) (C(0)=1)")
    ax.set_title(title)
    ax.axhline(0, linewidth=1)
    if xlim_max is not None:
        ax.set_xlim(0, float(xlim_max))
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


# -----------------------------
# Core: process one volume once, emit curves for many param sets
# -----------------------------
def compute_curves_for_paramsets(raw_path, mask_path, *,
                                paramsets,
                                p_low, p_high,
                                max_r_px,
                                permute_seed,
                                log_file=None):
    """
    Compute radial autocorrelation curves for a single volume across one or more
    parameter sets.

    The raw image and mask are loaded and normalized once, then reused for all
    parameter sets to avoid redundant I/O and preprocessing.
    """
    image_id = parse_image_id_from_raw_path(raw_path)
    condition = infer_condition_from_image_id(image_id)

    raw, mask = read_raw_and_mask_with_z_fix(raw_path, mask_path, log_file=log_file, allow_z_crop=True)
    if raw is None or mask is None:
        return None

    mask_bin = (mask > 0)

    norm, lo, hi = normalize_volume_once(raw, mask_bin, p_low=p_low, p_high=p_high)

    inside = norm[mask_bin]
    vol_mean_in_mask = float(np.mean(inside)) if inside.size else 0.0
    vol_mean_no_mask = float(np.mean(norm[np.isfinite(norm)])) if norm.size else 0.0

    z_has = np.where(mask_bin.reshape(mask_bin.shape[0], -1).any(axis=1))[0]
    if z_has.size == 0:
        log(f"SKIP (no nucleus slices): {image_id}", log_file)
        return None

    norm_slices = [norm[z] for z in z_has.tolist()]
    mask_slices = [mask_bin[z] for z in z_has.tolist()]

    out = []
    for ps in paramsets:
        curves = []
        rng = None
        if ps.permute_control:
            seed = (permute_seed + (hash(image_id) & 0xFFFFFFFF) + hash(ps.tag())) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)

        for img2d, m2d in zip(norm_slices, mask_slices):
            if ps.ignore_mask:
                r, c, w = compute_slice_curve(
                    img2d, None,
                    mean_mode=ps.mean_mode,
                    vol_mean=vol_mean_no_mask,
                    use_mask_correction=False,
                    max_r_px=max_r_px,
                    threshold_quantile=ps.threshold_quantile,
                    permute=False,
                    rng=None,
                )
            else:
                r, c, w = compute_slice_curve(
                    img2d, m2d,
                    mean_mode=ps.mean_mode,
                    vol_mean=vol_mean_in_mask,
                    use_mask_correction=ps.mask_correction,
                    max_r_px=max_r_px,
                    threshold_quantile=ps.threshold_quantile,
                    permute=ps.permute_control,
                    rng=rng,
                )
            if r is None:
                continue
            curves.append((r, c, w))

        if not curves:
            continue

        r_img, c_img = aggregate_slices(curves)
        if r_img is None:
            continue

        out.append({
            "paramset": ps,
            "image_id": image_id,
            "condition": condition,
            "raw_path": raw_path,
            "mask_path": mask_path,
            "norm_lo": lo,
            "norm_hi": hi,
            "n_slices_used": len(curves),
            "r_px": r_img,
            "c_r": c_img,
        })

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Masked radial autocorrelation for ChromExM.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-m", "--masks", required=True)
    parser.add_argument("-o", "--output", required=True)

    parser.add_argument("--max-r-px", type=int, default=200)
    parser.add_argument("--zoom-r-px", type=int, default=15)

    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.0)

    parser.add_argument("--sweep-thresholds", default="none",
                        help="Comma-separated threshold quantiles. Manuscript default: none.")
    parser.add_argument("--sweep-mean-modes", default="volume",
                        help="Comma-separated mean-subtraction modes (volume or slice). Manuscript default: volume.")

    parser.add_argument("--also-ignore-mask", action="store_true")
    parser.add_argument("--no-mask-correction", action="store_true")

    parser.add_argument("--permute-control", action="store_true")
    parser.add_argument("--permute-seed", type=int, default=0)

    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    thr_tokens = [t.strip() for t in args.sweep_thresholds.split(",") if t.strip()]
    thresholds = []
    for t in thr_tokens:
        if t.lower() in ("none", "null", "na"):
            thresholds.append(None)
        else:
            thresholds.append(float(t))

    mean_modes = [m.strip() for m in args.sweep_mean_modes.split(",") if m.strip()]
    for m in mean_modes:
        if m not in ("volume", "slice"):
            raise ValueError(f"Bad mean mode: {m}")

    ignore_mask_choices = [False] + ([True] if args.also_ignore_mask else [])
    mask_corr = not args.no_mask_correction

    base_paramsets = []
    for mm in mean_modes:
        for tq in thresholds:
            for im in ignore_mask_choices:
                base_paramsets.append(ParamSet(
                    mean_mode=mm,
                    threshold_quantile=tq,
                    ignore_mask=im,
                    mask_correction=(mask_corr and (not im)),
                    permute_control=False,
                ))

    control_paramsets = []
    if args.permute_control:
        for ps in base_paramsets:
            if ps.ignore_mask:
                continue
            control_paramsets.append(ParamSet(
                mean_mode=ps.mean_mode,
                threshold_quantile=ps.threshold_quantile,
                ignore_mask=False,
                mask_correction=ps.mask_correction,
                permute_control=True,
            ))

    all_paramsets = base_paramsets + control_paramsets

    matches = find_matching_masks(args.input, args.masks)
    if not matches:
        raise RuntimeError("No raw/mask matches found.")

    n_threads = args.threads or min(16, multiprocessing.cpu_count())
    log(f"Found {len(matches)} matched pairs. Using {n_threads} threads.", None)
    log(f"Sweep paramsets: {len(base_paramsets)} real"
        + (f" + {len(control_paramsets)} permute" if args.permute_control else ""), None)

    curves_by_param = {}
    curves_by_param_control = {}
    summary_rows_by_param = {}

    def _worker(pair):
        raw_path, mask_path = pair
        return compute_curves_for_paramsets(
            raw_path, mask_path,
            paramsets=all_paramsets,
            p_low=args.p_low, p_high=args.p_high,
            max_r_px=args.max_r_px,
            permute_seed=args.permute_seed,
            log_file=None,
        )

    results_flat = []
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for out in ex.map(_worker, matches):
            if out:
                results_flat.extend(out)

    if not results_flat:
        raise RuntimeError("No valid results produced.")

    r_ref = results_flat[0]["r_px"]

    # Collect all results
    for rec in results_flat:
        ps = rec["paramset"]
        ptag = ps.tag()
        cond = rec["condition"]
        c_r = rec["c_r"]

        if ps.permute_control:
            curves_by_param_control.setdefault(ptag, {}).setdefault(cond, []).append(c_r)
        else:
            curves_by_param.setdefault(ptag, {}).setdefault(cond, []).append(c_r)

        row = {
            "image_id": rec["image_id"],
            "condition": cond,
            "raw_path": rec["raw_path"],
            "mask_path": rec["mask_path"],
            "norm_lo": rec["norm_lo"],
            "norm_hi": rec["norm_hi"],
            "n_slices_used": rec["n_slices_used"],
        }
        summary_rows_by_param.setdefault(ptag, []).append(row)

    # Write outputs per REAL paramset
    for ps in base_paramsets:
        ptag = ps.tag()
        sweep_dir = os.path.join(args.output, f"sweep__{ptag}")
        os.makedirs(sweep_dir, exist_ok=True)
        log_file = os.path.join(sweep_dir, "run_log.txt")

        log(f"Writing outputs for {ptag}", log_file)

        # Summary CSV
        sdf = pd.DataFrame(summary_rows_by_param.get(ptag, []))
        sdf.to_csv(os.path.join(sweep_dir, "condition_summary.csv"), index=False)

        # Per-image curves ALWAYS
        per_image_dir = os.path.join(sweep_dir, "per_image_curves")
        os.makedirs(per_image_dir, exist_ok=True)
        for rec in results_flat:
            if rec["paramset"].tag() != ptag:
                continue

            suffix = "__permute" if rec["paramset"].permute_control else ""
            out_csv = os.path.join(per_image_dir, f"{rec['image_id']}__{rec['condition']}{suffix}.csv")
            pd.DataFrame({"r_px": rec["r_px"], "C_r": rec["c_r"]}).to_csv(out_csv, index=False)

        # Control curves (optional overlay)
        control_curves = None
        if args.permute_control and not ps.ignore_mask:
            ctag = ParamSet(ps.mean_mode, ps.threshold_quantile, False, ps.mask_correction, True).tag()
            control_curves = curves_by_param_control.get(ctag, None)

        cond_curves = curves_by_param.get(ptag, {})

        # Full plot
        title_full = f"Radial autocorr per nucleus (slice-wise 2D), mean±SEM by condition\n{ptag}"
        out_png = os.path.join(sweep_dir, "autocorr_by_condition_full.png")
        out_pdf = os.path.join(sweep_dir, "autocorr_by_condition_full.pdf")
        plot_conditions(out_png, out_pdf, r_ref, cond_curves, title_full,
                        xlim_max=args.max_r_px, cond_to_curves_control=control_curves)

        # Zoom plot
        title_zoom = f"Radial autocorr per nucleus (zoom), mean±SEM by condition\n{ptag}"
        out_png = os.path.join(sweep_dir, "autocorr_by_condition_zoom.png")
        out_pdf = os.path.join(sweep_dir, "autocorr_by_condition_zoom.pdf")
        plot_conditions(out_png, out_pdf, r_ref, cond_curves, title_zoom,
                        xlim_max=args.zoom_r_px, cond_to_curves_control=control_curves)

        log("Done.", log_file)

    log("All sweeps complete.", None)


if __name__ == "__main__":
    main()