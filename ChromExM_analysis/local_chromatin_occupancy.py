#!/usr/bin/env python3
"""
3D Local Chromatin Occupancy Analysis

This script measures local chromatin occupancy in 3D nuclear image volumes using a DNA intensity image and a matching 3D nuclear mask.

Chromatin is defined as voxels within the nuclear mask whose normalized DNA intensity exceeds a user-defined threshold.

Default manuscript setting
--------------------------
sliding_window_size = (2, 6, 6)
normalization = "mask"
chromatin_threshold = 0.1

Primary manuscript output
-------------------------
The manuscript analyses use sliding-window occupancy measurements. Tile-based occupancy measurements are also supported as an optional comparison 
but are not the primary measurements used in the paper.

Workflow
--------
1. Read a 3D DNA image and matching 3D nuclear mask.
2. Smooth the DNA image with a Gaussian filter.
3. Normalize DNA intensities using the selected percentile range and normalization mode.
4. Define chromatin-positive voxels within the nuclear mask.
5. Compute local chromatin occupancy using sliding-window and optional tile-based measurements.
6. Export per-nucleus summary tables and optional visualization outputs.

Outputs
-------
- per_nucleus_local_occupancy.csv
- condition_summary_local_occupancy.csv
- sliding-window occupancy summaries
- tile occupancy summaries
- condition-level histogram plots
- optional per-nucleus orthogonal-slice figures

Assumptions
-----------
- One nucleus per image volume.
- Each DNA image has a corresponding binary nuclear mask.
- Input image and mask volumes have matching dimensions.
"""

from __future__ import annotations

import os
import re
import glob
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter, uniform_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed


# Filename parsing settings
#user must add condition substrings here matching filenames for condition grouping
MANUAL_CONDITION_SUBSTRINGS: List[Tuple[str, str]] = []

IGNORE_TOKENS = ("stitched",)

DEFAULT_MASK_SUFFIXES = [
    "_mask_postprocessed",
    "_mask",
    "_cp_masks",
    "_seg",
    "_segmentation",
]


# =============================================================================
# Logging
# =============================================================================
def log(msg: str, log_file: Optional[str] = None):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


# =============================================================================
# File matching utilities
# =============================================================================
def stem_no_ext(path: str) -> str:
    base = os.path.basename(path)
    while True:
        root, ext = os.path.splitext(base)
        if ext.lower() in (".tif", ".tiff", ".ome"):
            base = root
        else:
            return base

def strip_any_suffix(stem: str, suffixes: List[str]) -> str:
    s = stem
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if s.endswith(suf):
                s = s[:-len(suf)]
                changed = True
    return s

def as_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

def pair_raw_and_masks(raw_dir: str, mask_dir: str, raw_glob: str) -> List[Tuple[str, str]]:
    raw_paths = sorted(glob.glob(os.path.join(raw_dir, raw_glob)))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif*")))

    mask_lookup: Dict[str, str] = {}
    for mp in mask_paths:
        st = stem_no_ext(mp)
        st2 = strip_any_suffix(st, DEFAULT_MASK_SUFFIXES)
        mask_lookup[st2] = mp

    pairs: List[Tuple[str, str]] = []
    missing = 0
    for rp in raw_paths:
        st = stem_no_ext(rp)
        if st in mask_lookup:
            pairs.append((rp, mask_lookup[st]))
        else:
            st_try = strip_any_suffix(st, DEFAULT_MASK_SUFFIXES)
            if st_try in mask_lookup:
                pairs.append((rp, mask_lookup[st_try]))
            else:
                missing += 1

    if missing > 0:
        log(f"WARNING: {missing} raw files had no matching mask (skipped).")

    return pairs


# =============================================================================
# Condition parsing
# =============================================================================

def infer_fallback_condition(image_id: str) -> str:
    s = str(image_id).lower()

    for tok in IGNORE_TOKENS:
        s = re.sub(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", "", s)

    s = re.sub(r"[_\s]+", "-", s)

    m_rep_tp = re.search(r"(?<![a-z0-9])(\d+)[-_](\d+(?:\.\d+)?)hpf(?![a-z0-9])", s)
    if m_rep_tp:
        return f"{m_rep_tp.group(1)}-{m_rep_tp.group(2)}hpf"

    m = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)hpf(?![a-z0-9])", s)
    if m:
        return f"{m.group(1)}hpf"

    return "unknown"


def infer_condition(image_id: str) -> str:
    s = str(image_id)
    s_lower = s.lower()

    for sub, cond in MANUAL_CONDITION_SUBSTRINGS:
        if sub.lower() in s_lower:
            return cond

    return infer_fallback_condition(s)

# =============================================================================
# Normalization and occupancy computation
# =============================================================================
def normalize_percentile(
    raw_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    p_low: float,
    p_high: float,
    norm_scope: str,
    eps: float,
) -> Tuple[np.ndarray, float, float]:
    vals = raw_zyx[mask_zyx]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([], dtype=np.float32), np.nan, np.nan

    if norm_scope == "mask":
        src = vals
    else:
        src = raw_zyx[np.isfinite(raw_zyx)]

    plo = float(np.percentile(src, p_low))
    phi = float(np.percentile(src, p_high))
    denom = float(phi - plo)

    if (not np.isfinite(plo)) or (not np.isfinite(phi)) or (denom < eps):
        return np.array([], dtype=np.float32), plo, phi

    norm_vals = np.clip((vals - plo) / denom, 0.0, 1.0).astype(np.float32, copy=False)
    return norm_vals, plo, phi

def build_norm_volume(raw_zyx: np.ndarray, mask_zyx: np.ndarray, *, p_lo: float, p_hi: float, eps: float) -> np.ndarray:
    denom = float(p_hi - p_lo)
    out = np.full(raw_zyx.shape, np.nan, dtype=np.float32)
    if (not np.isfinite(p_lo)) or (not np.isfinite(p_hi)) or (denom < eps):
        return out
    v = np.clip((raw_zyx.astype(np.float32) - float(p_lo)) / denom, 0.0, 1.0)
    out[mask_zyx] = v[mask_zyx].astype(np.float32, copy=False)
    return out

def tile_occupancy_3d(chromatin_zyx: np.ndarray, mask_zyx: np.ndarray, tile_zyx: Tuple[int, int, int]) -> np.ndarray:
    tz, ty, tx = tile_zyx
    if min(tz, ty, tx) < 1:
        raise ValueError("Tile sizes must be >= 1")

    Z, Y, X = chromatin_zyx.shape
    occ = np.full((Z, Y, X), np.nan, dtype=np.float32)

    for z0 in range(0, Z, tz):
        z1 = min(Z, z0 + tz)
        for y0 in range(0, Y, ty):
            y1 = min(Y, y0 + ty)
            for x0 in range(0, X, tx):
                x1 = min(X, x0 + tx)

                m = mask_zyx[z0:z1, y0:y1, x0:x1]
                denom = int(m.sum())
                if denom == 0:
                    continue
                c = chromatin_zyx[z0:z1, y0:y1, x0:x1]
                num = int(np.logical_and(c, m).sum())
                val = float(num) / float(denom)
                occ[z0:z1, y0:y1, x0:x1][m] = val

    return occ

def sliding_occupancy_3d(chromatin_zyx: np.ndarray, mask_zyx: np.ndarray, window_zyx: Tuple[int, int, int]) -> np.ndarray:
    """
    Compute local chromatin occupancy using a 3D sliding window.
    """
    wz, wy, wx = window_zyx
    if min(wz, wy, wx) < 1:
        raise ValueError("Window sizes must be >= 1")

    m = mask_zyx.astype(np.float32, copy=False)
    c = np.logical_and(chromatin_zyx, mask_zyx).astype(np.float32, copy=False)

    num = uniform_filter(c, size=(wz, wy, wx), mode="nearest")
    den = uniform_filter(m, size=(wz, wy, wx), mode="nearest")

    out = np.full(c.shape, np.nan, dtype=np.float32)
    good = den > 0
    out[good] = (num[good] / den[good]).astype(np.float32, copy=False)
    out[~mask_zyx] = np.nan
    return out


# =============================================================================
# Summaries statistics and histograms
# =============================================================================
def summarize_occ(values: np.ndarray) -> Dict[str, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return dict(mean=np.nan, std=np.nan, patchiness=np.nan, p10=np.nan, p50=np.nan, p90=np.nan)
    mean = float(np.mean(v))
    std = float(np.std(v))
    patch = float(std / mean) if (np.isfinite(mean) and mean > 0) else np.nan
    return dict(
        mean=mean,
        std=std,
        patchiness=patch,
        p10=float(np.percentile(v, 10)),
        p50=float(np.percentile(v, 50)),
        p90=float(np.percentile(v, 90)),
    )

def occ_hist(values: np.ndarray, bins: int) -> np.ndarray:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.zeros((bins,), dtype=np.float32)
    counts, _ = np.histogram(v, bins=bins, range=(0.0, 1.0))
    return (counts.astype(np.float32) / float(v.size))


# =============================================================================
# Plotting
# =============================================================================
def add_scale_bar(ax, voxel_nm_xy: Optional[Tuple[float, float]], length_nm: float = 500.0):
    if voxel_nm_xy is None:
        return
    xnm, ynm = voxel_nm_xy
    if (xnm is None) or (not np.isfinite(xnm)) or (xnm <= 0):
        return

    bar_len_px = length_nm / xnm
    margin_px = 10

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    x_start = x1 - margin_px - bar_len_px
    x_end = x1 - margin_px
    y_pos = y0 + margin_px

    ax.plot([x_start, x_end], [y_pos, y_pos], color="w", linewidth=3)
    ax.text((x_start + x_end) / 2, y_pos + margin_px * 0.8, f"{int(length_nm)} nm",
            color="w", ha="center", va="bottom", fontsize=8)

def plot_zslice_triptych_3slices(
    raw_zyx: np.ndarray,
    occ_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    out_path: str,
    title: str,
    voxel_um_zyx: Optional[Tuple[float, float, float]] = None,
    n_slices: int = 3,
    overlay_alpha: float = 0.75,
):
    """
    Paper-style visualization:
      For 3 Z slices, show:
        [Raw (gray)] [Occupancy map (10 discrete bins, red-blue LUT)] [Overlay]
    Occupancy map is DISCRETE (10 bins), not a smooth gradient.

    Notes:
      - occ_zyx expected in [0..1] with NaN outside mask.
      - Colors are 'RdBu_r' (blue=low, red=high) with 10 discrete steps.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import matplotlib.cm as cm

    Z, Y, X = raw_zyx.shape

    # Pick slice indices: center of mass +/- offsets within mask bounds
    idx = np.argwhere(mask_zyx)
    if idx.size > 0:
        zc = int(np.clip(int(np.round(idx[:, 0].mean())), 0, Z - 1))
        zmin = int(idx[:, 0].min())
        zmax = int(idx[:, 0].max())
    else:
        zc = Z // 2
        zmin, zmax = 0, Z - 1

    # Choose 3 slices: roughly lower/mid/upper around centroid but within nucleus extent
    if n_slices != 3:
        n_slices = 3
    span = max(1, (zmax - zmin))
    dz = max(1, span // 4)
    z_list = [max(zmin, zc - dz), zc, min(zmax, zc + dz)]
    z_list = [int(np.clip(z, 0, Z - 1)) for z in z_list]

    # 10 discrete bins on [0..1]
    # (Uniform bins; later, if you want more contrast near 1.0 we can change edges.)
    edges = np.linspace(0.0, 1.0, 11)

    # Discrete red-blue colormap (blue low, red high)
    base = cm.get_cmap("RdBu_r", 10)
    cmap = ListedColormap(base(np.arange(10)))
    norm = BoundaryNorm(edges, cmap.N, clip=True)

    # Scale bar conversion (optional)
    voxel_nm_xy = None
    if voxel_um_zyx is not None:
        vz, vy, vx = voxel_um_zyx  # (Z, Y, X)
        voxel_nm_xy = (vx * 1000.0, vy * 1000.0)

    def _add_scale_bar(ax, voxel_nm_xy, length_nm=500.0):
        if voxel_nm_xy is None:
            return
        xnm, ynm = voxel_nm_xy
        if (xnm is None) or (not np.isfinite(xnm)) or (xnm <= 0):
            return
        bar_len_px = length_nm / xnm
        margin_px = 10
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_start = x1 - margin_px - bar_len_px
        x_end = x1 - margin_px
        y_pos = y0 + margin_px
        ax.plot([x_start, x_end], [y_pos, y_pos], color="w", linewidth=3)
        ax.text((x_start + x_end) / 2, y_pos + margin_px * 0.8, f"{int(length_nm)} nm",
                color="w", ha="center", va="bottom", fontsize=8)

    # Figure layout: rows = slices, cols = (raw, map, overlay)
    fig = plt.figure(figsize=(9.6, 3.0 * len(z_list)))
    gs = fig.add_gridspec(len(z_list), 3, wspace=0.06, hspace=0.18)

    last_im = None
    for r, z in enumerate(z_list):
        raw2d = raw_zyx[z, :, :]
        occ2d = occ_zyx[z, :, :]
        m2d = mask_zyx[z, :, :]

        # Make masked occupancy (outside nucleus transparent)
        occ_masked = np.array(occ2d, copy=True)
        occ_masked[~m2d] = np.nan

        # RAW
        ax0 = fig.add_subplot(gs[r, 0])
        ax0.imshow(raw2d, cmap="gray", interpolation="nearest")
        ax0.set_title(f"Raw (Z={z})" if r == 0 else f"Z={z}", fontsize=10)
        ax0.set_xticks([]); ax0.set_yticks([])
        if r == 0:
            _add_scale_bar(ax0, voxel_nm_xy, length_nm=500.0)

        # MAP (discrete)
        ax1 = fig.add_subplot(gs[r, 1])
        # Use norm+cmap; NaNs will be transparent if we set bad alpha later (handled by imshow)
        last_im = ax1.imshow(occ_masked, cmap=cmap, norm=norm, interpolation="nearest")
        ax1.set_title("Occupancy (10 bins)" if r == 0 else "", fontsize=10)
        ax1.set_xticks([]); ax1.set_yticks([])

        # OVERLAY
        ax2 = fig.add_subplot(gs[r, 2])
        ax2.imshow(raw2d, cmap="gray", interpolation="nearest")
        last_im = ax2.imshow(occ_masked, cmap=cmap, norm=norm, alpha=overlay_alpha, interpolation="nearest")
        ax2.set_title("Overlay" if r == 0 else "", fontsize=10)
        ax2.set_xticks([]); ax2.set_yticks([])

    # Discrete colorbar with bin edges
    cax = fig.add_axes([0.92, 0.14, 0.02, 0.72])
    cb = fig.colorbar(last_im, cax=cax, boundaries=edges, ticks=edges)
    cb.set_label("Local chromatin occupancy\n(fraction > chromatin_t)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.90, 0.95])
    plt.savefig(out_path, dpi=220)
    plt.close()


# =============================================================================
# Per-nucleus analysis
# =============================================================================
def compute_one(raw_path: str, mask_path: str, args) -> Dict[str, object]:
    """
    Compute chromatin occupancy metrics for one raw image and matching nuclear mask.
    """
    image_id = stem_no_ext(raw_path)
    cond = infer_condition(image_id)

    try:
        raw = as_3d(tifffile.imread(raw_path)).astype(np.float32, copy=False)
        mask = as_3d(tifffile.imread(mask_path)) > 0
    except Exception as e:
        return {"status": "read_error", "image_id": image_id, "condition": cond, "raw_path": raw_path, "mask_path": mask_path, "error": str(e)}

    if raw.shape != mask.shape:
        return {"status": "shape_mismatch", "image_id": image_id, "condition": cond, "raw_path": raw_path, "mask_path": mask_path}

    raw = gaussian_filter(raw, sigma=float(args.gaussian_sigma_px))

    norm_vals_1d, p_lo, p_hi = normalize_percentile(
        raw, mask,
        p_low=float(args.p_low),
        p_high=float(args.p_high),
        norm_scope=str(args.norm_scope),
        eps=float(args.eps),
    )
    if norm_vals_1d.size == 0:
        return {"status": "bad_norm", "image_id": image_id, "condition": cond, "raw_path": raw_path, "mask_path": mask_path, "p_lo": float(p_lo), "p_hi": float(p_hi)}

    norm_vol = build_norm_volume(raw, mask, p_lo=float(p_lo), p_hi=float(p_hi), eps=float(args.eps))

    chromatin = np.zeros_like(mask, dtype=bool)
    chromatin[mask] = (norm_vol[mask] > float(args.chromatin_t))

    occ_sw = sliding_occupancy_3d(chromatin, mask, window_zyx=tuple(args.window))
    sw_vals = occ_sw[mask]
    sw_sum = summarize_occ(sw_vals)
    sw_hist = occ_hist(sw_vals, bins=int(args.occ_hist_bins))

    out: Dict[str, object] = {
        "status": "ok",
        "image_id": image_id,
        "condition": cond,
        "raw_path": raw_path,
        "mask_path": mask_path,
        "n_voxels_mask": int(mask.sum()),
        "p_lo": float(p_lo),
        "p_hi": float(p_hi),
        "norm_scope": str(args.norm_scope),
        "chromatin_t": float(args.chromatin_t),
        "chromatin_occupancy_global": float(np.mean(chromatin[mask])),
        "chromatin_free_fraction_global": float(np.mean(~chromatin[mask])),
        "window_zyx": "x".join(map(str, args.window)),
        "sw_occ_mean": sw_sum["mean"],
        "sw_occ_std": sw_sum["std"],
        "sw_patchiness_index": sw_sum["patchiness"],
        "sw_occ_p10": sw_sum["p10"],
        "sw_occ_p50": sw_sum["p50"],
        "sw_occ_p90": sw_sum["p90"],
    }

    for i in range(int(args.occ_hist_bins)):
        out[f"sw_occ_hist_{i:03d}"] = float(sw_hist[i])

    if args.compute_tile:
        occ_tile = tile_occupancy_3d(chromatin, mask, tile_zyx=tuple(args.tile))
        tile_vals = occ_tile[mask]
        tile_sum = summarize_occ(tile_vals)
        tile_hist = occ_hist(tile_vals, bins=int(args.occ_hist_bins))

        out.update({
            "tile_zyx": "x".join(map(str, args.tile)),
            "tile_occ_mean": tile_sum["mean"],
            "tile_occ_std": tile_sum["std"],
            "tile_patchiness_index": tile_sum["patchiness"],
            "tile_occ_p10": tile_sum["p10"],
            "tile_occ_p50": tile_sum["p50"],
            "tile_occ_p90": tile_sum["p90"],
        })

        for i in range(int(args.occ_hist_bins)):
            out[f"tile_occ_hist_{i:03d}"] = float(tile_hist[i])

    if args.make_figures:
        fig_dir = os.path.join(args.out_dir, "figures_per_nucleus", cond)
        os.makedirs(fig_dir, exist_ok=True)

        voxel_um_zyx = tuple(args.voxel_um) if args.voxel_um is not None else None

        plot_zslice_triptych_3slices(
            raw_zyx=raw, occ_zyx=occ_sw, mask_zyx=mask,
            out_path=os.path.join(fig_dir, f"{image_id}__sw_occ.png"),
            title=f"{cond} | {image_id} | window {args.window} | t>{args.chromatin_t}",
            voxel_um_zyx=voxel_um_zyx,
        )

        if args.compute_tile:
            plot_zslice_triptych_3slices(
                raw_zyx=raw, occ_zyx=occ_tile, mask_zyx=mask,
                out_path=os.path.join(fig_dir, f"{image_id}__tile_occ.png"),
                title=f"{cond} | {image_id} | tile {args.tile} | t>{args.chromatin_t}",
                voxel_um_zyx=voxel_um_zyx,
            )

    return out


# =============================================================================
# Condition-level histogram overlays
# =============================================================================
def plot_condition_hist_overlays(df_ok: pd.DataFrame, out_dir: str, bins: int):
    os.makedirs(out_dir, exist_ok=True)

    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _plot(kind: str):
        cols = [f"{kind}_occ_hist_{i:03d}" for i in range(bins)]
        plt.figure(figsize=(6.8, 5.0))
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for cond in sorted(df_ok["condition"].unique().tolist()):
            g = df_ok.loc[df_ok["condition"] == cond, cols].to_numpy(dtype=np.float32)
            if g.size == 0:
                continue
            med = np.nanmedian(g, axis=0)
            q25 = np.nanpercentile(g, 25, axis=0)
            q75 = np.nanpercentile(g, 75, axis=0)

            ax.plot(centers, med, linewidth=2, label=f"{cond} (n={g.shape[0]})")
            ax.fill_between(centers, q25, q75, alpha=0.2)

        ax.set_xlabel(f"Local occupancy (fraction > chromatin_t)")
        ax.set_ylabel("Probability per bin")
        ax.set_title(f"{kind.upper()} local occupancy histogram (median ± IQR)")
        ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"conditions__{kind}_occ_hist_overlay.png"), dpi=220)
        plt.savefig(os.path.join(out_dir, f"conditions__{kind}_occ_hist_overlay.pdf"))
        plt.close()

    _plot("tile")
    _plot("sw")


# =============================================================================
# CLI + main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="3D local chromatin occupancy analysis using sliding-window occupancy measurements."
    )
    p.add_argument("-i", "--input-dir", required=True, help="Directory of raw DNA TIFFs (one nucleus per file).")
    p.add_argument("-m", "--mask-dir", required=True, help="Directory of matching nuclear mask TIFFs.")
    p.add_argument("-o", "--out-dir", required=True, help="Output directory.")

    p.add_argument("--raw-glob", default="*.tif*", help="Glob for raw images. Default: *.tif*.")

    p.add_argument("--p-low", type=float, default=1.0, help="Lower percentile for normalization. Default: 1.")
    p.add_argument("--p-high", type=float, default=99.0, help="Upper percentile for normalization. Default: 99.")
    p.add_argument("--norm-scope", choices=["mask", "image"], default="mask",
                   help="Compute normalization percentiles within mask or whole image. Default: mask.")
    p.add_argument("--eps", type=float, default=1e-8, help="Epsilon to avoid divide-by-zero.")

    p.add_argument("--gaussian-sigma-px", type=float, default=1.0, help="Gaussian smoothing sigma in pixels. Default: 1.")
    p.add_argument("--chromatin-t", type=float, default=0.1, help="Chromatin threshold on normalized intensity. Default: 0.1.")

    p.add_argument("--window", nargs=3, type=int, default=[2, 6, 6],
                   metavar=("WZ", "WY", "WX"),
                   help="Sliding-window size in voxels (Z Y X). Default: 2 6 6.")

    p.add_argument("--compute-tile", action="store_true",
                   help="Also compute tile-based occupancy metrics.")
    p.add_argument("--tile", nargs=3, type=int, default=[2, 6, 6],
                   metavar=("TZ", "TY", "TX"),
                   help="Tile size in voxels (Z Y X), used only with --compute-tile. Default: 2 6 6.")

    p.add_argument("--occ-hist-bins", type=int, default=40, help="Histogram bins on [0, 1]. Default: 40.")

    p.add_argument("--make-figures", action="store_true", help="Save per-nucleus orthogonal-slice figures.")
    p.add_argument("--voxel-um", nargs=3, type=float, default=None,
                   metavar=("Z_UM", "Y_UM", "X_UM"),
                   help="Voxel size in microns (Z Y X) for scale bars. Optional. Example: 0.5 0.168 0.168")

    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Number of worker processes. Default: cpu_count - 1.")
    p.add_argument("--no-multiprocessing", action="store_true", help="Disable multiprocessing.")

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, "local_occupancy_log.txt")

    log("Starting local occupancy run", log_file=log_file)
    log(f"Input dir: {args.input_dir}", log_file=log_file)
    log(f"Mask dir:  {args.mask_dir}", log_file=log_file)
    log(f"Out dir:   {args.out_dir}", log_file=log_file)

    pairs = pair_raw_and_masks(args.input_dir, args.mask_dir, args.raw_glob)
    log(f"Found {len(pairs)} raw/mask pairs", log_file=log_file)
    if len(pairs) == 0:
        log("No pairs found. Check paths/globs.", log_file=log_file)
        return

    rows: List[Dict[str, object]] = []

    use_mp = (not args.no_multiprocessing) and (args.workers and args.workers > 1)
    if use_mp:
        log(f"Using multiprocessing: workers={args.workers}", log_file=log_file)
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [ex.submit(compute_one, rp, mp, args) for rp, mp in pairs]
            for fut in as_completed(futs):
                rows.append(fut.result())
    else:
        log("Running single-process", log_file=log_file)
        for rp, mp in pairs:
            rows.append(compute_one(rp, mp, args))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "per_nucleus_local_occupancy.csv")
    df.to_csv(out_csv, index=False)
    log(f"Wrote: {out_csv}", log_file=log_file)

    ok = df.loc[df["status"] == "ok"].copy()
    if ok.empty:
        log("No OK rows; skipping summaries/plots.", log_file=log_file)
        return

    metric_cols = [
        "chromatin_occupancy_global",
        "tile_occ_mean", "tile_occ_std", "tile_patchiness_index",
        "sw_occ_mean", "sw_occ_std", "sw_patchiness_index",
    ]

    def q25(x): return np.nanpercentile(x, 25)
    def q75(x): return np.nanpercentile(x, 75)

    summ = ok.groupby("condition")[metric_cols].agg(["median", q25, q75]).reset_index()
    summ.columns = ["_".join([c for c in col if c]).rstrip("_") for col in summ.columns.values]
    out_summ = os.path.join(args.out_dir, "condition_summary_local_occupancy.csv")
    summ.to_csv(out_summ, index=False)
    log(f"Wrote: {out_summ}", log_file=log_file)

    plot_dir = os.path.join(args.out_dir, "plots")
    plot_condition_hist_overlays(ok, out_dir=plot_dir, bins=int(args.occ_hist_bins))
    log(f"Wrote plots to: {plot_dir}", log_file=log_file)

    log("Done.", log_file=log_file)

if __name__ == "__main__":
    main()