#!/usr/bin/env python3

"""
Plotting and Statistics for Per-Embryo Nuclear Quantification CSVs
=================================================================

This script generates plots and statistical summaries from per-embryo
nuclear quantification CSV files produced by the unexpanded image analysis
pipeline.

Input file convention
---------------------
    C2-sampleX_conditionY_..._E_1_quantification.csv
    C2-sampleX_conditionY_..._E_2_quantification.csv
    ...

Where:
- Condition is inferred from the filename by stripping the trailing
  "_E_<n>_quantification.csv" portion.
- Embryo is inferred from the "_E_<n>" portion of the filename.

Statistical analysis
--------------------
Statistical comparisons are performed at the embryo level. Nucleus-level
measurements are first summarized as per-embryo means, and hypothesis tests
are then performed using embryo means as the replicate values.

The script reports:
- Descriptive statistics for nucleus-level measurements
- Descriptive statistics for per-embryo means
- Welch t-tests using per-embryo means
- Permutation tests using per-embryo means

Plots
-----
For each measurement column, the script generates:
- Nucleus-level bar plots
- Per-embryo mean plots
- Violin plots with embryo-level overlays

Usage
-----
    python plot_intensity_results_and_stats.py \
        --data_dir /path/to/<channel>_quantification \
        --control WT

Optional:
    --output_dir /path/to/save/plots_and_pdf
    --exclude substring_to_exclude
"""

import os
import re
import glob
import itertools
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind


from fpdf import FPDF


# ----------------------------
# Columns to plot
# ----------------------------
# Additional measurement columns can be added here if desired.

columns_to_plot = {
    'Mean Intensity': 'Mean Nuclear Intensity (a.u.)',
    'Sum Intensity': 'Total Nuclear Intensity (a.u.)',
    'Haralick Contrast d=2': 'Haralick Contrast (d=2 px)',
}

# ----------------------------
# Plot style guide
# ----------------------------
PLOT_BASE_WIDTH = 2.0
PLOT_PER_CONDITION = 0.95
PLOT_HEIGHT = 9.0
PLOT_MIN_WIDTH = 6.5
PLOT_MAX_WIDTH = 16.0

def pdf_safe(text: str) -> str:
    """
    FPDF (classic) writes text as latin-1. Replace common unicode characters
    (Greek letters, smart punctuation, etc.) with ASCII so pdf.output() cannot crash.
    """
    if text is None:
        return ""
    s = str(text)

    # Dashes / minus
    s = s.replace("\u2013", "-")  # en dash
    s = s.replace("\u2014", "-")  # em dash
    s = s.replace("\u2212", "-")  # minus sign

    # Quotes
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')

    # Common Greek letters that people put in stat strings
    s = s.replace("\u0394", "Delta")  # Δ
    s = s.replace("\u03b2", "beta")   # β (just in case)
    s = s.replace("\u03bc", "mu")     # μ (just in case)

    # Non-breaking spaces
    s = s.replace("\u00a0", " ")

    # Final safety: drop anything still not encodable in latin-1
    return s.encode("latin-1", errors="replace").decode("latin-1")


# ----------------------------
# Filename parsing
# ----------------------------

# Accept E1, E_1, E-1, E__1, E-_1, etc.
# Accept: condition_E1*, condition_E_1*, condition_E-1*,
# and allow optional suffixes like _sphericity_filtered before _quantification.csv
EMBRYO_FILE_REGEX = re.compile(
    r'^(?P<condition>.+?)[_-]+(?P<embryo>E(?:[_-]+\d+|\d+))(?P<suffix>.*)_quantification\.csv$'
)

EMBRYO_NUM_REGEX = re.compile(r'^E(?:(?:_|-)+)?(?P<num>\d+)$')


def parse_condition_and_embryo(filename: str):
    """
    Parse condition and embryo ID from a per-embryo quantification filename.

    Accepts embryo tokens like:
        E1, E_1, E-1, E__1, E-_1, ...

    Normalizes all embryo IDs to the canonical form:
        E_<number>
    """
    m = EMBRYO_FILE_REGEX.match(filename)
    if not m:
        return None, None

    condition = m.group("condition")
    embryo_token = m.group("embryo")

    m2 = EMBRYO_NUM_REGEX.match(embryo_token)
    if not m2:
        return None, None

    embryo_num = m2.group("num")
    embryo_id = f"E_{embryo_num}"  # canonical internal form
    return condition, embryo_id

def find_per_embryo_csvs(data_dir: str, exclude_substrings=None):
    """
    Find all per-embryo quantification CSVs in a directory.

    We ignore hidden macOS files like "._foo.csv".
    """
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    csvs = [p for p in csvs if not os.path.basename(p).startswith("._")]

    if exclude_substrings:
        exclude_substrings = [s.lower() for s in exclude_substrings]
        csvs = [
            p for p in csvs
            if not any(excl in os.path.basename(p).lower()
                       for excl in exclude_substrings)
        ]

    return csvs


def select_control_condition(all_conditions, control_substring: str):
    """
    Pick the control condition label using a substring match (case-insensitive).

    Rules:
    - If exactly one condition contains substring -> use it.
    - If multiple match -> choose the shortest label (heuristic) and warn.
    - If none -> raise ValueError.
    """
    needle = control_substring.lower().strip()
    matches = [c for c in sorted(all_conditions) if needle in c.lower()]
    if len(matches) == 0:
        raise ValueError(
            f"No condition matched --control '{control_substring}'.\n"
            f"Available conditions:\n  - " + "\n  - ".join(sorted(all_conditions))
        )
    if len(matches) == 1:
        return matches[0]

    # Multiple matches: select the shortest matching label and warn.
    chosen = min(matches, key=len)
    print(
        f"Warning: multiple conditions matched --control '{control_substring}'.\n"
        f"Matches:\n  - " + "\n  - ".join(matches) + "\n"
        f"Using shortest match as control: {chosen}\n"
        f"Tip: pass a more specific substring to --control if this is wrong."
    )
    return chosen


# ----------------------------
# Data loading
# ----------------------------

def build_long_table(per_embryo_files, column):
    """
    Build a long-form DataFrame with columns:
        Condition, Embryo, Value

    This table is the core representation used for nucleus-level plots,
    per-embryo summaries, and embryo-level statistical testing.

    NaN values are removed from the measurement column.
    """
    rows = []
    for fp in per_embryo_files:
        fn = os.path.basename(fp)
        condition, embryo = parse_condition_and_embryo(fn)
        if condition is None:
            # Skip files that don't match the expected per-embryo naming
            continue

        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip().str.replace("'", "")

        if column not in df.columns:
            print(f"Warning: '{column}' column not found in {fn}. Skipping.")
            continue

        values = df[column].dropna().values
        if values.size == 0:
            continue

        # Add one row per nucleus measurement
        rows.append(pd.DataFrame({
            "Condition": condition,
            "Embryo": embryo,
            "Value": values
        }))

    if not rows:
        return pd.DataFrame(columns=["Condition", "Embryo", "Value"])

    out = pd.concat(rows, ignore_index=True)

    # Ensure types are clean
    out["Condition"] = out["Condition"].astype(str)
    out["Embryo"] = out["Embryo"].astype(str)
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna(subset=["Value"])
    return out


# ----------------------------
# Plot styling helpers
# ----------------------------

def get_darker_color(color):
    dark_palette = sns.dark_palette(color, n_colors=3, reverse=True)
    return dark_palette[1]


def make_condition_colors(conditions):
    palette = sns.color_palette("pastel", len(conditions))
    return dict(zip(conditions, palette))


# ----------------------------
# Plotting
# ----------------------------

def figsize_by_conditions(n_conditions: int):
    w = PLOT_BASE_WIDTH + PLOT_PER_CONDITION * max(1, int(n_conditions))
    w = max(PLOT_MIN_WIDTH, min(w, PLOT_MAX_WIDTH))
    return (w, PLOT_HEIGHT)

def plot_column(long_df, column, ylabel, plot_dir, condition_colors):
    """
    Nucleus-level plot:
    - All nuclei points (stripplot)
    - Mean bar per condition
    - Std dev error bar (nucleus-level std) per condition

    Note: nucleus-level std primarily reflects within-embryo + between-embryo variability.
    For embryo-replicate variability, see per-embryo plot.
    """
    if long_df.empty:
        print(f"No data for column '{column}'. Skipping plot.")
        return

    conditions = list(long_df["Condition"].unique())

    plt.figure(figsize=figsize_by_conditions(len(conditions)))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'

    # Scatter of individual nuclei
    for condition in conditions:
        color = condition_colors[condition]
        condition_data = long_df[long_df["Condition"] == condition]
        sns.stripplot(
            x="Condition", y="Value", data=condition_data, color=color,
            alpha=0.5, jitter=0.2, dodge=False, ax=ax
        )

        # Show nuclei count
        num_values = len(condition_data)
        x_pos = conditions.index(condition)
        ax.text(
            x_pos, long_df["Value"].max() * 1.1, f'n={num_values}',
            ha='center', va='bottom', color='black', fontsize=9, weight='bold'
        )

    # Mean bars
    for condition in conditions:
        color = condition_colors[condition]
        stroke_color = get_darker_color(color)
        condition_data = long_df[long_df["Condition"] == condition]
        sns.barplot(
            x="Condition", y="Value", data=condition_data, estimator=np.mean,
            color=color, edgecolor=stroke_color, capsize=0.0, dodge=False,
            alpha=0.5, linewidth=1.5, width=0.6, ax=ax, errorbar=None
        )

    # Manual error bars (nucleus-level std)
    for condition in conditions:
        color = condition_colors[condition]
        condition_data = long_df[long_df["Condition"] == condition]
        x_pos = conditions.index(condition)
        mean_value = np.mean(condition_data["Value"])
        std_dev = condition_data["Value"].std()

        ax.errorbar(
            x=x_pos, y=mean_value, yerr=std_dev, fmt='none',
            ecolor=get_darker_color(color), elinewidth=2,
            capsize=4, capthick=2, zorder=3
        )

    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.title(f"{column}")
    plt.tight_layout(pad=2.5)

    tif_path = os.path.join(plot_dir, f"{column}_plot.tif")
    svg_path = os.path.join(plot_dir, f"{column}_plot.svg")
    plt.savefig(tif_path, format="tiff", dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"Plot saved as {tif_path} and {svg_path}.")
    plt.close()


def plot_column_per_embryo(long_df, column, ylabel, plot_dir, condition_colors):
    """
    Per-embryo plot:
    - Each point is an embryo mean
    - Mean bar per condition (across embryos)
    - Error bars = std of embryo means

    This is aligned with the embryo as the experimental replicate.
    """
    if long_df.empty:
        print(f"No data for column '{column}'. Skipping per-embryo plot.")
        return

    grouped = long_df.groupby(["Condition", "Embryo"], as_index=False, observed=False)["Value"].mean()
    if grouped.empty:
        print(f"No per-embryo means for '{column}'. Skipping.")
        return

    conditions = list(grouped["Condition"].unique())

    plot_dir_embryo = os.path.join(plot_dir, "plots_per_embryo")
    os.makedirs(plot_dir_embryo, exist_ok=True)

    conditions = list(long_df["Condition"].unique())
    plt.figure(figsize=figsize_by_conditions(len(conditions)))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'

    # Scatter embryo means
    for condition in conditions:
        color = condition_colors[condition]
        condition_data = grouped[grouped["Condition"] == condition]
        sns.stripplot(
            x="Condition", y="Value", data=condition_data, color=color,
            alpha=0.6, jitter=0.2, dodge=False, ax=ax
        )

        num_embryos = condition_data["Embryo"].nunique()
        x_pos = conditions.index(condition)
        ax.text(
            x_pos, grouped["Value"].max() * 1.1, f'n={num_embryos}',
            ha='center', va='bottom', color='black', fontsize=9, weight='bold'
        )

    # Bars of embryo means
    for condition in conditions:
        color = condition_colors[condition]
        stroke_color = get_darker_color(color)
        condition_data = grouped[grouped["Condition"] == condition]
        sns.barplot(
            x="Condition", y="Value", data=condition_data, estimator=np.mean,
            color=color, edgecolor=stroke_color, capsize=0.0, dodge=False,
            alpha=0.5, linewidth=1.5, width=0.6, ax=ax, errorbar=None
        )

    # Error bars: std across embryo means
    for condition in conditions:
        color = condition_colors[condition]
        condition_data = grouped[grouped["Condition"] == condition]
        x_pos = conditions.index(condition)
        mean_value = condition_data["Value"].mean()
        std_dev = condition_data["Value"].std()

        ax.errorbar(
            x=x_pos, y=mean_value, yerr=std_dev, fmt='none',
            ecolor=get_darker_color(color), elinewidth=2,
            capsize=4, capthick=2, zorder=3
        )

    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.title(f"{column} (Per-Embryo Means)")
    plt.tight_layout(pad=2.5)

    tif_path = os.path.join(plot_dir_embryo, f"{column}_per_embryo.tif")
    svg_path = os.path.join(plot_dir_embryo, f"{column}_per_embryo.svg")
    plt.savefig(tif_path, format="tiff", dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"Per-embryo plot saved as {tif_path} and {svg_path}.")
    plt.close()


def plot_violin_column(long_df: pd.DataFrame, column: str, ylabel: str, plot_dir: str, condition_colors: dict):
    """
    Violin plot (per-nucleus) with overlaid per-embryo means.

    - Violin: nucleus-level distribution for each condition.
    - Inner box: median and interquartile range.
    - Overlay points: per-embryo means.
    - Annotations: nucleus counts per condition.
    """
    if long_df.empty:
        print(f"No data available for column '{column}'. Skipping violin plot.")
        return

    # Stable condition order
    if pd.api.types.is_categorical_dtype(long_df["Condition"]):
        conditions = list(long_df["Condition"].cat.categories)
    else:
        conditions = list(long_df["Condition"].unique())

    # Per-embryo means for overlay (one dot per embryo)
    embryo_means = (
        long_df.groupby(["Condition", "Embryo"], as_index=False, observed=False)["Value"]
        .mean()
    )

    plt.figure(figsize=figsize_by_conditions(len(conditions)))
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"

    # --- Violin with inner box/line ---
    sns.violinplot(
        x="Condition",
        y="Value",
        data=long_df,
        order=conditions,
        palette=condition_colors,
        inner="box",
        cut=0,
        linewidth=1,
        saturation=1.0,  
        ax=ax
    )

    for coll in ax.collections:
        try:
            coll.set_alpha(0.5)
        except Exception:
            pass

    # --- Overlay per-embryo means ---
    # Dots use a darker shade of each pastel condition color.
    for i, cond in enumerate(conditions):
        color = condition_colors[cond]
        sub = embryo_means[embryo_means["Condition"] == cond]
        ax.scatter(
            np.full(len(sub), i),
            sub["Value"].to_numpy(),
            color=get_darker_color(color),
            edgecolors="none",
            alpha=0.7,
            s=60,
            zorder=4
        )

    # --- Annotate per-nucleus n ---
    y_top = float(long_df["Value"].max())
    for i, cond in enumerate(conditions):
        n_nuc = int((long_df["Condition"] == cond).sum())
        ax.text(
            i, y_top * 1.05, f"n={n_nuc}",
            ha="center", va="bottom",
            color="black", fontsize=9, weight="bold"
        )

    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.title(f"{column} (Violin Plot with Embryo Means)")
    plt.tight_layout(pad=2.5)

    tif_path = os.path.join(plot_dir, f"{column}_violin_plot.tif")
    svg_path = os.path.join(plot_dir, f"{column}_violin_plot.svg")
    plt.savefig(tif_path, format="tiff", dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"Violin plot saved as {tif_path} and {svg_path}.")
    plt.close()

# ----------------------------
# Stats helpers
# ----------------------------

def get_descriptive_stats(long_df):
    """
    Descriptive stats at nucleus-level for a given Condition slice of the long table.
    """
    mean = long_df["Value"].mean()
    median = long_df["Value"].median()
    std = long_df["Value"].std()
    count = long_df["Value"].count()
    return mean, median, std, count

def embryo_level_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse nucleus-level long_df (Condition, Embryo, Value) to embryo means.

    Returns a DataFrame with columns:
        Condition, Embryo, EmbryoMean

    This is the appropriate level for simple tests when the experimental unit is the embryo.
    """
    if long_df.empty:
        return pd.DataFrame(columns=["Condition", "Embryo", "EmbryoMean"])

    tbl = (
        long_df.groupby(["Condition", "Embryo"], as_index=False, observed=False)["Value"]
        .mean()
        .rename(columns={"Value": "EmbryoMean"})
    )
    return tbl


def perform_t_tests_embryo_level(emb_tbl: pd.DataFrame):
    """
    Welch's t-tests on embryo means (embryo is the replicate).
    Returns tuples: (cond1, cond2, t_stat, p_value, n_emb1, n_emb2)
    """
    results = []
    if emb_tbl.empty:
        return results

    conditions = list(emb_tbl["Condition"].unique())
    for cond1, cond2 in itertools.combinations(conditions, 2):
        d1 = emb_tbl[emb_tbl["Condition"] == cond1]["EmbryoMean"].dropna().values
        d2 = emb_tbl[emb_tbl["Condition"] == cond2]["EmbryoMean"].dropna().values
        if len(d1) < 2 or len(d2) < 2:
            continue
        t_stat, p_value = ttest_ind(d1, d2, equal_var=False)
        results.append((cond1, cond2, float(t_stat), float(p_value), len(d1), len(d2)))
    return results


def permutation_tests_embryo_level_vs_control(
    emb_tbl: pd.DataFrame,
    control_condition: str = None,
    num_permutations: int = 10000,
    seed: int = 0,
    max_exact: int = 200_000,
):
    """
    Permutation test on embryo means (embryo is the unit).

    If control_condition is provided:
      - compare each treatment condition against the control condition

    If control_condition is None:
      - run ALL pairwise comparisons between conditions

    Returns list of tuples:
      (treat, ctrl, observed_mean_diff, p_value, n_emb_treat, n_emb_control, method, n_perms_used)

    observed_mean_diff = mean(treat) - mean(ctrl)
    Two-sided p-value based on absolute mean difference.
    """
    if emb_tbl is None or emb_tbl.empty:
        return []

    conditions = sorted(set(emb_tbl["Condition"].unique()))
    if len(conditions) < 2:
        return []

    rng = np.random.default_rng(seed)

    def _one_pair(ctrl_cond: str, treat_cond: str):
        ctrl = emb_tbl.loc[emb_tbl["Condition"] == ctrl_cond, "EmbryoMean"].dropna().to_numpy()
        trt = emb_tbl.loc[emb_tbl["Condition"] == treat_cond, "EmbryoMean"].dropna().to_numpy()
        n_ctrl = len(ctrl)
        n_trt = len(trt)

        if n_ctrl < 2 or n_trt < 2:
            return None

        observed = trt.mean() - ctrl.mean()
        observed_abs = abs(observed)

        combined = np.concatenate([ctrl, trt]).astype(float)
        n_total = combined.size
        n_configs = math.comb(n_total, n_ctrl)

        if n_configs <= max_exact:
            idx = np.arange(n_total)
            total_sum = combined.sum()
            count = 0

            for ctrl_idx in itertools.combinations(idx, n_ctrl):
                ctrl_idx = np.fromiter(ctrl_idx, dtype=int, count=n_ctrl)
                sum_ctrl = combined[ctrl_idx].sum()
                mean_ctrl = sum_ctrl / n_ctrl

                sum_trt = total_sum - sum_ctrl
                mean_trt = sum_trt / n_trt

                perm_abs = abs(mean_trt - mean_ctrl)
                if perm_abs >= observed_abs:
                    count += 1

            p_value = count / n_configs
            method = "exact"
            n_used = n_configs

        else:
            labels = np.array([0] * n_ctrl + [1] * n_trt, dtype=int)
            count = 0
            for _ in range(num_permutations):
                rng.shuffle(labels)
                perm_ctrl = combined[labels == 0]
                perm_trt = combined[labels == 1]
                perm_abs = abs(perm_trt.mean() - perm_ctrl.mean())
                if perm_abs >= observed_abs:
                    count += 1

            p_value = (count + 1) / (num_permutations + 1)
            method = "mc"
            n_used = num_permutations

        return (treat_cond, ctrl_cond, float(observed), float(p_value),
                int(n_trt), int(n_ctrl), method, int(n_used))

    out = []

    if control_condition is not None and control_condition in conditions:
        # Compare each non-control condition against the selected control.
        for treat in [c for c in conditions if c != control_condition]:
            r = _one_pair(control_condition, treat)
            if r is not None:
                out.append(r)
        return out

    # If no control is provided, run all pairwise comparisons.
    for ctrl_cond, treat_cond in itertools.combinations(conditions, 2):
        # define "ctrl" as the first in the pair, "treat" as the second
        r = _one_pair(ctrl_cond, treat_cond)
        if r is not None:
            out.append(r)

    return out

def effect_size_vs_control_from_embryo_means(emb_tbl: pd.DataFrame, control_condition: str = None):
    """
    If control_condition is provided:
      returns dict[treat] -> (delta, pct_change, fold_change, ctrl_mean, trt_mean, nC, nT)

    If control_condition is None:
      returns dict[(treat, ctrl)] -> (delta, pct_change, fold_change, ctrl_mean, trt_mean, nC, nT)
      for ALL pairwise comparisons, where delta = mean(treat) - mean(ctrl).
    """
    out = {}
    if emb_tbl is None or emb_tbl.empty:
        return out

    conditions = sorted(set(emb_tbl["Condition"].unique()))
    if len(conditions) < 2:
        return out

    def _stats(ctrl_cond: str, treat_cond: str):
        ctrl = emb_tbl.loc[emb_tbl["Condition"] == ctrl_cond, "EmbryoMean"].dropna().to_numpy()
        trt = emb_tbl.loc[emb_tbl["Condition"] == treat_cond, "EmbryoMean"].dropna().to_numpy()
        if ctrl.size == 0 or trt.size == 0:
            return None

        ctrl_mean = float(np.mean(ctrl))
        trt_mean = float(np.mean(trt))
        nC = int(ctrl.size)
        nT = int(trt.size)

        delta = trt_mean - ctrl_mean
        pct = (100.0 * delta / ctrl_mean) if ctrl_mean != 0 else float("nan")
        fold = (trt_mean / ctrl_mean) if ctrl_mean != 0 else float("nan")
        return (delta, pct, fold, ctrl_mean, trt_mean, nC, nT)

    if control_condition is not None and control_condition in conditions:
        for treat in [c for c in conditions if c != control_condition]:
            s = _stats(control_condition, treat)
            if s is not None:
                out[treat] = s
        return out

    # If no control is provided, run all pairwise comparisons.
    for ctrl_cond, treat_cond in itertools.combinations(conditions, 2):
        s = _stats(ctrl_cond, treat_cond)
        if s is not None:
            out[(treat_cond, ctrl_cond)] = s

    return out

# ----------------------------
# PDF report
# ----------------------------

def generate_pdf_report(column_results, output_path, control_condition, control_substring):
    """
    Generate a PDF summary report for each measurement column.

    Each section includes nucleus-level descriptive statistics, per-embryo
    descriptive statistics, per-embryo Welch t-tests, and per-embryo
    permutation tests. Hypothesis testing is performed on embryo means.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for column, payload in column_results.items():
        descriptive_stats = payload.get("descriptive_stats", {})
        emb_tbl = payload.get("emb_tbl", pd.DataFrame())
        emb_ttest_results = payload.get("emb_ttest_results", [])
        perm_results = payload.get("perm_results", [])
        effects_vs_ctrl = payload.get("effects_vs_ctrl", {})

        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, pdf_safe(f"Statistics Summary: {column}"), ln=True, align="C")
        pdf.ln(2)

        pdf.set_font("Arial", "", 10)
        notes_text = (
            "Notes:\n"
            "- Embryo-level tests operate on per-embryo means, with one value per embryo.\n"
            "- Nucleus-level measurements are summarized descriptively but are not used as independent replicates for hypothesis testing.\n"
            f"- Control selection: --control '{control_substring}' -> control condition label: '{control_condition}'."
        )
        pdf.multi_cell(0, 6, pdf_safe(notes_text))
        pdf.ln(2)

        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, pdf_safe("Descriptive statistics (nucleus-level):"), ln=True)
        pdf.set_font("Arial", "", 10)

        if descriptive_stats:
            for condition, stats in descriptive_stats.items():
                mean, median, std, count = stats
                pdf.multi_cell(
                    0, 6,
                    pdf_safe(
                        f"{condition}\n"
                        f"  Mean={mean:.4f}  Median={median:.4f}  SD={std:.4f}  n(nuclei)={count}"
                    )
                )
                pdf.ln(1)
        else:
            pdf.multi_cell(0, 6, pdf_safe("No nucleus-level descriptive stats available."))
        pdf.ln(1)

        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, pdf_safe("Descriptive statistics (per-embryo means):"), ln=True)
        pdf.set_font("Arial", "", 10)

        if emb_tbl is None or (isinstance(emb_tbl, pd.DataFrame) and emb_tbl.empty):
            pdf.multi_cell(0, 6, pdf_safe("No embryo-level means available."))
        else:
            for cond in sorted(emb_tbl["Condition"].unique()):
                sub = emb_tbl.loc[emb_tbl["Condition"] == cond, "EmbryoMean"].dropna().values
                if sub.size == 0:
                    continue
                sd = float(np.std(sub, ddof=1)) if sub.size >= 2 else float("nan")
                pdf.multi_cell(
                    0, 6,
                    pdf_safe(
                        f"{cond}\n"
                        f"  Mean={float(np.mean(sub)):.4f}  Median={float(np.median(sub)):.4f}  "
                        f"SD={sd:.4f}  n(embryos)={int(sub.size)}"
                    )
                )
                pdf.ln(1)

            pdf.ln(1)
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, pdf_safe("Welch's t-tests (per-embryo means):"), ln=True)
            pdf.set_font("Arial", "", 10)

            if emb_ttest_results:
                for r in emb_ttest_results:
                    cond1, cond2, t_stat, p_value, n1, n2 = r
                    pdf.multi_cell(
                        0, 6,
                        pdf_safe(
                            f"{cond1} vs {cond2}\n"
                            f"  t={t_stat:.3f}  p={p_value:.4g}  n1={n1} n2={n2}"
                        )
                    )
                    pdf.ln(1)
            else:
                pdf.multi_cell(0, 6, pdf_safe("No embryo-level t-tests (need >=2 embryos per group)."))

            pdf.ln(1)
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, pdf_safe("Permutation tests (per-embryo means):"), ln=True)
            pdf.set_font("Arial", "", 10)

            if perm_results:
                for r in perm_results:
                    treat, ctrl, obs, pval, nT, nC, method, n_used = r
                    key = treat if (treat in effects_vs_ctrl) else (treat, ctrl)
                    delta, pct, fold, ctrl_mean, trt_mean, nC2, nT2 = effects_vs_ctrl.get(
                        key, (np.nan, np.nan, np.nan, np.nan, np.nan, nC, nT)
                    )

                    pdf.multi_cell(
                        0, 6,
                        pdf_safe(
                            f"{treat} vs {ctrl}\n"
                            f"  embryo-mean ctrl={ctrl_mean:.4f}  treat={trt_mean:.4f}\n"
                            f"  Delta={delta:.4f}  %change={pct:.2f}%  fold={fold:.3f}\n"
                            f"  permutation p={pval:.4g}\n"
                            f"  nT={nT} nC={nC}  method={method} permutations={n_used}"
                        )
                    )
                    pdf.ln(1)
            else:
                pdf.multi_cell(0, 6, pdf_safe("No permutation test results (insufficient data)."))

    pdf.output(output_path)

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate plots and embryo-level statistics from per-embryo quantification CSVs."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Path to the <channel>_quantification directory containing per-embryo *_quantification.csv files.")
    parser.add_argument("--output_dir",
                        help="Optional output directory. Default: <data_dir>/plots")
    parser.add_argument(
        "--control",
        default=None,
        help="Optional: substring that identifies the control condition in filenames (case-insensitive), e.g. 'mEmerald'. "
             "If omitted, control-based permutation tests are skipped."
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude files whose filenames contain this substring "
             "(case-insensitive). Repeatable."
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    control_substring = args.control
    control_condition = None

    if not os.path.isdir(data_dir):
        raise ValueError(f"--data_dir does not exist or is not a directory: {data_dir}")

    plot_dir = args.output_dir if args.output_dir else os.path.join(data_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Find per-embryo CSVs
    per_embryo_files = find_per_embryo_csvs(
        data_dir,
        exclude_substrings=args.exclude
    )
    if not per_embryo_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    # Determine available conditions from filenames
    conditions = set()
    for fp in per_embryo_files:
        cond, emb = parse_condition_and_embryo(os.path.basename(fp))
        if cond is not None:
            conditions.add(cond)

    if not conditions:
        raise ValueError(
            "No files matched the expected pattern: <condition>_E_<n>_quantification.csv\n"
            f"Example expected: blahblah_E_1_quantification.csv\n"
            f"Directory scanned: {data_dir}"
        )

    # Select control condition label, if provided
    control_condition = None
    if control_substring:
        control_condition = select_control_condition(conditions, control_substring)
        print(f"Control condition selected: {control_condition}")
    else:
        print("No --control provided: skipping control-based permutation tests.")

    # Color palette per condition (stable ordering)
    sorted_conditions = sorted(conditions)
    condition_colors = make_condition_colors(sorted_conditions)

    # Process each column: load long table, plot, compute stats
    column_results = {}

    for column, ylabel in columns_to_plot.items():
        long_df = build_long_table(per_embryo_files, column)

        # Restrict to conditions we successfully parsed (and keep stable ordering)
        if long_df.empty:
            print(f"No data found for '{column}'.")
            continue

        # Make Condition a categorical with a stable order (useful for consistent plot order)
        present_conditions = sorted(long_df["Condition"].unique())
        long_df["Condition"] = pd.Categorical(long_df["Condition"], categories=present_conditions, ordered=True)

        # Plots
        plot_column(long_df, column, ylabel, plot_dir, condition_colors)
        plot_column_per_embryo(long_df, column, ylabel, plot_dir, condition_colors)
        plot_violin_column(long_df, column, ylabel, plot_dir, condition_colors)

        # Descriptive stats per condition (nucleus-level)
        descriptive_stats = {
            cond: get_descriptive_stats(long_df[long_df["Condition"] == cond])
            for cond in long_df["Condition"].cat.categories
            if (long_df["Condition"] == cond).any()
        }

        # Embryo-level table (one mean per embryo)
        emb_tbl = embryo_level_table(long_df)
        effects_vs_ctrl = effect_size_vs_control_from_embryo_means(emb_tbl, control_condition)

        # Embryo-level tests
        emb_ttest_results = perform_t_tests_embryo_level(emb_tbl)
        perm_results = permutation_tests_embryo_level_vs_control(
            emb_tbl,
            control_condition=control_condition,
            num_permutations=10000,
            seed=0
        )

        column_results[column] = {
            "descriptive_stats": descriptive_stats,
            "emb_tbl": emb_tbl,
            "emb_ttest_results": emb_ttest_results,
            "perm_results": perm_results,
            "effects_vs_ctrl": effects_vs_ctrl,
        }

    # PDF report
    pdf_path = os.path.join(plot_dir, "descriptive_statistics_and_embryo_level_tests.pdf")
    generate_pdf_report(column_results, pdf_path, control_condition=control_condition, control_substring=control_substring)
    print(f"Saved PDF report: {pdf_path}")


if __name__ == "__main__":
    main()