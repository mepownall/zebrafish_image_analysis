# ChromExM Image Analysis Pipeline

## Overview

This directory contains analysis workflows used for quantitative analysis of zebrafish embryo ChromExM datasets.

The workflow includes:

- Extended z-stack stitching
- Photobleaching measurement
- Photobleaching correction
- Nucleus segmentation using a custom 2.5D U-Net model
- Quantitative nuclear measurements
- Local chromatin occupancy analysis
- Chromatin autocorrelation analysis
- Expansion factor measurements

The analysis is organized as a collection of modular workflows rather than a single fully automated pipeline. Individual components can be run independently depending on the experiment.

---

## Analysis Workflow

The primary workflow used in this study is:

```text
Raw expanded images
    ↓
Extended z-stack stitching (optional)
    ↓
Photobleaching measurement
    ↓
Nucleus segmentation and quantification
    ↓
Photobleaching correction
    ↓
Local chromatin occupancy analysis
    ↓
Autocorrelation analysis
```

---

## Scripts

### Main Analysis Pipeline

#### `expanded_pipeline_nucleus_and_particles.py`

Main expanded-image analysis pipeline.

Performs:

- Image preprocessing
- Nucleus segmentation
- Particle segmentation
- Quantitative measurements
- Output generation

Particle segmentation functionality is included but was not used for the analyses presented in this study.

#### `expanded_pipeline_nucleus_and_particles_treat_T_as_Z.py`

Variant of the main pipeline used for time-series datasets.

This version swaps the time and z dimensions to allow processing of time-series image stacks using the same workflow developed for z-stack image volumes.

#### `pipeline_config.json`

Configuration file defining:

- Input and output paths
- Segmentation models
- Imaging channels
- Pipeline parameters

---

## Nucleus Segmentation

#### `segment_with_2_5D_unet_volume_norm.py`

Performs nucleus segmentation using a trained 2.5D U-Net model.

#### `unet_training_volumes.py`

Training script used to generate the segmentation model from annotated image volumes.

#### `customunet_ResASPP_25D.py`

Definition of the custom ResASPP-based 2.5D U-Net architecture.

---

## Photobleaching Analysis

#### `measure_time_series_bleaching_with_3D_mask.py`

Measures photobleaching from time-series imaging datasets using 3D nuclear masks.

#### `plot_time_series_bleaching.py`

Visualizes photobleaching measurements across imaging time series.

#### `merge_bleaching_csvs_for_correction.py`

Combines bleaching measurements into a consolidated correction dataset.

#### `bleach_correction.py`

Applies bleaching correction factors to image datasets.

#### `bleach_correction_time_series.py`

Applies bleaching correction to time-series imaging datasets.

#### `plot_bleaching_summary.py`

Generates summary plots comparing bleaching behavior across conditions.

---

## Local Chromatin Occupancy Analysis

#### `local_chromatin_occupancy.py`

Measures local chromatin occupancy and chromatin volume concentration metrics from 3D image data.

Outputs include local occupancy maps and summary occupancy statistics.

---

## Autocorrelation Analysis

#### `autocorrelation_parameter_sweep.py`

Computes autocorrelation measurements and evaluates parameter choices across datasets.

#### `plot_correlation_length.py`

Generates plots of autocorrelation-derived correlation length measurements.

#### `replot_autocorrelation.py`

Recreates autocorrelation summary figures from existing analysis outputs.

---

## Extended Z-Stack Stitching

#### `extended_z_stack_stitching.py`

Stitches extended z-stack image volumes into unified image datasets for downstream analysis.

---

## Expansion Factor Analysis

Additional scripts are provided for measuring sample expansion factors and validating expansion performance across experiments.

---

## Model Requirements

The nucleus segmentation workflow requires a trained 2.5D U-Net model.

Model weights are not included in this repository and must be obtained separately.

---

## Configuration

Most workflows are controlled through configuration files and script parameters.

Before running any workflow:

- Verify all input paths.
- Verify all output paths.
- Verify model locations.
- Verify channel assignments.

---

## Outputs

The expanded-image workflows generate:

- Segmentation masks
- Quantitative measurement tables
- Bleach-corrected image volumes
- Local chromatin occupancy maps
- Occupancy summary statistics
- Autocorrelation measurements
- Correlation length estimates
- Publication-quality figures

---

## Notes

These scripts were developed for analysis of ChromExM datasets from zebrafish embryos and are provided as modular analysis components. Individual workflows can be run independently depending on the imaging experiment and analysis goals.
