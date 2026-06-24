# ChromExM Image Analysis Pipeline

## Overview

This directory contains analysis workflows used for quantitative analysis of zebrafish embryo ChromExM datasets.

The workflow includes:
- Nucleus segmentation using a custom 2.5D U-Net model
- Local chromatin occupancy analysis
- Chromatin autocorrelation analysis

The analysis is organized as a collection of modular workflows rather than a single fully automated pipeline. Individual components can be run independently depending on the experiment.

---

## Analysis Workflow

The primary workflow used in this study is:

```text
Raw expanded images
    ↓
Nucleus segmentation and quantification
    ↓
Autocorrelation analysis
    ↓
Local chromatin occupancy analysis

```

---

## Scripts

### Main Analysis Pipeline

#### `ChromExM_pipeline.py

Main expanded-image analysis pipeline.

Performs:

- Image preprocessing
- Nucleus segmentation
- Particle segmentation
- Quantitative measurements
- Output generation


#### `ChromExM_pipeline_config.json`

Configuration file defining:

- Input and output paths
- Segmentation models
- Imaging channels
- Pipeline parameters

- Before running the pipeline, update:
- 
- input_directory
- output_directory
- nuclei_unet_script
- nuclei_unet_model
- background_image_path

to match your local installation.

---

## Nucleus Segmentation

#### `segment_with_unet.py`

Performs nucleus segmentation using a trained 2.5D U-Net model.

#### `unet_resapp_25d.py`

Definition of the custom ResASPP-based 2.5D U-Net architecture.


---

## Local Chromatin Occupancy Analysis

#### `local_chromatin_occupancy.py`

Measures local chromatin occupancy from ChromExM images.

Outputs include occupancy statistics.

---

## Autocorrelation Analysis

#### `radial_autocorrelation_measurement.py`

Computes autocorrelation measurements.

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
- Occupancy summary statistics
- Autocorrelation measurements
- Correlation length estimates

---

## Notes

These scripts were developed for analysis of ChromExM datasets from zebrafish embryos and are provided as modular analysis components. Individual workflows can be run independently depending on the imaging experiment and analysis goals.
