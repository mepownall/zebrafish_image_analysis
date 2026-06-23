# Unexpanded Image Analysis Pipeline

## Overview

This directory contains the image analysis workflow used for quantitative analysis of diffraction-limited microscopy datasets of zebrafish embryos.

The pipeline performs:

- Image preprocessing

- Channel extraction and splitting

- Background subtraction

- Nuclear segmentation using Cellpose-SAM

- Segmentation mask filtering

- Per-nucleus intensity quantification

- Pairwise channel correlation analysis

- Statistical analysis

- Automated figure generation

## Scripts

### `260125_unexpanded_image_processing_pipeline.py`

Main analysis pipeline.

### `260518_plot_intensity_results_and_stats_with_exclude_optionl.py`

Generates statistical analyses, publication-quality plots, and PDF reports from quantification outputs.

### `run_cellpose_in_batches_for_pipeline_250702_cellpose-sam_model_gpu5000_test.sh`

Runs Cellpose-SAM segmentation using a pretrained model.

---

## Input Naming Convention

The pipeline supports either:

### One directory per image

```text

raw/

├── 260616_WT_H3K9me3-ab5-Alexa647_DAPI_6hpf_60x_E_1/

├── 260616_WT_H3K9me3-ab5-Alexa647_DAPI_6hpf_60x_E_2/

└── ...

```

### Flat image directory

```text

raw/

├── 260616_WT_H3K9me3-ab5-Alexa647_DAPI_6hpf_60x_E_1.ome.tif

├── 260616_WT_H3K9me3-ab5-Alexa647_DAPI_6hpf_60x_E_2.ome.tif

└── ...

```

Folder names or image filenames are used as unique image identifiers throughout the workflow and are propagated into segmentation masks, quantification outputs, correlation analyses, and plots.

For proper grouping during downstream statistical analysis and plotting:

- Experimental conditions should be encoded consistently.

- Components should be separated by underscores.

- Embryo replicates should be identified using the suffix `E_<number>`.

Example:

```text

260616_WT_H3K9me3-ab5-Alexa647_DAPI_6hpf_60x_E_1

260616_flavopiridol_H3K9me3-ab5-Alexa647_DAPI_6hpf_60x_E_4

```

---

## Configuration

The pipeline is controlled through a JSON configuration file.

Before running the pipeline, update all paths in the configuration file to match your system, including:

- Input and output directories

- Cellpose segmentation script

- Cellpose model location

- Plotting script

- Correlation-analysis script

All downstream processing is launched automatically through the configuration file.

---

## Cellpose Model

The pipeline uses a fine-tuned Cellpose-SAM model for nuclear segmentation.

The model is not included in this repository and must be obtained separately.

Update the model path within:

```text

run_cellpose_in_batches_for_pipeline.sh

```

and make the script executable:

```bash

chmod +x run_cellpose_in_batches_for_pipeline.sh

```

---

## Running the Pipeline

```bash

python unexpanded_image_processing_pipeline.py config.json

```

---

## Outputs

The pipeline generates:

- Split-channel images

- Segmentation masks

- Filtered segmentation masks

- Per-nucleus quantification CSV files

- Correlation-analysis outputs

- Statistical summary tables

- Publication-quality figures

- PDF statistical reports

- Run logsa
