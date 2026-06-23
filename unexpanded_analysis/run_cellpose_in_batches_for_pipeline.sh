#!/bin/bash
# Run Cellpose on a specified directory passed as an argument

# Check if the directory is passed as an argument
if [ -z "$1" ]; then
  echo "Error: No directory provided. Usage: ./run_cellpose.sh <directory>"
  exit 1
fi

# Set the directory from the argument
directory="$1"

# Pretrained model path 
pretrained_model="REPLACE_WITH_MODEL_PATH"

# Get a list of files in the directory
files=("$directory"/*)

# Loop through the files, processing one file at a time
for file in "${files[@]}"; do
    command="python -m cellpose --image_path \"$file\" --pretrained_model \"$pretrained_model\" --use_gpu --save_tif --verbose --no_npy --in_folders --do_3D --exclude_on_edges --flow3D_smooth 2.0 --anisotropy 2.75 --batch_size 96"
    echo "Running command: $command"
    eval "$command"
done
