#!/bin/bash
# setup_env.sh — Run once on HPC login node to set up the conda environment
# Usage: bash setup_env.sh

echo "Setting up Urban Tree Monitoring environment..."

# Create and activate environment
conda create -n urban_tree python=3.10 -y
source activate urban_tree

# PyTorch with CUDA 12 (H100 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorFlow (H100-compatible, self-contained CUDA/cuDNN)
pip install "tensorflow[and-cuda]"

# Google Earth Engine + geospatial
pip install earthengine-api geemap
pip install rasterio geopandas shapely pyproj

# ML / data science
pip install numpy pandas matplotlib scikit-learn
pip install albumentations segmentation-models-pytorch
pip install pillow tqdm requests

# OpenCV (headless — no display on HPC)
pip install opencv-python-headless

# Jupyter (for graphical exploration via port-forwarding)
pip install jupyterlab ipywidgets

echo ""
echo "========================================"
echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. conda activate urban_tree"
echo "  2. earthengine authenticate"
echo "  3. nvidia-smi -L   (get your MIG UUID)"
echo "  4. Update job_gpu.pbs with your MIG UUID"
echo "  5. qsub job_cpu.pbs   (run preprocessing)"
echo "  6. qsub job_gpu.pbs   (run training)"
echo "========================================"
