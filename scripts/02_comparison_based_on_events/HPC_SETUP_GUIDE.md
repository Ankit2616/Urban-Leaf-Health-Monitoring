# 🚀 H100 HPC Setup Guide — Urban Tree Monitoring System
### NIT Jalandhar | CSE HPC (10.10.11.201)

---

## PHASE 0: PRE-HPC CHECKLIST (Do on your LOCAL machine first)

Before touching the HPC, validate locally:
- [ ] Google Earth Engine auth works (`ee.Initialize()` runs without error)
- [ ] Your GEE Project ID is set correctly
- [ ] At least 1 sample image downloads successfully (run `01_data_collection.py` for 5 images)
- [ ] All Python packages install without conflict on your machine

> ⚠️ **Rule #5 from HPC Manual**: You MUST validate scripts locally before running on HPC.

---

## PHASE 1: GETTING ACCESS

1. Log into NITJ ERP → **Equipment Booking Module** → Book HPC
2. Wait for admin approval → You'll receive email with credentials
3. Access is granted for **15 days at a time** — plan your runs accordingly

---

## PHASE 2: CONNECT & TRANSFER FILES

### 2.1 SSH Login
```bash
ssh username@10.10.11.201
# Password won't show on screen — that's normal
```

### 2.2 Create Your Project Folder (in /Data — NOT home)
```bash
# Large files MUST go in /Data (not ~/)
mkdir -p /Data/username/urban_tree_project
mkdir -p /Data/username/urban_tree_project/raw_tiff
mkdir -p /Data/username/urban_tree_project/processed
mkdir -p /Data/username/urban_tree_project/augmented
mkdir -p /Data/username/urban_tree_project/models
mkdir -p /Data/username/urban_tree_project/results
mkdir -p /Data/username/urban_tree_project/logs
```

### 2.3 Transfer Your Code (from local terminal)
```bash
# Transfer entire project folder
scp -r ./urban_tree_project/ username@10.10.11.201:/Data/username/

# Or use WinSCP (drag-and-drop GUI) — Host: 10.10.11.201, Port: 22
```

---

## PHASE 3: ENVIRONMENT SETUP (Run on HPC login node)

Run `setup_env.sh` OR paste these commands manually:

```bash
# Step 1: Create conda environment
conda create -n urban_tree python=3.10 -y
conda activate urban_tree

# Step 2: Install PyTorch with CUDA 12 (for H100)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install TensorFlow (H100-compatible, bundles CUDA/cuDNN)
pip install "tensorflow[and-cuda]"

# Step 4: Install GEE + geospatial stack
pip install earthengine-api geemap
pip install rasterio geopandas shapely pyproj
pip install numpy pandas matplotlib scikit-learn pillow tqdm
pip install albumentations segmentation-models-pytorch
pip install opencv-python-headless  # headless = no display needed on HPC

# Step 5: Authenticate GEE (one-time, on login node)
earthengine authenticate
# Follow the URL, paste the token back

# Step 6: Verify GPU is visible
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## PHASE 4: FIND YOUR MIG UUID (Critical for H100)

```bash
nvidia-smi -L
```
Output looks like:
```
GPU 0: NVIDIA H100 NVL ...
  MIG 3g.47gb  Device 0: (UUID: MIG-4d38d5cf-c802-5308-80b8-251f3cec7480)
  MIG 3g.47gb  Device 1: (UUID: MIG-9b919b48-2562-5791-8778-080ddb153351)
...
```
**Pick any free MIG instance** and copy its UUID. You'll paste it into your PBS script.

---

## PHASE 5: JUPYTER NOTEBOOK (Graphical Access via Port Forwarding)

For visual work (viewing NDVI maps, segmentation outputs), run Jupyter remotely:

### Step 1: On HPC, start Jupyter (no browser)
```bash
conda activate urban_tree
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
# Copy the token from the output, e.g.: token=abc123xyz
```

### Step 2: On your LOCAL machine, open SSH tunnel
```bash
ssh -L 8888:localhost:8888 username@10.10.11.201
```

### Step 3: Open in your browser
```
http://localhost:8888/?token=abc123xyz
```

> Note: Jupyter is for **exploration only**. For actual training, always use PBS scripts.

---

## PHASE 6: PBS JOB SUBMISSION

### CPU Job (data download, preprocessing) → `job_cpu.pbs`
```bash
qsub job_cpu.pbs
```

### GPU Job (model training) → `job_gpu.pbs`
```bash
# First update CUDA_VISIBLE_DEVICES in job_gpu.pbs with your MIG UUID
qsub job_gpu.pbs
```

### Monitor Jobs
```bash
qstat -u your_username          # See job status (Q=queued, R=running, E=exiting)
tracejob 3028                   # Debug a specific job ID
tail -f output_gpu.log          # Watch live output
```

### Cancel a Job
```bash
qdel 3028                       # Replace with your job ID
```

---

## PHASE 7: FULL PIPELINE EXECUTION ORDER

```
Step 1 [cpuq]  → 01_data_collection.py     (Downloads TIFF from GEE)
Step 2 [cpuq]  → 02_preprocessing.py       (Cloud mask, normalize, patch)
Step 3 [cpuq]  → 03_augmentation.py        (4000-5000 augmented samples)
Step 4 [workq] → 04_train.py               (U-Net training on H100)
Step 5 [cpuq]  → 05_predict_visualize.py   (Inference + map generation)
```

---

## PHASE 8: BACKUP YOUR RESULTS

```bash
# From LOCAL machine — pull results back after computation
scp -r username@10.10.11.201:/Data/username/urban_tree_project/results ./

# OR push to your cloud storage before your 15-day access expires
```

---

## QUICK REFERENCE

| Task | Command |
|------|---------|
| Login | `ssh username@10.10.11.201` |
| Check GPUs | `nvidia-smi -L` |
| Activate env | `conda activate urban_tree` |
| Submit CPU job | `qsub job_cpu.pbs` |
| Submit GPU job | `qsub job_gpu.pbs` |
| Check job status | `qstat -u username` |
| View live logs | `tail -f output_gpu.log` |
| Debug job | `tracejob <job_id>` |

---

## ACKNOWLEDGEMENT (Required for publications)
> "The authors gratefully acknowledge the High Performance Computing (HPC) facility provided by Dr. B. R. Ambedkar National Institute of Technology Jalandhar (NIT Jalandhar) for supporting the computational requirements of this research work."
