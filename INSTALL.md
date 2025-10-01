# BuckshotKernels Installation Guide

## Quick Install (For People Who Just Want It To Work)

```bash
# 1. Go to the directory
cd ~/Desktop/SAAAM_LLC/SAAAM_DEV_AREA/TERNARY/buckshotkernels-main

# 2. Install it
pip install -e .

# 3. Test it
python3 -c "from buckshotkernels import TernaryKernelManager; print('Success!')"
```

That's it. If it says "Success!" you're done.

---

## Detailed Install (For When Things Go Wrong)

### Step 1: Check Your Python

You need Python 3.8 or newer:

```bash
python3 --version
```

If it's less than 3.8, update Python first.

### Step 2: Install NumPy

```bash
pip3 install numpy
```

### Step 3: (Optional) Install CUDA Toolkit for GPU Speed

Check if you have it:

```bash
nvcc --version
```

If that works, you're good. If not:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit
```

**Other Linux:**
Download from NVIDIA's website: https://developer.nvidia.com/cuda-downloads

**Mac:**
Macs don't have NVIDIA GPUs (except old ones). CPU-only for you.

**Windows:**
Good luck, you're on your own. (JK - download CUDA toolkit from NVIDIA's website)

### Step 4: Install BuckshotKernels

**Option A: Development Install (Recommended)**

This lets you edit the code and see changes immediately:

```bash
cd ~/Desktop/SAAAM_LLC/SAAAM_DEV_AREA/TERNARY/buckshotkernels-main
pip install -e .
```

**Option B: Regular Install**

```bash
cd ~/Desktop/SAAAM_LLC/SAAAM_DEV_AREA/TERNARY/buckshotkernels-main
pip install .
```

**Option C: From GitHub (When You Publish It)**

```bash
pip install git+https://github.com/SAAAM-LLC/buckshotkernels.git
```

### Step 5: Verify It Works

```bash
python3 example_basic.py
```

If you see matrix multiplication results, you're golden!

---

## Troubleshooting

### "No module named 'buckshotkernels'"

**Problem:** It's not installed.

**Fix:**
```bash
cd ~/Desktop/SAAAM_LLC/SAAAM_DEV_AREA/TERNARY/buckshotkernels-main
pip install -e .
```

### "CUDA Toolkit: Not found"

**Problem:** CUDA isn't installed or can't be found.

**Fix 1:** Install CUDA toolkit (see Step 3 above)

**Fix 2:** Set CUDA_HOME manually:
```bash
export CUDA_HOME=/usr
python3 your_script.py
```

**Fix 3:** Just use CPU - it still works!

### Compilation Errors

**Problem:** Can't compile the C/CUDA kernels.

**Fix:** Make sure you have a C compiler:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# Mac
xcode-select --install
```

### "Permission denied"

**Problem:** You don't have write permissions.

**Fix:** Use `--user` flag:
```bash
pip install -e . --user
```

Or use a virtual environment (better):
```bash
python3 -m venv ~/venv_buckshot
source ~/venv_buckshot/bin/activate
pip install -e .
```

---

## Uninstall

If you want to remove it:

```bash
pip uninstall buckshotkernels
```

---

## What Gets Installed

When you run `pip install -e .`, it:

1. Creates a link to the code (so you can edit it)
2. Installs NumPy if you don't have it
3. Adds `buckshotkernels` to your Python path
4. Does NOT compile kernels yet (that happens on first import)

The kernels compile the **first time** you import and use them. This takes a few seconds but only happens once.

---

## Virtual Environment (Recommended)

If you want to keep things clean:

```bash
# Create virtual environment
python3 -m venv ~/buckshot_env

# Activate it
source ~/buckshot_env/bin/activate

# Install
cd ~/Desktop/SAAAM_LLC/SAAAM_DEV_AREA/TERNARY/buckshotkernels-main
pip install -e .

# Use it
python3 example_basic.py

# When done
deactivate
```

---

## System Requirements

**Minimum:**
- Python 3.8+
- NumPy
- 4GB RAM
- Any CPU

**Recommended:**
- Python 3.10+
- NumPy 1.24+
- 16GB+ RAM
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+
- GCC 9.0+ or equivalent C compiler

**Tested On:**
- Ubuntu 22.04+
- NVIDIA TITAN X Pascal
- CUDA 12.2
- Python 3.13

---

## Still Having Problems?

1. Read the error message (it usually tells you what's wrong)
2. Check README_COMPLETE.md for solutions
3. Make sure all requirements are installed
4. Try in a fresh virtual environment
5. Ask Michael (but only after trying steps 1-4)

---

**Good luck! 🦌**
