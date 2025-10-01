# 🦌 BuckshotKernels - SAAAM LLC

**Custom CUDA & SIMD C Kernels for Ternary Neural Networks — Written in the Woods, Built for Speed**
🏆 Peak Kernel Performance: 1.43 TRILLION ops/sec (2048x2048 CUDA matmul, TITAN X Pascal)
---

## What the hell is this?

BuckshotKernels is a collection of **hand-rolled CUDA and CPU (SIMD) matrix multiplication, convolution, and quantization kernels** for neural nets that use {-1, 0, 1} values.  
It’s built for those who want raw performance and full control—no Python training wheels, no bloated frameworks, no safe spaces.
Fast as hell ternary ({-1, 0, 1}) matrix operations for neural networks. No PyTorch bullshit, just raw CUDA and optimized C code.

You got a GPU? It’ll use it.  
Only got a CPU? It’ll squeeze every last cycle outta that bastard with AVX2/AVX512 or NEON if you’re on ARM.  
No third-party ops—**this is as close to the metal as you can get without chewing on solder.**

Think of it like this: - Normal AI: “Let me multiply 3.14159 × 2.71828…” (slow as fuck) - Ternary AI: “That’s a 1, add it. That’s a -1, subtract it. That’s a
0, skip it.” (fast as hell)
---

## Features

- 🔥 **Raw CUDA Kernels:** Real C++ code, compiled with `nvcc`, optimized for shared memory, tile size, and ternary skips.
- ⚡ **SIMD-Vectorized CPU Kernels:** AVX2/AVX512 or NEON, hand-tuned for modern CPUs.
- 🪓 **Automatic Hardware Detection:** Picks best kernels and compilers based on your machine.
- 💣 **Bit-Packing & Quantization:** Fastest possible float-to-ternary mapping and memory packing.
- 🧨 **No Dependencies:** No PyTorch, no Tensorflow, no CuPy—just you, your hardware, and the will to go fast.
- 🦍 **Benchmarking Included:** Run real benchmarks against NumPy/CuPy and see how much faster you can go.

---

## Usage

### 1. Clone and build

```bash
git clone https://github.com/SAAAM-LLC/buckshotkernels.git
cd buckshotkernels
pip install -r requirements.txt  # Only needs numpy (and cupy if you want CUDA fallback)
```

### 2. Test it works

```bash
cd buckshotkernels-main
python3 -c "from buckshotkernels import TernaryKernelManager; print('It works!')"
```

If you see "It works!" you're done. If you see errors, read them and Google it like a normal person.

---

## How to Use It (For Dummies)

### Example 1: Basic Matrix Multiply

```python
import numpy as np
from buckshotkernels import TernaryKernelManager

# Initialize (this compiles the kernels first time - takes a few seconds)
kernels = TernaryKernelManager()

# Make some ternary matrices (values must be -1, 0, or 1)
A = np.random.choice([-1, 0, 1], size=(512, 512)).astype(np.int8)
B = np.random.choice([-1, 0, 1], size=(512, 512)).astype(np.int8)

# Multiply them (auto-picks GPU if available, CPU if not)
C = kernels.ternary_matmul(A, B)

print(f"Result shape: {C.shape}")
print(f"Result dtype: {C.dtype}")  # int32 because adding lots of small numbers
```

### Example 2: Force CPU or GPU

```python
# Use CPU only (good for small matrices)
C_cpu = kernels.ternary_matmul(A, B, device='cpu')

# Use GPU only (good for big matrices)
C_gpu = kernels.ternary_matmul(A, B, device='cuda')

# Auto-pick (default - it's smart about it)
C_auto = kernels.ternary_matmul(A, B, device='auto')
```

### Example 3: Convert Regular Weights to Ternary

```python
# You have some normal neural network weights
normal_weights = np.random.randn(512, 512).astype(np.float32)

# Convert to ternary (anything close to 0 becomes 0, positive becomes 1, negative becomes -1)
ternary_weights = np.zeros_like(normal_weights, dtype=np.int8)
ternary_weights[normal_weights > 0.1] = 1
ternary_weights[normal_weights < -0.1] = -1

# Now use them
C = kernels.ternary_matmul(ternary_weights.astype(np.int8), B)
```

---

## Performance - The Numbers Don't Lie

Tested on NVIDIA TITAN X Pascal:

| Matrix Size | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 128x128     | 0.30ms   | ~4s*     | N/A     |
| 256x256     | 1.95ms   | 1.10ms   | 1.77x   |
| 512x512     | 14.69ms  | 1.59ms   | **9.23x** |
| 1024x1024   | ~150ms   | ~5ms     | **30x+** |

*First GPU call has compilation overhead, then it's fast

**TL;DR:** For big matrices, GPU is 10-30x faster. For tiny ones, CPU is fine.

---

## Common Problems (And How to Fix Them)

### "CUDA Toolkit: Not found"

**Problem:** You don't have CUDA installed or it's not in the right place.

**Fix:**
```bash
# Install it
sudo apt-get install nvidia-cuda-toolkit

# Or set the path manually
export CUDA_HOME=/usr
```

### "Import failed" or syntax errors

**Problem:** The code has a bug (probably my fault).

**Fix:** Check if there are any files with `.py` extension that have syntax errors. Read the error message - it usually tells you the line number.

### "Compilation failed"

**Problem:** Your compiler is missing or broken.

**Fix:**
```bash
# Make sure you have gcc
sudo apt-get install build-essential
```

### GPU is slower than CPU for small matrices

**Not a problem:** That's normal. GPU has overhead. Use CPU for small stuff, GPU for big stuff. The `device='auto'` option handles this.

---

## FAQ (Frequently Asked Questions)

**Q: Do I need a GPU?**
A: Nope! CPU works fine, just slower for big matrices.

**Q: Will this work on Mac?**
A: CPU version yes. GPU version no (Macs don't have NVIDIA GPUs unless you have an old one).

**Q: How is this different from PyTorch?**
A: This is bare metal - no framework overhead. PyTorch is easier but slower and uses more memory.

**Q: Can I use this with PyTorch/TensorFlow?**
A: Yeah, just convert between NumPy and PyTorch tensors. It's annoying but works.

**Q: What if I have multiple GPUs?**
A: Currently uses GPU 0. Multi-GPU support would be cool but I haven't built it yet.

**Q: Is this production-ready?**
A: It works and it's fast. Is it battle-tested on millions of users? No. Use at your own risk.

---

## What's Inside (File Structure)

```
buckshotkernels-main/
├── buckshotkernels.py      # Main Python code (compiles & runs kernels)
├── ternary_cpu_kernels.c   # C code for CPU (AVX2 optimized)
├── ternary_kernels.cu      # CUDA code for GPU
├── README.md               # Original short README
└── README_COMPLETE.md      # This file (the good one)
```

---

## Technical Details (For Nerds)

### How Ternary Works

Regular matmul: `C[i,j] = sum(A[i,k] * B[k,j])`

Ternary matmul optimization:
- If A[i,k] == 0 or B[k,j] == 0: skip (no multiply needed)
- If A[i,k] == 1: `C[i,j] += B[k,j]` (just add, no multiply)
- If A[i,k] == -1: `C[i,j] -= B[k,j]` (just subtract, no multiply)

Result: Way fewer operations, way more speed.

### CPU Optimizations

- **Block tiling** (64x64 blocks for cache locality)
- **AVX2 SIMD** (process 32 values at once)
- **Loop unrolling** (compiler does this automatically)
- **Zero skipping** (don't process zeros)

### GPU Optimizations

- **Shared memory tiling** (16x16 tiles in fast on-chip memory)
- **Coalesced memory access** (efficient GPU memory reads)
- **Zero skipping** (same as CPU)
- **Compute capability targeting** (auto-detects your GPU and compiles for it)

---

## Future Improvements (Maybe)

- [ ] Bit-packing (store 4 ternary values per byte instead of 1 per byte)
- [ ] Multi-GPU support
- [ ] Fused operations (quantize + matmul in one kernel)
- [ ] INT4 support (for even more compression)
- [ ] AMD ROCm support (for non-NVIDIA GPUs)

---

###  Why Buckshot?

##  Because sometimes you gotta spray and pray:

    Want max speed with zero bloat?

    Need to slap together a ternary inference engine in the back of a pickup?

    Don’t trust other people’s code or “best practices?”

#  This is for you.
Supported Platforms

    Linux, Windows, Mac (if you hate yourself)

    x86_64 (AVX2/AVX512), ARM64 (NEON)

    NVIDIA GPUs (CUDA 11.x+)

---

## Credits

**Built by:** Michael @ SAAAM LLC
**Inspired by:** Frustration with slow AI frameworks
**Tested on:** NVIDIA TITAN X Pascal, lots of coffee, and determination

---

## Need Help?

1. Read this README again (seriously, the answer is probably here)
2. Check the error message (it usually tells you what's wrong)
3. Google the error (someone else has probably had it)
4. Ask me (Michael) but only after you've tried steps 1-3

---    
    
##  License

#  MIT, because GPL is for lawyers and Apache is for folks with something to lose.
Contributing

Pull requests, issues, and creative swearing all welcome.
Disclaimer

No warranty, no guarantees, and if you burn down a datacenter, you’re on your own.
But you’ll have the fastest damn ternary kernels in the county.
