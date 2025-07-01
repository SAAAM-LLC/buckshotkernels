# 🦌 BuckshotKernels - SAAAM LLC

**Custom CUDA & SIMD C Kernels for Ternary Neural Networks — Written in the Woods, Built for Speed**

---

## What the hell is this?

BuckshotKernels is a collection of **hand-rolled CUDA and CPU (SIMD) matrix multiplication, convolution, and quantization kernels** for neural nets that use {-1, 0, 1} values.  
It’s built for those who want raw performance and full control—no Python training wheels, no bloated frameworks, no safe spaces.

You got a GPU? It’ll use it.  
Only got a CPU? It’ll squeeze every last cycle outta that bastard with AVX2/AVX512 or NEON if you’re on ARM.  
No third-party ops—**this is as close to the metal as you can get without chewing on solder.**

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

##  License

#  MIT, because GPL is for lawyers and Apache is for folks with something to lose.
Contributing

Pull requests, issues, and creative swearing all welcome.
Disclaimer

No warranty, no guarantees, and if you burn down a datacenter, you’re on your own.
But you’ll have the fastest damn ternary kernels in the county.

Example: Custom Ternary Matrix Multiply
```
from buckshotkernels import TernaryKernelManager

kernels = TernaryKernelManager()
A = np.random.choice([-1, 0, 1], size=(512, 512), p=[0.25, 0.5, 0.25]).astype(np.int8)
B = np.random.choice([-1, 0, 1], size=(512, 512), p=[0.25, 0.5, 0.25]).astype(np.int8)

C = kernels.ternary_matmul(A, B)  # Automatically uses fastest kernel
```
