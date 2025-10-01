#!/usr/bin/env python3
"""
BuckshotKernels - Benchmark Example
====================================
Compare CPU vs GPU performance at different matrix sizes.
"""

import numpy as np
import time
from buckshotkernels import TernaryKernelManager

print("🦌 BuckshotKernels - CPU vs GPU Benchmark\n")

# Initialize
print("Loading kernels...")
kernels = TernaryKernelManager()
print("✅ Ready!\n")

# Test different sizes
sizes = [128, 256, 512, 1024]

print("=" * 70)
print(f"{'Size':<15} {'CPU Time':<15} {'GPU Time':<15} {'Speedup':<15}")
print("=" * 70)

for size in sizes:
    # Create test matrices
    A = np.random.choice([-1, 0, 1], size=(size, size), p=[0.25, 0.5, 0.25]).astype(np.int8)
    B = np.random.choice([-1, 0, 1], size=(size, size), p=[0.25, 0.5, 0.25]).astype(np.int8)

    # CPU benchmark
    start = time.time()
    C_cpu = kernels.ternary_matmul(A, B, device='cpu')
    cpu_time = time.time() - start

    # GPU benchmark (skip first run overhead)
    if size == sizes[0]:
        _ = kernels.ternary_matmul(A, B, device='cuda')  # warmup

    start = time.time()
    C_gpu = kernels.ternary_matmul(A, B, device='cuda')
    gpu_time = time.time() - start

    # Calculate speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0

    # Print results
    print(f"{size}x{size:<10} {cpu_time*1000:>10.2f}ms  {gpu_time*1000:>10.2f}ms  {speedup:>10.2f}x")

print("=" * 70)
print("\n📊 Conclusion:")
print("  - Small matrices (< 256): CPU is competitive")
print("  - Large matrices (>= 512): GPU dominates")
print("  - Use device='auto' to let it pick for you!")
