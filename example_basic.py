#!/usr/bin/env python3
"""
BuckshotKernels - Basic Example
================================
The simplest possible example. Just multiply two ternary matrices.
"""

import numpy as np
from buckshotkernels import TernaryKernelManager

print("🦌 BuckshotKernels - Basic Example\n")

# Step 1: Initialize the kernel manager
print("Step 1: Loading kernels...")
kernels = TernaryKernelManager()
print("✅ Kernels loaded!\n")

# Step 2: Create some ternary matrices
print("Step 2: Creating 512x512 ternary matrices...")
# Values MUST be -1, 0, or 1
# dtype MUST be int8
A = np.random.choice([-1, 0, 1], size=(512, 512), p=[0.25, 0.5, 0.25]).astype(np.int8)
B = np.random.choice([-1, 0, 1], size=(512, 512), p=[0.25, 0.5, 0.25]).astype(np.int8)
print(f"✅ Created A: {A.shape}, dtype={A.dtype}")
print(f"✅ Created B: {B.shape}, dtype={B.dtype}\n")

# Step 3: Multiply them
print("Step 3: Running ternary matrix multiplication...")
C = kernels.ternary_matmul(A, B)
print(f"✅ Result C: {C.shape}, dtype={C.dtype}\n")

# Step 4: Show some results
print("Step 4: Sample values from result:")
print(f"  C[0, :10] = {C[0, :10]}")
print(f"  C[100, :10] = {C[100, :10]}\n")

print("🔥 Done! That was easy.")
