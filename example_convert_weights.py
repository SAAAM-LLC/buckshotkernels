#!/usr/bin/env python3
"""
BuckshotKernels - Convert Normal Weights to Ternary
====================================================
Shows how to take regular neural network weights and convert them to ternary.
"""

import numpy as np
from buckshotkernels import TernaryKernelManager

print("🦌 BuckshotKernels - Weight Conversion Example\n")

# Step 1: Simulate some "normal" neural network weights
print("Step 1: Creating fake neural network weights...")
# Imagine these came from a trained model
layer1_weights = np.random.randn(512, 784).astype(np.float32)  # Input layer
layer2_weights = np.random.randn(256, 512).astype(np.float32)  # Hidden layer
print(f"  Layer 1: {layer1_weights.shape}, dtype={layer1_weights.dtype}")
print(f"  Layer 2: {layer2_weights.shape}, dtype={layer2_weights.dtype}\n")

# Step 2: Convert to ternary
print("Step 2: Converting to ternary...")
# Simple threshold: anything close to zero becomes 0
threshold = 0.1

def to_ternary(weights, threshold=0.1):
    """Convert float weights to ternary {-1, 0, 1}"""
    ternary = np.zeros_like(weights, dtype=np.int8)
    ternary[weights > threshold] = 1
    ternary[weights < -threshold] = -1
    return ternary

layer1_ternary = to_ternary(layer1_weights, threshold)
layer2_ternary = to_ternary(layer2_weights, threshold)

print(f"  Layer 1 ternary: {layer1_ternary.shape}, dtype={layer1_ternary.dtype}")
print(f"  Layer 2 ternary: {layer2_ternary.shape}, dtype={layer2_ternary.dtype}\n")

# Step 3: Check sparsity (how many zeros)
def check_sparsity(ternary_weights):
    """See how many zeros we have (good for speed!)"""
    total = ternary_weights.size
    zeros = np.sum(ternary_weights == 0)
    ones = np.sum(ternary_weights == 1)
    neg_ones = np.sum(ternary_weights == -1)
    return {
        'zeros': zeros / total * 100,
        'ones': ones / total * 100,
        'neg_ones': neg_ones / total * 100
    }

print("Step 3: Checking sparsity...")
l1_stats = check_sparsity(layer1_ternary)
l2_stats = check_sparsity(layer2_ternary)

print(f"  Layer 1: {l1_stats['zeros']:.1f}% zeros, {l1_stats['ones']:.1f}% ones, {l1_stats['neg_ones']:.1f}% -1s")
print(f"  Layer 2: {l2_stats['zeros']:.1f}% zeros, {l2_stats['ones']:.1f}% ones, {l2_stats['neg_ones']:.1f}% -1s")
print(f"  (More zeros = faster computation!)\n")

# Step 4: Use them with BuckshotKernels
print("Step 4: Using ternary weights with BuckshotKernels...")
kernels = TernaryKernelManager()

# Simulate an input
input_data = np.random.choice([-1, 0, 1], size=(32, 784), p=[0.25, 0.5, 0.25]).astype(np.int8)

# Forward pass through layer 1
hidden = kernels.ternary_matmul(input_data, layer1_ternary.T)
print(f"  Hidden layer output: {hidden.shape}")

# Forward pass through layer 2
output = kernels.ternary_matmul(hidden.astype(np.int8), layer2_ternary.T)
print(f"  Final output: {output.shape}\n")

print("🔥 Done! You just ran a ternary neural network!")
print("\n💡 Tips:")
print("  - Lower threshold = more non-zero values = more accurate but slower")
print("  - Higher threshold = more zeros = faster but less accurate")
print("  - Typical sweet spot: threshold between 0.05 and 0.15")
