#!/usr/bin/env python3
"""
BuckshotKernels Setup
=====================
Install with: pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read the README for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README_COMPLETE.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Fast ternary neural network kernels with CUDA and CPU optimization"

setup(
    name='buckshotkernels',
    version='1.0.0',
    author='Michael @ SAAAM LLC',
    author_email='your-email@saaam.com',  # Update this
    description='Custom CUDA & SIMD C kernels for ternary neural networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SAAAM-LLC/buckshotkernels',  # Update when you make a repo

    # Just include the main Python file
    py_modules=['buckshotkernels'],

    # Include the C and CUDA source files
    package_data={
        '': ['*.c', '*.cu', '*.md'],
    },
    include_package_data=True,

    # Requirements
    install_requires=[
        'numpy>=1.19.0',
    ],

    # Optional dependencies
    extras_require={
        'dev': ['pytest', 'black', 'flake8'],
    },

    # Python version
    python_requires='>=3.8',

    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C',
        'Programming Language :: C++',
    ],

    # Keywords
    keywords='ternary neural-networks cuda gpu optimization machine-learning deep-learning',

    # Entry points (if you want command-line tools)
    entry_points={
        'console_scripts': [
            # Add if you want: 'buckshot-benchmark=buckshotkernels:run_benchmark',
        ],
    },

    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/SAAAM-LLC/buckshotkernels/issues',
        'Source': 'https://github.com/SAAAM-LLC/buckshotkernels',
        'Documentation': 'https://github.com/SAAAM-LLC/buckshotkernels/blob/main/README_COMPLETE.md',
    },
)
