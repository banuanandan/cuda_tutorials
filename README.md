# CUDA Tutorials

This repository contains CUDA programming tutorials and examples to help you learn GPU programming with NVIDIA CUDA.

## Overview

A collection of CUDA tutorials covering fundamental concepts and advanced techniques for GPU programming.

## Usage

Browse the tutorial directories and follow the examples to learn CUDA programming concepts. Each tutorial includes source code and documentation.

## Environments Tested

| Environment |     Version      | Status |
|-------------|------------------|--------|
| Ubuntu OS   |      22.04       |   ✅   |
| Kernel      | 6.8.0-79-generic |   ✅   |
| CUDA        |      12.6        |   ✅   |
| GCC         |      11.4.0      |   ✅   |
| Bazel       |      8.2.1       |   ✅   |

## Build Instructions

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Bazel build system installed

### Build with Bazel

```bash
# Build all targets
bazel build //...

# Build specific tutorial
bazel build //path/to/tutorial_name:target_name
```

### Run Instructions

```bash
# Run specific tutorial
bazel run //path/to/tutorial_name:target_name

# Run with arguments
bazel run //path/to/tutorial_name:target_name -- --arg1 value1
```