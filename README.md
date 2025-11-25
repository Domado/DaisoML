# DaisoML
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/Domado/dcp/actions)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)
![Version](https://img.shields.io/badge/version-0.1.0-informational)

🤖 Cross platform, high-speed, advanced artificial intelligence inference library, made by Daiso

## Overview

DaisoML is a lightweight, high-performance C++ inference library designed for Transformer-based Large Language Models (LLMs). Built with standard C++, it provides a clean and modular implementation of modern architecture components found in state-of-the-art models, focusing on portability and educational value.

## Features

* **Modern C++ implementation:** Written in C++ with minimal external dependencies.
* **Advanced Transformer Architecture:**
    * **RMSNorm:** Pre-normalization for stable training and inference.
    * **RoPE (Rotary Positional Embeddings):** Applied in the Attention layer for better relative position handling.
    * **SwiGLU:** Gated linear unit activation function used in FeedForward layers.
    * **KV-Caching:** Efficient handling of Key and Value states for accelerated generation.
* **Custom Tensor Engine:** Includes a standalone tensor library handling matrix multiplication, softmax, and other element-wise operations.
* **Binary Model Format:** Efficient loading via a custom, lightweight binary format.

## Project Structure

The codebase is organized as follows:

* `main.cpp`: Entry point for the CLI inference application.
* `model.cpp` / `model.h`: The core Transformer model definition and forward pass logic.
* `layers/`: Implementation of neural network layers:
    * `attention.cpp`: Multi-head attention with RoPE.
    * `feed_forward.cpp`: SwiGLU feed-forward network.
    * `rmsnorm.cpp`: Root Mean Square Layer Normalization.
    * `embedding.cpp`: Token embedding lookup.
* `tensor.cpp` / `tensor.h`: Basic N-dimensional tensor class and math operations.
* `sampler.cpp`: Logic for token sampling (Temperature, Top-P).
* `tokenizer.cpp`: Tokenizer interface (currently a placeholder implementation).
* `create_dummy_model.cpp`: Utility to generate random model weights for testing.

## Build Instructions

DaisoML uses CMake for building. Ensure you have a C++17 compatible compiler and CMake installed.

### Steps

1.  Create a build directory:
    ```bash
    mkdir build
    cd build
    ```

2.  Configure the project:
    ```bash
    cmake ..
    ```

3.  Compile:
    ```bash
    make
    ```

This will produce two executables in the `build` directory:
* `daiso_run`: The main inference engine.
* `create_dummy_model`: A tool to generate test model files.

## Usage

### 1. Generating a Test Model
Since this is a custom library, you first need a model file in the specific DaisoML binary format. You can generate a "dummy" model with random weights to test the pipeline:

```bash
./create_dummy_model
````

*Output:* This will create `dummy_model.bin` in your current directory.

### 2\. Running Inference

Run the inference engine by providing the path to the model file:

```bash
./daiso_run dummy_model.bin
```

**Expected Output:**
The program will load the model configuration, process a hardcoded prompt ("Hello, my name is"), and generate a sequence of tokens.

> **Note:** Since the dummy model uses random weights and a dummy tokenizer, the generated text will be nonsensical characters.

## Technical Details

### Model File Format

DaisoML models use a specific binary structure starting with the magic number `0x64616973` ("dais").

  * **Header:** Contains metadata like `dim`, `n_layers`, `n_heads`, `vocab_size`, etc.
  * **Weights:** Raw float data for tensors stored in a strict order (Embeddings -\> Layer Weights -\> Output Head).

### Current Limitations & Roadmap

  * **Tokenizer:** The current tokenizer is a dummy implementation (char-to-int). Future updates will support BPE or SentencePiece.
  * **Sampling:** The sampler currently implements a basic argmax strategy. Full temperature and top-p sampling logic is planned.
  * **Optimization:** Matrix multiplication is currently a naive implementation. AVX/SIMD optimizations are planned for speedups.

