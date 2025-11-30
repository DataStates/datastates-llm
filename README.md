# DataStates-LLM: Asynchronous I/O Engine

---

[![License](https://img.shields.io/github/license/DataStates/datastates-llm)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2406.10707-B31B1B.svg)](https://arxiv.org/abs/2406.10707)

DataStates-LLM is a high-performance I/O engine for GPU-accelerated workloads, with a particular focus on large-scale DeepSpeed/Megatron training. It provides lazy, asynchronous checkpointing backed by CUDA and `io_uring`, allowing you to overlap checkpoint I/O with forward/backward passes and reduce checkpoint overheads at scale.

For a detailed description of the design, implementation, and evaluation against state-of-the-art checkpointing engines, see our HPDC’24 paper:

> Avinash Maurya, Robert Underwood, M. Mustafa Rafique, Franck Cappello, and Bogdan Nicolae.  
> **“DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models”**.  
> HPDC’24: The 33rd International Symposium on High-Performance Parallel and Distributed Computing (Pisa, Italy, 2024).  

---

## 1. Features Overview

- Lazy, asynchronous checkpointing for large language model training
- GPU-aware checkpoint engine (CUDA) with host/device tiers
- `io_uring`-based file I/O path for low-overhead persistence
- C++ core library with a thin Python binding (nanobind)
- Integrates with DeepSpeed’s async checkpointing API
- Designed for HPC/AI environments (multi-GPU, multi-node)

--- 

## 2. Installation and Tests

### 2.1. Prerequisites
- Linux (tested on recent x86_64 distributions)
- C++17 compiler (GCC 11+ recommended)
- CUDA toolkit (matching your cluster environment)
- `liburing` development headers
- CMake ≥ 3.15
- Python ≥ 3.8
- (For Python bindings)  
  - PyTorch  
  - fasteners
  - `nanobind` (pulled in automatically by `install.sh`)


### 2.2. Clone and Build
```bash
git clone https://github.com/DataStates/datastates-llm.git
cd datastates-llm/

# Activate your target Python/conda environment first.
# By default, install.sh will:
#   - detect the active Python env
#   - install into its site-packages
#   - build C++ core + Python bindings
./install.sh
```

By default, `install.sh` builds the C++ core library and the Python bindings and installs into the active environment's `site-packages`.

You can also control the install prefix and whether Python bindings are built:
```bash
# 1st arg: install prefix (optional)
# 2nd arg: build Python bindings? [on/off/yes/no/1/0]

# Example: install into a custom prefix WITH Python bindings
./install.sh /path/to/prefix on

# Example: install core library only, no Python bindings,
# into the active Python environment’s site-packages
./install.sh "" off
```

### 2.3. Tests
C++ core engine test
```bash
# Run the test binary directly
./build/tests/test_core_engine
# Or run through ctests
cd build/
ctest 
```

Python tests
```bash
python tests/python/test_base_state_provider.py # Without DeepSpeed
python tests/python/test_llm_ckpt_state_engine.py # With DeepSpeed
```

---

## 3. Using DataStates-LLM with DeepSpeed

DeepSpeed provides an official tutorial on enabling DataStates-based asynchronous checkpointing through a single JSON entry in the config file: [Official DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/datastates-async-checkpointing/)

That tutorial covers:
- Configuring DeepSpeed to use DataStates-LLM as the asynchronous checkpoint backend
- Relevant DeepSpeed configuration options
- Example training scripts integrating DataStates-LLM

---

## 4. Contributions and Issues
We welcome feedback, bug reports, and contributions.
- File issues and feature requests via the GitHub Issue tracker.
- Contributions in the form of bug fixes, portability improvements, and integration with additional frameworks are particularly appreciated.
