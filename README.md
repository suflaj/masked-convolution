# Masked Convolution

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A PyTorch implementation of a thin wrapper for masked convolutions.


## What are masked convolutions?

Similarly to [partial convolutions](https://github.com/NVIDIA/partialconv), masked convolutions mask a part of the kernel, essentially ignoring data at specific locations. For an example, consider

```python
a = [1, 2, 3, 4, 5]
```

assuming we have a convolution kernel

```python
kernel = [1, 1, 1]
```

convolving over `a` would give us

```python
a_conv = [6, 9, 12]
```

However, if we were to mask the convolution kernel with a mask

```python
mask = [1, 0, 1]
```

**masked convolving** over `a` would return

```python
a_masked_conv = [4, 6, 8]
```

One use of masked convolutions is emulating skip-grams.


## Installation

First, make sure you have PyTorch installed. This was tested on **Python 3.8** and **PyTorch 1.7.1**. Further testing is needed to determine whether it works on a different setup - chances are it does. The recommended way to install this is through PyPi by running:

```bash
pip install masked-convolution
```

Other than that, you can clone this repository, and in its root directory (where `setup.py` is located) run

```bash
pip install .
```

## Benchmarks

Every build, automatic benchmarks are run in order to determine how much overhead the implementation brings. The ordinary convolutions are used as a baseline, while the the performance of masked convolutions is described as a percentage of throughput of their respective baselines.

Keep in mind that these benchmarks are in no way professional, they only serve to give users a general idea. Their results greatly differ, so they should be taken with a grain of salt.

- Masked Convolution 1D: **85.29 %** Convolution 1D throughput
- Masked Convolution 2D: **85.64 %** Convolution 2D throughput
- Masked Convolution 3D: **97.79 %** Convolution 3D throughput

