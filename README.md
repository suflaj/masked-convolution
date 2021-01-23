# Masked Convolution

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
