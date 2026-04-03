# CuPy Installation Progress

## What Worked
- CUDA runtime libs installed via conda: `libcudart.so.12`, `libcublas.so.12` in `$CONDA_PREFIX/lib`
- CuPy 14.0.1 installed via pip (cupy-cuda12x)
- LD_LIBRARY_PATH set correctly

## What Failed
CuPy 14.0.1 requires `cuda-python` package (the `cuda` module):
```
ModuleNotFoundError: No module named 'cuda'
```
This is the `cuda-python` package from NVIDIA that provides Python bindings.

## Fix
Need to install `cuda-python` package:
```
pip install cuda-python
```
or
```
conda install -c nvidia cuda-python
```
