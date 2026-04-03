# CuPy Installation Diagnosis on Aziz

## Problem
CuPy failed to install via both pip and conda on the A100 node.

### pip failure (cupy-cuda12x)
- Tries to build numpy from source as a dependency
- System GCC is 4.8.5 (Red Hat 7), but numpy 2.4.4 requires GCC >= 9.3
- Root cause: **ancient system compiler on Aziz compute nodes**

### conda failure
- glibc version conflict: cupy requires `__glibc >=2.28`, system has older
- Python 3.12 pin conflicts with available numpy-base packages
- Root cause: **CentOS 7 / RHEL 7 with glibc 2.17** — too old for modern cupy conda packages

## Solution Options
1. **Pin older cupy version**: `pip install cupy-cuda11x==12.3.0` (may have pre-built wheel for this system)
2. **Use conda with older channel**: `conda install -c conda-forge cupy=12.3 numpy=1.26.4`
3. **Create new conda env with compatible versions**: Build a fresh env with Python 3.11 + compatible glibc
4. **Use module system**: Check if Aziz has cupy as a module (`module avail cupy`)
5. **Compile from source with newer GCC**: Load a newer GCC module first
