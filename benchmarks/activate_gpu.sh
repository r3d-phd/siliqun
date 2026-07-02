#!/bin/bash
source /opt/share/anaconda3/2024.06/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/share/anaconda3/2023.09/etc/profile.d/conda.sh 2>/dev/null
conda activate quantum_drl_gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
