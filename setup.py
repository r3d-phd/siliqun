"""SiliQun: GPU-Accelerated Tensor-Network Gymnasium Environment for Silicon Spin Qubit Control."""

from setuptools import setup, find_packages

setup(
    name="siliqun",
    version="2.0.0",
    author="Raad Al-Shehri",
    author_email="ralshehri0468@stu.kau.edu.sa",
    description=(
        "A GPU-accelerated tensor-network Gymnasium environment for "
        "deep reinforcement learning-based control of silicon spin qubits"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ralshehri/siliqun",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "gymnasium>=0.29",
    ],
    extras_require={
        "gpu": ["cupy-cuda12x>=13.0"],
        "dev": ["pytest>=7.0", "matplotlib>=3.7"],
        "rl": ["stable-baselines3>=2.0", "torch>=2.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="quantum computing, silicon spin qubits, reinforcement learning, "
             "tensor networks, GPU simulation, gymnasium",
)
