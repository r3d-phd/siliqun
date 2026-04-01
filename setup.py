"""SiliQun — Silicon Qubits Simulator."""
from setuptools import setup, find_packages

setup(
    name="siliqun",
    version="1.0.0",
    description="Tensor-network simulator for silicon spin qubit systems",
    author="Raad Al-Shehri",
    author_email="ralshehri0468@stu.kau.edu.sa",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.9",
        "gymnasium>=0.26",
    ],
    extras_require={
        "gpu": ["jax[cuda]>=0.4"],
        "hpc": ["quimb>=1.6", "cotengra>=0.5"],
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.11",
    ],
)
