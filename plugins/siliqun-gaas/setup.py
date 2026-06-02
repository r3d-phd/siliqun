from setuptools import setup, find_packages

setup(
    name="siliqun-gaas",
    version="1.0.0",
    description="SiliQun plugin for gate-all-around (GAA) silicon spin qubits",
    author="Raad Alshehri",
    author_email="r.alshehri@stu.kau.edu.sa",
    license="MIT",
    packages=find_packages(),
    package_data={"siliqun_gaas": ["gaa_calibration.json"]},
    python_requires=">=3.9",
    install_requires=["siliqun>=1.0.0", "numpy>=1.24"],
    entry_points={
        "siliqun.plugins": [
            "gaa_nominal = siliqun_gaas:GAAProfile",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
