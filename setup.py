from setuptools import setup, find_packages

setup(
    name="mmwave_beam_management",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.10",
)
