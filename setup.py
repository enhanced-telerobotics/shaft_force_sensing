"""Setup configuration for shaft_force_sensing package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="shaft_force_sensing",
    version="0.1.0",
    author="Erie Lab",
    description="Transformer-based force prediction for shaft sensors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enhanced-telerobotics/shaft_force_sensing",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "torch>=1.9.0",
        "pytorch-lightning>=1.5.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
    ],
)
