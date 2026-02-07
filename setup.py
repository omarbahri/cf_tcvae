"""
CF-TCVAE: Robust Counterfactual Explanations via Class-Conditional β-TCVAE
Setup configuration for package installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cf-tcvae",
    version="1.0.0",
    author="Anonymous Authors",
    author_email="",
    description="Robust Counterfactual Explanations via Class-Conditional β-TCVAE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/cf_tcvae",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "cf_tcvae": [
            "datasets/data/MNIST/raw/*",
            "datasets/data/adult/*",
            "classifiers/checkpoints/*",
        ],
    },
)
