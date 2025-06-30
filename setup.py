#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bapred",
    version="1.0.0",
    author="BAPred Team",
    author_email="",
    description="Protein-ligand Binding Affinity Prediction using Graph Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eightmm/BAPred",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "bapred": ["weight/*.pth"],
    },
    entry_points={
        "console_scripts": [
            "bapred=bapred.inference:main",
        ],
    },
    keywords="protein ligand binding affinity prediction machine learning gnn graph neural network",
) 