from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="Radar3D-Rehab", 
    version="0.1.0",
    packages=find_packages(),
    install_requires=required_packages,
    author="Kailu Guo",
    long_description="README.md",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
