# -*- coding: utf-8 -*-

from setuptools import setup


setup(
    name="HelioSat",
    packages=[
        "heliosat",
        "heliosat.data"
        ],
    package_data={"heliosat": ["json/*.json"]},
    version="0.2.0",
    author="Andreas J. Weiss",
    author_email="andreas.weiss@oeaw.ac.at",
    keywords=["astrophysics", "heliophysics", "space weather"],
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ajefweiss/HelioSat",
    install_requires=[
        "bs4",
        "netcdf4",
        "numba",
        "numpy",
        "requests",
        "requests-ftp",
        "scipy",
        "spacepy",
        "spiceypy"
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)