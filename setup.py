# -*- coding: utf-8 -*-

from setuptools import setup


setup(
    name="HelioSat",
    packages=[
        "heliosat"
    ],
    package_data={"heliosat": ["spacecraft/*.json"]},
    version="0.8.0",
    author="Andreas J. Weiss",
    author_email="andreas.weiss@oeaw.ac.at",
    keywords=["astrophysics", "solar physics", "space weather"],
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ajefweiss/HelioSat",
    install_requires=[
        "beautifulsoup4>=4.11.1",
        "cdflib>=0.3.19",
        "netcdf4>=1.6.2",
        "numba>=0.56.4",
        "numpy>=1.23.5",
        "requests>=2.28.1",
        "requests-ftp>=0.3.1",
        "spiceypy>=5.1.2"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
