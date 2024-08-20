#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="topic_classifier",
    version="1.0",
    description="Packages for building topic classifiers and testing them against datasets",
    author="Brendan & Jenn",
    author_email="brendan.homnick@gmail.com",
    install_requires=["click", "datasets", "scikit-learn"],
    extras_require={
        "test": [
            "black",
            "flake8",
            "isort",
            "pylint",
            "pytest",
            "pytest-integration",
        ]
    },
    packages=find_packages(exclude=["tests"]),
    entry_points={"console_scripts": ["tcp=pipeline.main:evaluate"]},
)
