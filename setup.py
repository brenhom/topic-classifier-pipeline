#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="topic_classifier",
    version="1.0",
    description="Packages for building topic classifiers and testing them against datasets",
    author="Brendan & Jenn",
    author_email="brendan.homnick@gmail.com",
    install_requires=["scikit-learn"],
    extras_require={"test": ["black", "flake8", "isort", "pylint", "pytest"]},
    packages=find_packages(exclude=["tests"]),
)
