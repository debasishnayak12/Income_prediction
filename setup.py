from setuptools import find_packages,setup
from typing import List

setup(
    name="Income_prediction",
    version="0.0.1",
    author="Debasish Nayak",
    author_email="nayakdeabsish7205@gmail.com",
    install_requires=['scikit-learn',"pandas","numpy"],
    packages=find_packages()
)