# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="robosuite",
    packages=[package for package in find_packages() if package.startswith("robosuite")],
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.52.0,<=0.53.1",
        "scipy>=1.2.3",
        "free-mujoco-py==2.1.6",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
)
