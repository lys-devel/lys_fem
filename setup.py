from setuptools import setup
import sys
sys.path.append('./lys_fem')

setup(
    name="lys_fem",
    version="0.0.1",
    install_requires=["gmsh", "pyvistaqt"],
)
