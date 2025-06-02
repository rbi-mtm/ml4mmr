import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()

setup(
    name="SPaiNN",
    version="1.0.0",
    author="Sascha Mausenberger, Carolin Müller, Julia Westermayr, Michael Gastegger, Philipp Marquetand",
    url="https://github.com/smausenberger/SPaiNN",
    packages=find_packages("src"),
    scripts=[
        "src/scripts/spainn-db",
        "src/scripts/aselen",
        "src/scripts/spainn-train",
    ],
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "ase>=3.21",
        "schnetpack>=2.0.0",
    ],
    include_package_data=True,
    license="MIT",
    description="SPaiNN - Exited State Dynamics with Machine Learning",
    long_description="""
        A SchNetPack/SHARC interface for machine-learning accelerated excited state simulations.
        Based on SchNarc with major performance and usability improvements.""",
)
