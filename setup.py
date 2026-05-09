import os
from pathlib import Path

import setuptools

package_name = "tinygrad_lightning"

here = Path(__file__).parent

main_ns = {}
with (here / package_name / "version.py").open() as ver_file:
    exec(ver_file.read(), main_ns)

with (here / "requirements.txt").open() as h:
    requirements = [r.strip() for r in h.readlines() if r.strip() and not r.startswith("#")]

with (here / "README.md").open() as fh:
    long_description = fh.read()


setuptools.setup(
    name=package_name,
    version=main_ns["__version__"],
    description="high level interface for tinygrad (pytorch-lightning style)",
    author="Malte Koch",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["tinygrad", "lightning", "training", "deep-learning"],
    author_email="malte-koch@gmx.net",
    maintainer="Malte Koch",
    maintainer_email="malte-koch@gmx.net",
    url="https://github.com/baudcode/tinygrad-lightning",
    python_requires=">=3.11",
    packages=setuptools.find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "mlflow": ["mlflow>=2.0"],
        "tensorboard": ["tensorboardX"],
        "test": [
            "pytest>=7.0",
            "lightning>=2.0",
            "torch>=2.0",
            "mlflow>=2.0",
            "tensorboardX",
            "safetensors",
        ],
    },
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
