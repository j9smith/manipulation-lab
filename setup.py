from setuptools import setup, find_packages
import toml
import os

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXT_TOML_PATH = os.path.join(EXTENSION_PATH, "extension.toml")
EXT_DATA = toml.load(EXT_TOML_PATH)["package"] if os.path.exists(EXT_TOML_PATH) else {}

setup(
    name="manipulation_lab",
    url=EXT_DATA.get("repository", "https://github.com/j9smith/manipulation-lab"),
    version=EXT_DATA.get("version", "0.1"),
    description=EXT_DATA.get("description", "Benchmarking framework for manipulation tasks within Isaac Lab."),
    author=EXT_DATA.get("author", "Joel Smith"),
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    include_package_data=True,
    install_requires=[
        "psutil"
    ],
    python_requires=">=3.10",
    zip_safe=False
)