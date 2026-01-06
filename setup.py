from setuptools import setup, find_packages

setup(
    name="mlProject",
    version="0.0.1",
    author="Mansi Tyagi",
    author_email="mansityagi0606",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
