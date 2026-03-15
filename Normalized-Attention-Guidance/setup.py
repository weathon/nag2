from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="nag",
    version="0.0.0",
    description="Normalized Attention Guidance for Diffusion Models",
    author="ChenDarYen",
    packages=find_packages(include=["nag", "nag.*"]),
    install_requires=requirements,
    python_requires=">=3.10",
)