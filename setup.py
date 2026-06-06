from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pits_mras",
    version="0.5.4",
    author="PITS-MRAS Contributors",
    author_email="your.email@example.com",
    description="Physics-Informed Time-Series Model-Reference Adaptive Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PITS-MRAS",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/PITS-MRAS/issues",
        "Documentation": "https://github.com/yourusername/PITS-MRAS/tree/main/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "logging": [
            "tensorboard",
            "wandb",
        ],
    },
)
