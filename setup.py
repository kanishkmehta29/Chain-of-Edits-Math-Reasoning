from setuptools import setup, find_packages

setup(
    name="coe-math-reasoning",
    version="0.1.0",
    description="Chain-of-Edits Math Reasoning Environment",
    author="Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "sympy>=1.12",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    },
    python_requires=">=3.8",
)