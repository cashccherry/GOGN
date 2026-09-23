from setuptools import find_packages, setup

setup(
    name="gogn",
    version="0.1.0",
    description="Gradient-only Gauss–Newton acoustic FWI paper experiments",
    packages=find_packages(include=["gogn", "gogn.*"]),
    package_data={"gogn.objectives": ["FWI/*.png", "FWI/*/*.npy"]},
    python_requires=">=3.11",
    install_requires=[
        "torch==2.6.0", "torchvision==0.21.0", "deepwave==0.0.20",
        "numpy==2.3.0", "scipy==1.16.0", "matplotlib==3.10.3", "psutil==7.1.0",
    ],
    extras_require={"test": ["pytest", "sympy==1.14.0"]},
)
