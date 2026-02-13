from setuptools import setup, find_packages

setup(
    name="visiongauge",
    version="0.0.1",
    description="A computer vision model to detect and read u-tube manometers.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9,<3.15",
    packages=find_packages(where="."),  # procura na raiz
    install_requires=[
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "ultralytics>=8.0.0",
    "huggingface_hub>=0.20.0",
    "datasets>=2.14.0",
    "albumentations>=1.3.0",
    "tqdm>=4.65.0",
    "opencv-python>=4.8.0",
    "numpy>=1.24.0,<2.0",
    "matplotlib>=3.7.0"],
    author="Clayton Silva dos Santos, Marcos Noboru Arima",
    author_email="santoscsdos@gmail.com, mnoboru@n-thermo.com.br",
    maintainer="Clayton Silva dos Santos",
    maintainer_email="santoscsdos@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
