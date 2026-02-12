from setuptools import setup, find_packages

setup(
    name="VisionGauge",
    version="0.0.1",
    description="A computer vision model to detect and read u-tube manometers.",
    long_description=open("README.txt", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.12,<3.15",
    packages=find_packages(),
    install_requires=[
        "torch>=2.9.0",
        "torchvision>=0.24.0",
        "ultralytics>=8.4.14",
        "huggingface_hub>=1.4.0",
        "albumentations>=2.0.8",
        "tqdm>=4.67.3",
        "opencv-python>=4.13.0"
    ],
    author="Clayton Silva dos Santos, Marcos Noboru Arima",
    author_email="santoscsdos@gmail.com, mnoboru@n-thermo.com.br",
    maintainer="Clayton Silva dos Santos",
    maintainer_email="santoscsdos@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
