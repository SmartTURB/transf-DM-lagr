from setuptools import setup

setup(
    name="transformer-diffusion",
    py_modules=[
        "guided_diffusion",
        "continuous_diffusion",
        "palette_diffusion",
        "transformer_diffusion"
    ],
    install_requires=[
        "blobfile>=1.0.5",
        "tqdm",
        "h5py",
        "timm"
    ],
)
