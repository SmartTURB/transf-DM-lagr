# transf-DM-lagr

This repository provides a transformer-based variant of the diffusion model for Lagrangian turbulence, as presented in our paper:

**"Deterministic Diffusion Models for Lagrangian Turbulence: Architectural Robustness and Extreme Event Signatures"**  
*(Link will be added here once the paper is published)*

It extends the [SmartTURB/diffusion-lagr](https://github.com/SmartTURB/diffusion-lagr) repository by replacing the U-Net backbone with a Diffusion Transformer (DiT)-based architecture for noise prediction, following the formulation introduced in [Peebles et al., 2023](https://arxiv.org/abs/2303.11305).

## Usage

The training and sampling pipelines remain unchanged from the original repository. The only modification lies in the `--model` flag:

```bash
MODEL_FLAGS="--model DiT-M/8 --image_size 2000 --in_channels 3 --num_classes 0"
```

The model name `DiT-M/8` refers to a predefined architecture with settings such as depth, hidden size, patch size, and number of attention heads. Other variants (e.g., `DiT-S/125`, `DiT-B/8`) are defined in [`transformer_diffusion/models.py`](transformer_diffusion/models.py), and can be modified or extended as needed.

All other training, data, and diffusion settings remain exactly the same as in the original U-Net-based repository; for full details on data format, preprocessing, training flags, and sampling scripts, please refer to [SmartTURB/diffusion-lagr](https://github.com/SmartTURB/diffusion-lagr).

## Citation

If you use this code in your research, please cite:

> Tianyi Li, Flavio Tuteri, Michele Buzzicotti, Fabio Bonaccorso, Luca Biferale.  
> *Deterministic Diffusion Models for Lagrangian Turbulence: Architectural Robustness and Extreme Event Signatures*, 2024.  
> [link to be added]
