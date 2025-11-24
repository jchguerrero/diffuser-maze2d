# Diffuser for Maze2D Navigation

A diffusion model implementation for trajectory planning in 2D maze environments using the D4RL maze2d datasets.

## Overview

This project trains a diffusion probabilistic model to generate feasible trajectories for navigation in maze environments. The model learns from offline trajectory data and can generate collision-free paths from start to goal positions.

## Features

- ✅ Diffusion model training for maze2d
- ✅ Trajectory generation and visualization
- ✅ Jupyter notebook and Google Colab support
- ✅ GPU-accelerated training
- ✅ Checkpoint saving and loading

## Installation

### Requirements

- Python 3.8
- CUDA-capable GPU
- MuJoCo 2.1.0

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/diffuser-maze2d.git
cd diffuser-maze2d

# Create conda environment
conda env create -f environment.yml
conda activate diffuser_maze2d

# Install the package
pip install -e .
```

## Quick Start

### Training

**Jupyter Notebook:**

```bash
jupyter notebook train_diffuser.ipynb
```

### Configuration

Key training parameters (in `config/maze2d.py`):

- `horizon`: Trajectory length (default: 128)
- `n_train_steps`: Total training steps (default: 50000)
- `batch_size`: Batch size (default: 16-32)
- `n_diffusion_steps`: Diffusion timesteps (default: 64)

## Datasets

Uses D4RL maze2d environments:

- `maze2d-umaze-v1`: Small U-shaped maze
- `maze2d-medium-v1`: Medium-sized maze
- `maze2d-large-v1`: Large complex maze

## Citation

Based on the Diffuser framework:

```
@inproceedings{janner2022diffuser,
  title={Planning with Diffusion for Flexible Behavior Synthesis},
  author={Janner, Michael and Du, Yilun and Tenenbaum, Joshua and Levine, Sergey},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```

## License

MIT License

## Acknowledgments

- Original Diffuser implementation by [jannerm](https://github.com/jannerm/diffuser)
- D4RL dataset by [Farama Foundation](https://github.com/Farama-Foundation/d4rl)
- Course: ENME 618 - Methods in Uncertainty Quantification and Machine Learning for Scientific Engineering
  Applications, University of Calgary
