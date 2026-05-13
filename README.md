<h1 align="center">
  VMatcher: State-Space Semi-Dense Local Feature Matching
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2507.23371"><img src="https://img.shields.io/badge/arXiv-2507.23371-b31b1b.svg" alt="arXiv"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<h3 align="center"><a href="https://ayoussf.github.io/">Ali Youssef</a></h3>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#configurations">Configurations</a> •
  <a href="#training">Training</a> •
  <a href="#evaluations">Evaluations</a> •
  <a href="#checkpoints">Checkpoints</a> •
  <a href="#hardware-specifications">Hardware</a> •
  <a href="#reporting-issues">Issues</a> •
  <a href="#acknowledgements">Acknowledgements</a> •
  <a href="#citation">Citation</a>
</p>

## Overview

VMatcher is a hybrid Mamba‑Transformer network for semi‑dense local feature matching. It combines Mamba's efficient Selective Scan Mechanism with Transformer's attention mechanism, balancing accuracy and computational efficiency.

## Installation

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/ayoussf/VMatcher.git

# Install the package
python setup.py install  # or develop for development mode

# Install dependencies
pip install -r requirements.txt

# Install Mamba
cd third_party/mamba/
pip install .
cd ../..
```

> [!NOTE]
> If Mamba installation fails, try: `pip install . --no-build-isolation`

> [!WARNING]
> Triton>2.0.0 causes slower runtime at lower resolutions. Install Triton==2.0.0 for optimal performance.

## Configurations

The main configuration file is located at `configs/conf_blocks.py`, which controls both training and testing parameters. Specialised configurations can be found in:
- `configs/train.py` - Training specific settings
- `configs/test.py` - Testing specific settings

To view all available configurations:
```bash
python engine.py -h
```

## Training

### Quick Start
```bash
# Run the training script
sh scripts/train/train.sh
```

To train the Tiny model variant, either:
- Change `num_layers` to 14 in `configs/conf_blocks.py`, or
- Run: `python engine.py --task=train --config.train.mamba-config.num-layers=14`

> [!NOTE]
> For Base model, the gradient accumulation was set to 8, while for Tiny model, it was set to 32. Adjust 'gradient_accumulation_steps' in `configs/train.py` if needed.

> [!NOTE]
> Currently training only supports a batch size of 1, to accomodate a larger batch size, modify the [following lines](https://github.com/ayoussf/VMatcher/blob/46cba063cb28908750e8b56b0d4a3a6ce72c56ce/VMatch/src/VMatcher/VMatcher.py#L108C9-L155C143) in `VMatcher.py` to loop over batch samples, followed by stacking the outputs post the loop.

### Training Data

**MegaDepth Dataset (199GB)**
- [Download from the MegaDepth website](https://www.cs.cornell.edu/projects/megadepth/)
- Or run: `sh data/megadepth/download_megadepth.sh`

After downloading, process the images:
```bash
python data/megadepth/process_megadepth_images.py --root_directory /path_to_megadepth/phoenix/S6/zl548/MegaDepth_v1
```

**Training Indices**
- [Download from Google Drive](https://drive.google.com/file/d/1O3691mkd3hwWDRJDwM3mgl9kLxmPURoe/view?usp=drive_link)
- Or run: `gdown --id 1O3691mkd3hwWDRJDwM3mgl9kLxmPURoe`

**Directory Setup**
```bash
cd data/megadepth
mkdir train index
ln -sv /path_to_megadepth/phoenix path_to_VMatcher/data/megadepth/train
ln -sv /path_to_megadepth_indices path_to_VMatcher/data/megadepth/index
```

## Evaluations

Multiple evaluation scripts are available in the `scripts/test` directory:
- `scripts/test/*.sh` - Scripts for baseline model evaluation
- `scripts/test/opt/*.sh` - Scripts for optimised variant evaluation
- `scripts/test/tune/*.sh` - Scripts for evaluation with multiple RANSAC thresholds

### Running Evaluations
```bash
# Example: Testing on MegaDepth
sh scripts/test/test_megadepth.sh
```

### Test Datasets

| Dataset | Download Link | `gdown` Command |
|---------|---------------|-----------------|
| **MegaDepth1500** | [Google Drive](https://drive.google.com/file/d/1K5hpS4xg6OLMCx0tLUXG8wqokK80fPnb/view?usp=sharing) | `gdown --id 1K5hpS4xg6OLMCx0tLUXG8wqokK80fPnb` |
| **ScanNet1500** | [Google Drive](https://drive.google.com/file/d/1Ryv2YSC277ec8Ki6e34vfbqIMb5BKB-r/view?usp=sharing) | `gdown --id 1Ryv2YSC277ec8Ki6e34vfbqIMb5BKB` |
| **HPatches** | [Google Drive](https://drive.google.com/file/d/1IAUC44oR0ggUPONLy_stLxRhpAMKZm2b/view?usp=sharing) | `gdown --id 1IAUC44oR0ggUPONLy_stLxRhpAMKZm2b` |
   
## Checkpoints

Pre-trained model checkpoints are available for download:

| Model Variant | Download Link | `gdown` Command |
|--------------|---------------|-----------------|
| **VMatcher-B** (Base) | [Google Drive](https://drive.google.com/file/d/1ENP_DhAihiv5WJrRWoAXmOHUdrddJaxU/view?usp=sharing) | `gdown --id 1ENP_DhAihiv5WJrRWoAXmOHUdrddJaxU` |
| **VMatcher-T** (Tiny) | [Google Drive](https://drive.google.com/file/d/1TRiKdPhGjpQ1F2_O9KJ8CyaPffWfNf61/view?usp=sharing) | `gdown --id 1TRiKdPhGjpQ1F2_O9KJ8CyaPffWfNf61` |

## Hardware Specifications

The training and evaluation environment utilised:
- **GPU**: NVIDIA GeForce RTX 3090Ti
- **CUDA**: 11.8 (V11.8.89)
- **Python**: 3.9.19
- **PyTorch**: 2.2.2+cu118
- **Triton**: 2.0.0

## Reporting Issues

If you encounter any bugs or issues, please feel free to [open an issue](https://github.com/ayoussf/VMatcher/issues) or submit a pull request. Your contributions are greatly appreciated!

## Acknowledgements

Special thanks to the authors of [ELoFTR](https://github.com/zju3dv/ELoFTR), as a significant portion of their codebase was utilised in this project.

## Citation
If you find this work useful, please consider citing the paper:

```bibtex
@misc{youssef2025vmatcherstatespacesemidenselocal,
      title={VMatcher: State-Space Semi-Dense Local Feature Matching}, 
      author={Ali Youssef},
      year={2025},
      eprint={2507.23371},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.23371}, 
}
```