# SSSUMO: Real-Time Semi-Supervised Submovement Decomposition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dolphin-in-a-coma/sssumo/blob/main/notebooks/Inference.ipynb)

Follow the Colab link to check SSSUMO inference on both synthetic and organic data.
**NOTE: Train and Analysis notebooks need updates to run smoothly.**

This repository accompanies the article "SSSUMO: Real-Time Semi-Supervised Submovement Decomposition". It is a work in progress and is going to be refactored.


## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/dolphin-in-a-coma/sssumo.git
```

Or clone and install locally:

```bash
git clone https://github.com/dolphin-in-a-coma/sssumo.git
cd sssumo
pip install .
```

## Citation

If you find the work helpful for your research, please cite it as:

```
@misc{rudakov2025sssumorealtimesemisupervisedsubmovement,
      title={SSSUMO: Real-Time Semi-Supervised Submovement Decomposition}, 
      author={Evgenii Rudakov and Jonathan Shock and Otto Lappi and Benjamin Ultan Cowley},
      year={2025},
      eprint={2507.08028},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2507.08028}, 
}
```

## Project Structure

- **sssumo/**: Contains the core implementation
  - `models.py`: Models for submovement detection and reconstruction
  - `data.py`: Dataset implementations for synthetic and organic movement data
  - `utils.py`: Utility functions for data processing and evaluation
  - `dataset_reader.py`: Functions for creating STV data from the original datasets.
  - `alternative_detectors.py`: Contains code for the Peak Detector and the preliminary version of Scattershot
  - `movement_decompose.py`: The final Scattershot version used


- **notebooks/**: Contains inference, evaluation, and training notebooks.
  - `Inference.ipynb`: Notebook showcasing inference on synthetic and organic data.
  - `Train.ipynb`: Notebook for training the models. Designed for Google Colab.
  - `Analysis - organic and synth.ipynb`: Notebook used to analyse results, evaluate the model and baselines, and generate figures.

- **configs/**: YAML configuration files for model architecture, training parameters, dataset options, and ablation studies.

- **checkpoints/**: Contains both pre-trained and fine-tuned model checkpoints. Only the fine-tuned checkpoint released under CC BY 4.0 is included here; the checkpoint trained on the hand-writing data (research-only licence) will be linked later.

- **data/**: Tangential velocity data files for organic human motion datasets.
