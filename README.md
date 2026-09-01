# SSSUMO: Real-Time Semi-Supervised Submovement Decomposition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dolphin-in-a-coma/sssumo/blob/main/notebooks/Inference.ipynb)

Follow the Colab link to check SSSUMO inference on both synthetic and organic data.
**NOTE: the Analysis notebook still needs updates to run smoothly.**

This repository accompanies the article "SSSUMO: Real-Time Semi-Supervised Submovement Decomposition". It is a work in progress and is going to be refactored.

**New to the code?** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) explains how the
method and the code fit together — the two models, the two training stages, the
configuration system, and where everything lives.


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

## Quickstart

Train from the command line, without the notebook:

```bash
python scripts/train.py --config configs/config-0423-ModGaussian_ampl.yaml \
    --root-dir <root> --experiment-name my-pretrain
```

`--root-dir` is a working directory that must contain `data/`, `weights/` and
`logs/` — it is not the repository root. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#running-it) for the flags that matter,
and [`scripts/colab/README.md`](scripts/colab/README.md) to run the same thing on a
Colab GPU.

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
  - `training.py`: The training loop, `train()`, shared by the notebook and the CLI
  - `utils.py`: `Config`, metrics, evaluation, and bootstrap utilities
  - `dataset_reader.py`: Functions for creating STV data from the original datasets.
  - `alternative_detectors.py`: Contains code for the Peak Detector and the preliminary version of Scattershot
  - `movement_decompose.py`: The final Scattershot version used


- **notebooks/**: Contains inference, evaluation, and training notebooks.
  - `Inference.ipynb`: Notebook showcasing inference on synthetic and organic data.
  - `Train.ipynb`: Notebook for training the models. Designed for Google Colab.
  - `Analysis - organic and synth.ipynb`: Notebook used to analyse results, evaluate the model and baselines, and generate figures.

- **configs/**: YAML configuration files for model architecture, training parameters, dataset options, and ablation studies.

- **checkpoints/**: Contains both pre-trained and fine-tuned model checkpoints. Only the fine-tuned checkpoint released under CC BY 4.0 is included here; the checkpoint trained on the hand-writing data (research-only licence) will be linked later.

- **data/**: Tangential velocity data files for organic human motion datasets. Gitignored (~1.9 GB); fetch it from the public archive — there is a `curl` one-liner in `scripts/colab/README.md`.

- **scripts/**: Command-line tooling.
  - `train.py`: CLI entry point for training.
  - `colab/`: Driving a Colab GPU session from the terminal.
  - `analysis/`: Offline recomputation of evaluation intervals from per-trial dumps (no GPU).

- **docs/**: Method and study documentation.
  - `ARCHITECTURE.md`: How the method and code work — start here.
  - `VALIDATION.md`: The checkpoint reproduction study and the interval methodology.
  - `IMPROVEMENTS.md`, `RUN_INVENTORY.md`: Known issues worth fixing, and the runs behind the study.
