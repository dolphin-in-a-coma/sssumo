# SSSUMO: Real-Time Semi-Supervised Submovement Decomposition

The repository is not in its final form yet; it will be updated iteratively.

## Citation

If you find the work helpful for your research, please cite it as

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

- **src/**: Contains the core implementation
  - `models.py`: Models for submovement detection and reconstruction
  - `data.py`: Dataset implementations for synthetic and organic movement data
  - `utils.py`: Utility functions for data processing and evaluation
  - `dataset_reader.py`: Functions for creating STV data from the original datasets.
  - `alternative_detectors.py`: Contains code for the Peak Detector and preliminary version of Scattershot
  - `movement_decompose.py`: The final Scattershot version used

  - `Train.ipynb`: Jupyter notebook for training the models. It was run in a Google Colab environment.
  - `Analysis - organic and synth.ipynb`: Notebook used to analyse results, run inference for the model and baselines, and generate the figures.

- **configs/**: YAML configuration files for different model variations - pre-training, full fine-tuning, leave-one-dataset-out, ablations
  - Configuration for model architecture, training parameters, and dataset options

- **checkpoints/**: Contains both pre-trained and fine-tuned model checkpoints. Only the fine-tuned checkpoint released under CC BY 4.0 is included here; the checkpoint trained on the hand-writing data (research-only licence) will be linked later.”
