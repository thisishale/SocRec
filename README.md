This repo contains the implementation of our paper:

## Socially-Informed Reconstruction for Pedestrian Trajectory Forecasting

Haleh Damirchi, Ali Etemad, Michael Greenspan

**WACV 2025**  
[[paper](https://arxiv.org/pdf/2412.04673)]

<p align="center">
  <img src="/image/method.jpg" width="100%">
</p>


### Environment

* **Tested OS:** Windows 10

* python 3.9.6
Libraries are included in the requirements.txt.

## Overview

Pedestrian trajectory prediction remains a challenge for autonomous systems, particularly due to the intricate dynamics of social interactions. Accurate forecasting requires a comprehensive understanding not only of each pedestrian's previous trajectory but also of their interaction with the surrounding environment, an important part of which are other pedestrians moving dynamically in the scene. To learn effective socially-informed representations, we propose 
a model that uses a reconstructor alongside a conditional variational autoencoder-based trajectory forecasting module. This module generates pseudo-trajectories, which we use as augmentations throughout the training process. To further guide the model towards social awareness, we propose a novel social loss that aids in forecasting of more stable trajectories. We validate our approach through extensive experiments, demonstrating strong performances in comparison to state-of-the-art methods on the ETH/UCY and SDD benchmarks.


### Datasets
Dataset files are in data folder. please adjust ``dataset_folder'' argument.

You can use run_all.sh to train the models.


# Citation
If you find our work useful in your research, please cite our paper:
```bibtex
update
```