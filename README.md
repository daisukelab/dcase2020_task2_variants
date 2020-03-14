# DCASE2020 Challenge Task 2 baseline variants
This is a repository to share variants of baseline system for **DCASE 2020 Challenge Task 2 "Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring"**. 

http://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds

## Description

Baseline system implements Autoencoder with Keras, and provides reproducible training & testing codes.

This repository expands the baseline by:

- Making it easier to duplicate solution, and try your own ideas on it with less effort.
- Providing PyTorch version of the baseline.
- VAE version is included.
- TBD --> Hoping to have time to show some of my ideas.

## Using examples

Prepare `dev_data` folder as described in the [original baseline USAGE](https://github.com/y-kawagu/dcase2020_task2_baseline#usage).

    ./dcase2020_task2_variants (this folder)
        /dev_data              (prepare this)

### PyTorch version baseline

In folder `1pytorch/`, there're both training and test code, accompanied with `config.yaml` and `pytorch_model.py`.

You can run a Jupyter notebook `00-train-with-visual.ipynb` to train your models, this will also show some visualizations.

And run `01_test.py` to evaluate your models as follows, make sure you run this under folder `1pytorch/`.

```sh
your/1pytorch$ python 01_test.py -d
```

* Stable reproduction of good performance is hard so far.

### VAE version

In folder `2vae_pytorch/`, you can find VAE solution.

It was hard to use VAE with basic configuration, so it has special setting with loss calculation.

### Original baseline

Original files are moved to a folder `0original/`, just follow the same as described in the [USAGE](https://github.com/y-kawagu/dcase2020_task2_baseline#usage).

## Links

- [Original baseline repository - https://github.com/y-kawagu/dcase2020_task2_baseline](https://github.com/y-kawagu/dcase2020_task2_baseline)
