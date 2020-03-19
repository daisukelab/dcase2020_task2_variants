# DCASE2020 Challenge Task 2 baseline variants
This is a repository to share variants of baseline system for **DCASE 2020 Challenge Task 2 "Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring"**. 

http://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds

## 1. Description

Baseline system implements Autoencoder with Keras, and provides reproducible training & testing codes.

This repository expands the baseline by:

- Solutions by folders. `0original` holds everything about the original baseline, `1pytorch` is PyTorch version, and so on. Making it easier to duplicate, and try your own ideas on it.
- PyTorch version provided.
- VAE implementation (but result is not good) included.
- Convolutional Autoencoder implementation (but result is good) included.
- ... and would have some more in the future.

## 2. Using examples

1. Prepare `dev_data` folder as described in the [original baseline USAGE](https://github.com/y-kawagu/dcase2020_task2_baseline#usage).

    ./dcase2020_task2_variants (this folder)
        /dev_data              (prepare this)

2. Train models. `python 00_train.py -d` will train if solution folder has `00_train.py`, or run a notebook named `00-train-with-visual.ipynb`.

3. Test models. `python 01_test.py -d` will evaluate performance with test samples, and will summarize in `result/result.csv`.

4. Visualize predictions. `02-visualize-predictions.ipynb` will show how models reconstruct samples.

That's all basically, you might find something more in some solution folder.

## 3. 0original: Original baseline

Original files are moved to a folder `0original/`, just follow the same as described in the [USAGE](https://github.com/y-kawagu/dcase2020_task2_baseline#usage).

## 4. 1pytorch: PyTorch version baseline

`1pytorch/` is the PyTorch version of the original baseline done by myself.
You will find training and test code, accompanied with `config.yaml` and `model.py`.

Run a Jupyter notebook `00-train-with-visual.ipynb` to train your models, this will also show some visualizations.

And run `01_test.py` to evaluate your models as follows, make sure you run this under the folder `1pytorch/`.

```sh
your/1pytorch$ python 01_test.py -d
```

CAUTION: Performance reproduction is not stable so far.

## 5. 2vae_pytorch: VAE implementation

In a folder `2vae_pytorch/`, you can find notebooks of simple test results.

My conclusion is that we cannot apply VAE model as long as using the same data structure wth the original baseline.

It is known that VAE works fine with simple dataset like MNIST, but it doesn't with more complicated datasets.
Then simple dataset has clean clusters which VAE can model well, but current data is, I think, almost single cluster (because we are training 1 class samples) with larger variance.

- Attempt #1, tried VAE as is, model learned mean signal...
- Attempt #2, loss weights (Reconstruction: 1.0 vs. KLD: 0.01); closer to usual Autoencoder.
- Attempt #3, loss weights (Reconstruction: 1.0 vs. KLD: 0.0); i.e. non-variational, usual Autoencoder.
- Attempt #4, loss weights (Reconstruction: 0.01 vs. KLD: 1.0); ??
- Attempt #5, loss weights (Reconstruction: 0.0 vs. KLD: 1.0); What if we don't penalize reconstruction loss?

### 5.1 Sample result: Attempt #2, loss weights (Reconstruction: 1.0 vs. KLD: 0.01)

Closer to Autoencoder baseline result.

`result.csv`
```
ToyCar
id		AUC		pAUC
01		0.791331		0.665015
02		0.843655		0.766151
03		0.615191		0.540928
04		0.851536		0.700667
Average		0.775428		0.668190

ToyConveyor
id		AUC		pAUC
01		0.714744		0.579638
02		0.614856		0.525890
03		0.710169		0.577999
Average		0.679923		0.561176

fan
id		AUC		pAUC
00		0.566143		0.491788
02		0.691003		0.548893
04		0.599511		0.525408
06		0.695817		0.523692
Average		0.638118		0.522445

pump
id		AUC		pAUC
00		0.660000		0.569378
02		0.591712		0.569938
04		0.867700		0.670526
06		0.738824		0.589267
Average		0.714559		0.599777

slider
id		AUC		pAUC
00		0.970337		0.859551
02		0.769438		0.624483
04		0.915955		0.672383
06		0.674157		0.486103
Average		0.832472		0.660630

valve
id		AUC		pAUC
00		0.588571		0.509067
02		0.617083		0.518421
04		0.691250		0.513158
06		0.530583		0.482456
Average		0.606872		0.505775
```

### 5.2 Training curve visualization example

Thanks to PyTorch Lightning and tensorboard, we could visualize training curve. Here's an example.

![training curve](2vae_pytorch/2vae-loss.png)

## 6. 3cnn_ae_pytorch: Convolutional Autoencoder

This is good visualized solution, but the result is not good as you can find in `00-train-with-visual.ipynb`.
Leaving here as an example. If you will try this, you will need improvements and fine tuning to get better results.

1. Edit `image_preprocess_level1.py` so that it can reach your copy of dataset.
2. Run `python image_preprocess_level1.py` it will create preprocessed data files under `dev_data`.
3. Run `00-train-with-visual.ipynb` to train and evaluate models.
4. If you run `python 01_test.py -d`, you can evaluate models as same as other implementations.

## 7. Links

- [Original baseline repository - https://github.com/y-kawagu/dcase2020_task2_baseline](https://github.com/y-kawagu/dcase2020_task2_baseline)
