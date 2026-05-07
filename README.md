# Data Tourism

This repository contains the code and data used for tourism time-series modelling and sentiment-related data processing in the accompanying paper.

## Repository Structure

- `Models/`: model code and experiment data.
- `Models/Baseline/CNN/`: CNN baseline implementation.
- `Models/Baseline/itransfomer/`: iTransformer baseline implementation.
- `Models/Baseline/timenet +tsmixer+lstm/`: TimeNet, TSMixer, and LSTM baseline implementations.
- `Models/Data/`: processed modelling datasets in CSV/XLSX format.
- `Comment data/`: original and processed comment data.

## Main Entry Points

- `Models/Baseline/CNN/cnn.py`
- `Models/Baseline/itransfomer/itransformer.py`
- `Models/Baseline/timenet +tsmixer+lstm/timenet +tsmixer+lstm/TimeNet.py`
- `Models/Baseline/timenet +tsmixer+lstm/timenet +tsmixer+lstm/Tsmixer.py`
- `Models/Baseline/timenet +tsmixer+lstm/timenet +tsmixer+lstm/LSTM.py`

## Environment

Recommended Python version: 3.9 or 3.10.

Common dependencies include:

- einops
- matplotlib
- numpy
- pandas
- reformer-pytorch
- scikit-learn
- scipy
- sktime
- sympy
- tqdm
- torch
- tensorflow

Install dependencies as needed, for example:

```bash
pip install numpy pandas matplotlib scikit-learn scipy tqdm einops reformer-pytorch sktime sympy tensorflow
```

Install PyTorch according to the official instructions for your CUDA/CPU environment.

## Notes

Some scripts contain local absolute data paths from the original experiment environment. Before running, update the data paths and target column names according to the dataset used in your environment.

