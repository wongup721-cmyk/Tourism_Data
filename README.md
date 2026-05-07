# A Task-Driven Multi-Agent Collaborative Framework for Dynamic Tourism Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-TDF--Agents-orange)
![Task](https://img.shields.io/badge/Task-Tourism%20Demand%20Forecasting-green)

This repository provides the code and data resources for the paper:

**A Task-Driven Multi-Agent Collaborative Framework for Dynamic Tourism Demand Forecasting**

Di Han, Bocheng Wang, Jianqing Li, Xicheng Du, Qixian Li, and Liangzhou Qu

## Introduction

Tourism demand forecasting has become more difficult in the post-pandemic period because travelers' decisions are influenced not only by historical demand patterns, but also by online sentiment, health-risk perception, policy changes, weather, holidays, and other heterogeneous signals.

The accompanying paper proposes **Tourism Demand Forecasting Agents (TDF-Agents)**, an end-to-end LLM-based multi-agent collaborative framework for tourism demand forecasting. TDF-Agents automates the workflow from multi-source data collection and processing to feature recommendation and forecasting. The empirical study uses weekly tourist arrival data for Macau and evaluates whether the feature sets constructed by TDF-Agents improve multiple forecasting models.

This repository contains the modelling datasets, comment sentiment data, and baseline forecasting implementations used in the empirical analysis.

## TDF-Agents Framework

![TDF-Agents Framework](assets/tdf-agent-framework.png)

TDF-Agents follows a four-layer collaborative design:

- **Data Collection Agent (DCA):** collects and integrates multi-source heterogeneous data, including structured sources and unstructured web information.
- **Data Processing Agent (DPA):** cleans, aligns, transforms, and standardizes raw data, then constructs candidate features.
- **Feature Recommendation Agent (FRA):** combines tourism-domain knowledge, retrieval-augmented reasoning, and regularization constraints to recommend interpretable feature sets.
- **Forecasting Agent (FA):** trains and evaluates time-series forecasting models using the recommended features.

The framework is designed to reduce manual intervention in tourism forecasting workflows and improve feature adaptability under changing market conditions, especially after major public health events.

## Main Contributions

- **Methodological innovation:** introduces an LLM-based multi-agent collaborative mechanism for tourism demand forecasting.
- **Standardized data paradigm:** builds a reproducible workflow for processing multi-source heterogeneous tourism data.
- **Empirical insight:** shows that infectious disease-related features become important predictors in the post-pandemic forecasting context.
- **Cross-model validation:** evaluates the TDF-Agents feature set across CNN, LSTM, TimesNet, TSMixer, and iTransformer baselines.

## Repository Structure

```text
.
├── Comment data/
│   ├── Comments_original/       # Original tourism comment data
│   └── Comments_processed/      # Sentiment-processed comment data
├── Models/
│   ├── Data/                    # Feature sets and modelling datasets
│   └── Baseline/
│       ├── CNN/                 # CNN baseline
│       ├── itransfomer/         # iTransformer baseline
│       └── timenet +tsmixer+lstm/
│           └── timenet +tsmixer+lstm/
│               ├── TimeNet.py
│               ├── Tsmixer.py
│               └── LSTM.py
└── README.md
```

## Data

The repository includes the following feature-set files under `Models/Data/`:

- `agent.csv` / `agent.xlsx`: feature set constructed by TDF-Agents.
- `agent_ex.csv` / `agent_ex.xlsx`: TDF-Agents feature set excluding infectious disease-related features.
- `MI.csv` / `MI.xlsx`: feature set selected by mutual information.
- `MI_ex.csv` / `MI_ex.xlsx`: MI feature set excluding infectious disease-related features.
- `mRMR.csv` / `mRMR.xlsx`: feature set selected by mRMR.
- `mRMR_ex.csv` / `mRMR_ex.xlsx`: mRMR feature set excluding infectious disease-related features.
- `Total_data.xlsx`: integrated modelling data.

The `Comment data/` directory contains original and sentiment-processed Macau tourism comment data used for sentiment-related feature construction.

## Baseline Models

The empirical evaluation uses five forecasting models:

- CNN
- LSTM
- TimesNet
- TSMixer
- iTransformer

Main entry files:

- `Models/Baseline/CNN/cnn.py`
- `Models/Baseline/itransfomer/itransformer.py`
- `Models/Baseline/timenet +tsmixer+lstm/timenet +tsmixer+lstm/TimeNet.py`
- `Models/Baseline/timenet +tsmixer+lstm/timenet +tsmixer+lstm/Tsmixer.py`
- `Models/Baseline/timenet +tsmixer+lstm/timenet +tsmixer+lstm/LSTM.py`

## Experimental Findings

The paper reports that incorporating the TDF-Agents feature set improves forecasting accuracy across all baseline models compared with using no external feature set. It also achieves strong performance compared with traditional feature-selection strategies such as MI and mRMR, especially on MAPE.

The ablation study further shows that removing infectious disease-related features generally increases forecasting error, supporting the conclusion that health-risk indicators are important for post-pandemic tourism demand forecasting.

## Installation

Recommended Python version: **3.9 or 3.10**.

Install the common dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn scipy tqdm einops reformer-pytorch sktime sympy tensorflow
```

Install PyTorch according to your CUDA or CPU environment:

```bash
pip install torch
```

For GPU usage, please follow the official PyTorch installation instructions for your local CUDA version.

## Usage

Clone the repository:

```bash
git clone https://github.com/wongup721-cmyk/Tourism_Data.git
cd Tourism_Data
```

Run a baseline model after updating the dataset path and target column name inside the selected script:

```bash
python "Models/Baseline/CNN/cnn.py"
```

For the TimeNet, TSMixer, LSTM, and iTransformer baselines, enter the corresponding model directory before running the script so that local imports such as `models` and `utils` can be resolved correctly.

## Notes

Some scripts retain absolute paths from the original experimental environment. Before running the code, update:

- dataset file path
- target column name
- feature-set file name
- time-frequency parameter

The repository is intended as the code and data companion for the paper's empirical study. The complete production-grade multi-agent service can be further packaged as an MCP service in future work.

## Citation

If this repository is useful for your research, please cite the accompanying paper:

```bibtex
@article{han2026tdfagents,
  title  = {A Task-Driven Multi-Agent Collaborative Framework for Dynamic Tourism Demand Forecasting},
  author = {Han, Di and Wang, Bocheng and Li, Jianqing and Du, Xicheng and Li, Qixian and Qu, Liangzhou},
  year   = {2026}
}
```
