# LSTM-Multibranch (2023 project for statistics project) 
A study on the prediction of systematic risk in stock market based on multibranch LSTM model with multidimensional heterogeneous perspective
# A Study on the Prediction of Systematic Risk in Stock Market

This project explores the prediction of systematic risk in the stock market using a multi-branch LSTM model. The study integrates multidimensional and heterogeneous data perspectives to enhance prediction accuracy, combining various data preprocessing and feature extraction techniques.

## Project Structure

- **Data**:
  - `raw_data`: Contains the raw dataset used for the study.
  - `ExportData.xls`: Processed and structured dataset for experiments.

- **Scripts**:
  - `bert.py` and `bert_preprocess.py`: Scripts for processing and extracting features using BERT for textual data.
  - `cnn+lstm.py`: Combines CNN and LSTM models for feature extraction and prediction.
  - `gru.py`, `lstm.py`: Implementation of GRU and LSTM models for sequential data analysis.
  - `our_model.py`: Core implementation of the multi-branch LSTM model.

- **Notebooks**:
  - `graph_feature_extraction.ipynb`: Feature extraction using graph-based techniques.

- **MATLAB**:
  - `CoES.m`: MATLAB script for CoES (Conditional Expected Shortfall) calculations.

## Key Features

- **Multi-Branch LSTM Model**: Leverages multiple input branches for processing diverse data sources.
- **Heterogeneous Data Perspectives**: Incorporates text, numerical, and graph features to improve prediction accuracy.
- **Graph-Based Feature Extraction**: Utilizes network analysis techniques to extract key features from stock market data.
- **Integration of Deep Learning Models**: Combines CNN, LSTM, GRU, and BERT for comprehensive data analysis.
- **Risk Metrics**: Implementation of financial risk metrics, including CoES, for better decision-making.

## Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
