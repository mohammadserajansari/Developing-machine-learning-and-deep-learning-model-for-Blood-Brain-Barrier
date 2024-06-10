# Developing machine learning and deep learning model for Blood Brain Barrier

This repository contains a Python script that predicts whether a given molecule can cross the blood-brain barrier (BBB) based on its SMILES representation. The model uses a Random Forest classifier trained on molecular descriptors.

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bbbpredict.streamlit.app/)

## Overview

The script performs the following steps:

1. Converts a given SMILES string to its isomeric form.
2. Calculates molecular descriptors using the Mordred library.
3. Filters out molecules with disconnected fragments.
4. Loads a pre-trained Random Forest model along with the scaler and selected feature names.
5. Transforms the descriptors using the scaler and makes a prediction using the loaded model.

## Installation

To set up the environment and install the required libraries, follow these steps:

1. Clone the repository:
    ```bash
    git clone (https://github.com/mohammadserajansari/Developing-machine-learning-and-deep-learning-model-for-Blood-Brain-Barrier.git
    cd bbb_prediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    conda create -n python=3.8

    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    
4. Run Streamlit script
   ```bash 
   streamlit run updated_try_bbb.py
   ```
## Usage

1. Run the script to make the prediction:
    ```bash
    streamlit run updated_try_bbb.py
    ```

2. The script will output whether the molecule is predicted to be able to cross the blood-brain barrier:
    On web app interface enter the SMILES & AI model will predict the permeability 
   ```
   Enter the SMILES : --> COc1ccc2c(c1)nc([nH]2)S(=O)Cc1ncc(c(c1C)OC)C --> Our assessment concludes that the molecule's BBB Permeability is negative
   ```
