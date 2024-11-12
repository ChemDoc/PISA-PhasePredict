# PISA-PhasePredict

## Description:
This repository provides a predictive model for phase diagrams in copolymer systems, specifically targeting nanostructure morphologies (spheres, worms, vesicles, and mixed phases) in PGMA-PHPMA systems. Leveraging data from literature, this project employs a deep neural network (DNN) to predict morphologies based on key polymer characteristics such as molecular weight and solids content. The project is designed to minimize experimental workload by providing accurate phase predictions via a trained model.

## Background and Objective:
Phase morphology prediction is critical in material science, especially when working with nanostructures and copolymer systems. This project addresses phase diagram prediction for morphologies (spheres, worms, vesicles, or mixed phases) formed by copolymers, utilizing literature data to build a reliable model.

The goal is to streamline the prediction process, providing accurate morphology predictions based on key input features like molecular weight, solids content, and other copolymer-specific parameters.

## Getting Started:
To test and use this model, follow these steps:

### Prerequisites:
- numpy==1.26.4 
- pandas==2.2.1 
- matplotlib==3.8.3 
- seaborn==0.13.2 
- scikit-learn==1.4.1 
- tensorflow==2.16.1 
- optuna==3.6.1 

### Download the Sample Data:
Download data_sample.csv from this repository, which contains sample data points with features required for model training and prediction. Use this file to test the functionality of the code.

### Code Execution
The code is provided in a single Python script, phase_diagram_prediction.py, which includes the full workflow: data preprocessing, hyperparameter optimization, model training, and prediction.
