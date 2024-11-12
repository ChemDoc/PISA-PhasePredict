# PISA-PhasePredict
his repository provides a predictive model for phase diagrams in copolymer systems, specifically targeting nanostructure morphologies (spheres, worms, vesicles, and mixed phases) in PGMA-PHPMA systems. Leveraging data from literature, this project employs a deep neural network (DNN) to predict morphologies based on key polymer characteristics such as molecular weight and solids content. The project is designed to minimize experimental workload by providing accurate phase predictions via a trained model.

Objective: To aid researchers in predicting phase morphology in copolymer nanostructures using the Polymerization-Induced Self-Assembly (PISA) method, thereby reducing the need for extensive experimental trials.

Features:

Data Preprocessing: Manage and prepare copolymer data for predictive modeling.
Hyperparameter Optimization: Uses Optuna to fine-tune DNN hyperparameters for optimal performance.
Model Training: Trains a neural network to make high-accuracy predictions on phase morphologies.
Phase Prediction: Provides predictions for new copolymer data, assisting in experimental planning.
