#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created on 13 Nov 2024
# "Neural Network-Driven Exploration of Solvophilic Block Size Effects in Polymerization-Induced Self-Assembly:
# From 2D To 3D Comprehensive Pseudo-Phase Diagram"

# Authors:
# Erika Paola Fonseca Parra,a Jihad Oumerri,a Ana Arteni Andrea,b Jean-Luc Six,a Steven Peter Armes,c Khalid Ferji a
# a Université de Lorraine, CNRS, LCPM, F-54000 NANCY, France
# b Université Paris-Saclay, CEA, CNRS, I2BC, Gif-sur-Yvette, France  
# c Dainton Building, Department of Chemistry, The University of Sheffield, Brook Hill, Sheffield, South Yorkshire, UK

# khalid.ferji@univ-lorraine.fr

# This model uses three primary features to predict the morphology class of polymeric systems prepared by PISA process.

"""
Features:
- Mn_PGMA (g/mol): The molar mass of the solvophilic PGMA (Poly(glycidyl methacrylate)) block, in grams per mole, which directly influences the stability and structure of the formed aggregates.
- f_PHPMA: The mass fraction of the PHPMA (Poly(2-hydroxypropyl methacrylate)) block within the copolymer.
- Solid Content (%): The percentage of solid content, affecting the polymer concentration in the medium and influencing the formation of different morphologies.

Target Morphology Classes:
- S (Spheres): Spherical micelles
- W (Worms): worm-like micelles
- V (Vesicles): Membranes forming cavities
- M (Mixed): Assemblies with mixed morphologies

The model aims to predict the dominant morphology based on these parameters, facilitating the optimization of synthesis conditions to obtain specific structures.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import optuna.multi_objective as mo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import matplotlib.patches as mpatches
import joblib
from optuna.visualization import plot_pareto_front

# Set seeds for reproducibility
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

# Load the original dataset
data_path = '/path/to/your/data_sample.csv' # Replace "/path/to/your" with the path to your data folder

original_data = pd.read_csv(data_path)
print(f"Dataset shape: {original_data.shape}")

# Create a directory for saving figures
output_folder = '/path/to/your/output_folder' # Replace "/path/to/your output_folder" with the path to your output folder

os.makedirs(output_folder, exist_ok=True)

# Define colormap for morphology classes
#S: sphere, W: worm-like micelle, V: vesicle, M: mixture
viridis = plt.get_cmap('viridis_r')
base_colors = [viridis(0.1)[:3], viridis(0.7)[:3], viridis(1.0)[:3], viridis(0.3)[:3]]
classes = ['S', 'W', 'V', 'M']

# Function to obtain color based on morphology class
def get_color(class_label):
    if class_label in classes:
        return base_colors[classes.index(class_label)]
    return 'gray'  # Default color for unknown classes

# Calculate and visualize class distribution
class_counts = original_data.iloc[:, 4].value_counts(normalize=True) * 100
class_counts = class_counts.reindex(classes)

plt.figure(figsize=(12, 8))
class_counts.plot(kind='bar', color=[get_color(cls) for cls in classes])
plt.title(f'Percentage of Each Class - {len(original_data)} Data Points', fontsize=22)
plt.xlabel('Class')
plt.ylabel('Percentage (%)')
plt.savefig(os.path.join(output_folder, 'class_distribution.png'), dpi=300)


# Separate features (X) and target (y), scale and encode
X = original_data.iloc[:, [0, 2, 3]]
y = original_data.iloc[:, 4]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=0, stratify=y)

# Define model creation function with Optuna hyperparameter suggestions
def create_model(trial):
    model = tf.keras.Sequential()
    units1 = trial.suggest_int('units_layer1', 12, 24)
    units2 = trial.suggest_int('units_layer2', 6, 12)
    model.add(tf.keras.layers.Dense(units1, activation='relu'))
    model.add(tf.keras.layers.Dense(units2, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Objective function for Optuna study
def objective(trial):
    model = create_model(trial)
    batch_size = trial.suggest_int('batch_size', 15, 25)
    epochs = trial.suggest_int('epochs', 10, 500)
    early_stopping_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-3)
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopping_loss, reduce_lr], verbose=1)
    
    val_loss, val_accuracy = model.evaluate(X_test, y_test, verbose=0)
    train_accuracy = history.history['accuracy'][-1]
    train_loss = history.history['loss'][-1]
    gap_loss=abs(train_loss-val_loss)

    return train_accuracy, val_loss, gap_loss

# Define Optuna study and optimize
storage = f'sqlite:///{os.path.join(output_folder, "multi_objective_optimization.db")}'
study = mo.create_study(study_name='multi_objective_optimization',
                        directions=['maximize', 'minimize', 'minimize'])

study.optimize(objective, n_trials=100)

# Extract results and save as CSV
trials_data = []
for trial in study.trials:
    trial_info = {
        "number": trial.number,
        **trial.params,
        "train_accuracy": trial.values[0],
        "val_loss": trial.values[1]
    }
    trials_data.append(trial_info)

trials_df = pd.DataFrame(trials_data)
trials_df.to_csv(os.path.join(output_folder, "optuna_multi_objective_results.csv"), index=False)

# Function to select objectives 'Validation Loss' and 'Train Accuracy'
def select_train_accuracy_loss(trial):
    return trial.values[1], trial.values[0]

# Visualize the Pareto front for 'Validation Loss' and 'Train Accuracy'
fig = plot_pareto_front(
    study,
    targets=select_train_accuracy_loss,
    target_names=["val_loss", "train_accuracy"]
)

# Save the interactive plot as an HTML file
pareto_path = os.path.join(output_folder, 'pareto_front.html')
fig.write_html(pareto_path)


# Train final model with best hyperparameters (example)
best_params = {'units_layer1': 21, 'units_layer2': 9, 'learning_rate': 0.03, 'batch_size': 19, 'epochs': 300}
final_model = tf.keras.Sequential([
    tf.keras.layers.Dense(best_params['units_layer1'], activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(best_params['units_layer2'], activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

final_model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=best_params['learning_rate']),
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])

final_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

# Evaluate and save model
test_loss, test_accuracy = final_model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Save model and scaler/encoder
model_path = os.path.join(output_folder, 'final_model.keras')
final_model.save(model_path)
joblib.dump(scaler, os.path.join(output_folder, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(output_folder, 'encoder.pkl'))

# Generate confusion matrix and save
y_pred = np.argmax(final_model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}')
plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'), dpi=300)


# Load or create new data for predictions (example structure)
new_data = pd.DataFrame({
    'Mn_PGMA': [7500, 8800, 10400, 12500],  # replace with actual values
    'fmass': [0.75, 0.8, 0.85, 0.65],
    'solid_content': [15, 20, 25, 10],
})

# Scale the new data using the scaler fitted on training data
new_data_scaled = scaler.transform(new_data)

# Predict probabilities and classes
predicted_probabilities = final_model.predict(new_data_scaled)
predicted_classes = np.argmax(predicted_probabilities, axis=-1)
predicted_classes_labels = label_encoder.inverse_transform(predicted_classes)

# Append the predictions to the new data
new_data['Predicted_Morphology'] = predicted_classes_labels
print(new_data)




