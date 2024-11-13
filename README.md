# PISA-PhasePredict

## Description:
This repository provides a predictive model for phase diagrams in copolymer systems, specifically targeting nanostructure morphologies (spheres, worms, vesicles, and mixed phases) in PGMA-PHPMA systems. Leveraging data from literature, this project employs a deep neural network (DNN) to predict morphologies based on key polymer characteristics such as molecular weight and solids content. The project is designed to minimize experimental workload by providing accurate phase predictions via a trained model.

## Background and Objective:
Phase morphology prediction is critical in material science, especially when working with nanostructures and copolymer systems. This project addresses phase diagram prediction for morphologies (spheres, worms, vesicles, or mixed phases) formed by copolymers, utilizing literature data to build a reliable model.

The goal is to streamline the prediction process, providing accurate morphology predictions based on key input features like molecular weight, solids content, and other copolymer-specific parameters.

## Getting Started:
To test and use this model, follow these steps:

### Install the Prerequisites::
Make sure to have an active virtual environment (optional but recommended).
Install the required dependencies listed in requirements.txt


### Download the Sample Data:
Download data_sample.csv from this repository, which contains sample data points with features required for model training and prediction. Use this file to test the functionality of the code.

The data have been collected from the following publications:

1.	Czajka, A.; Armes, S. P., In situ SAXS studies of a prototypical RAFT aqueous dispersion polymerization formulation: monitoring the evolution in copolymer morphology during polymerization-induced self-assembly. Chem. Sci. 2020, 11 (42), 11443-11454.
2.	Thompson, K. L.;  Mable, C. J.;  Cockram, A.;  Warren, N. J.;  Cunningham, V. J.;  Jones, E. R.;  Verber, R.; Armes, S. P., Are block copolymer worms more effective Pickering emulsifiers than block copolymer spheres? Soft Matter 2014, 10 (43), 8615-8626.
3.	Blanazs, A.;  Madsen, J.;  Battaglia, G.;  Ryan, A. J.; Armes, S. P., Mechanistic Insights for Block Copolymer Morphologies: How Do Worms Form Vesicles? J. Am. Chem. Soc. 2011, 133 (41), 16581-16587.
4.	Blanazs, A.;  Ryan, A.; Armes, S., Predictive phase diagrams for RAFT aqueous dispersion polymerization: effect of block copolymer composition, molecular weight, and copolymer concentration. Macromolecules 2012, 45 (12), 5099-5107.
5.	Blanazs, A.;  Verber, R.;  Mykhaylyk, O. O.;  Ryan, A. J.;  Heath, J. Z.;  Douglas, C. W. I.; Armes, S. P., Sterilizable Gels from Thermoresponsive Block Copolymer Worms. J. Am. Chem. Soc. 2012, 134 (23), 9741-9748.
6.	Li, Y.; Armes, S. P., RAFT Synthesis of Sterically Stabilized Methacrylic Nanolatexes and Vesicles by Aqueous Dispersion Polymerization. Angew. Chem. Int. Ed. 2010, 49 (24), 4042-4046.
7.	Warren, N. J.;  Mykhaylyk, O. O.;  Ryan, A. J.;  Williams, M.;  Doussineau, T.;  Dugourd, P.;  Antoine, R.;  Portale, G.; Armes, S. P., Testing the Vesicular Morphology to Destruction: Birth and Death of Diblock Copolymer Vesicles Prepared via Polymerization-Induced Self-Assembly. J. Am. Chem. Soc. 2015, 137 (5), 1929-1937.
8.	Chambon, P.;  Blanazs, A.;  Battaglia, G.; Armes, S., Facile synthesis of methacrylic ABC triblock copolymer vesicles by RAFT aqueous dispersion polymerization. Macromolecules 2012, 45 (12), 5081-5090.

### Cite the code: 
[![DOI](https://zenodo.org/badge/887578376.svg)](https://doi.org/10.5281/zenodo.14143425)




### Code Execution
The code is provided in a single Python script, DNN_prediction_nano_PISA_V1.py, which includes the full workflow: data preprocessing, hyperparameter optimization, model training, and prediction.
