# Sound Source Localization Neural Network

## Project Overview

This project was developed as part of an internship at LAAS-CNRS, Toulouse, France by Emily Holmes (me!). 
As of project completion, I am an engineering student in Electrical Engineering pursuing a Master's degree in Embedded Systems.

This project focuses on sound source localization using neural networks. The project was supervised and mentored by Daniela Dragomirescu, Patrick Danès and Gaël Loubet. 

## Introduction

Sound source localization is a crucial task in audio processing, with applications in fields such as robotics, environment monitoring, and human-computer interaction. 

This project focuses on simple neural network architectures, specifically implemented in Python using TensorFlow, to tackle sound source localization. This simplicity is key, as the latter part of the project focuses on hardware integration of a neural network onto FPGA with a focus on energy-efficiency. 

We specifically tackle problems with a singular source. 

## Installation recommendations

- **Python Version**: This project is written in Python 3.11.4.
- **TensorFlow Version**: It is recommended to use TensorFlow version 2.13. You can follow the installation recommendations provided on the [official TensorFlow website](https://www.tensorflow.org/install) to ensure you have the compatible version installed.

## Files

The project consists of the following files:

1. **compute_features.py**:
   - This script performs the pre-processing of the input features, extracting GCC-PHAT (Generalized Cross-Correlation with Phase Transform), and pre-processing the output labels to prepare the data for training.

2. **fc_model.py**:
   - This script: 
     - Pipelines the data for training to handle a significant amount of data without relying (entirely) on RAM.
     - Defines the neural network model architecture.
     - Performs the training and saves the weights and biases.
     - Performs predictions using the trained model.

3. **compare.py**:
   - This script is used to visualize the model's outputs and compare them with ground truth values. It aids in evaluating the model's performance.

4. **spec_py.py**:
   - This script implements SpecAugment to artificially increase the dataset size.

## Dataset
The pre-processing was done with the SSLR (Sound Source Localization for Robots) dataset in mind, which is available [here](https://zenodo.org/record/4555356). Permission must be granted for download.

## Usage

To use this project for sound source localization, follow these steps:

1. Use `compute_features.py` to pre-process your input features and labels.
2. Use `fc_model.py` to train the model.
3. Use `compare.py` to visualize and evaluate the model's performance against ground truth data.

`spec_aug.py` may be used to increase data size, especially with azimuth poorly represented.

## Acknowledgments

I would like to express our gratitude to LAAS-CNRS for providing the opportunity to work on this internship and for the valuable guidance provided by Daniela Dragomirescu, Patrick Danès and Gaël Loubet throughout the project.
