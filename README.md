# Plant Disease Detection Using CNNs

This repository contains code for a machine learning project focused on detecting plant diseases from images using Convolutional Neural Networks (CNNs). The project leverages TensorFlow to preprocess and train on a dataset of plant images, detecting various plant diseases based on image data.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The goal of this project is to classify plant diseases using images and convolutional neural networks. The model is trained on a large dataset of plant leaves, each belonging to one of 38 classes of plant diseases. The project aims to help farmers and agricultural experts quickly and accurately diagnose plant diseases using machine learning models.

## Dataset
The dataset used in this project is the **PlantVillage Dataset**, which contains 70,296 images across 38 different classes of plant diseases. Images are categorized by disease type, and the dataset is split into training and validation sets.

- **Number of images**: 70,296
- **Number of classes**: 38

## Project Structure
- `PlantDiseases -10.ipynb`: Jupyter notebook containing the project code, from preprocessing to model training and evaluation.
- `train/`: Directory containing the training images.
- `test/`: Directory containing the test images.

## Setup
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Amitabh0954/plant-disease-detection.git

2. Navigate to the project directory:
Copy code
cd plant-disease-detection
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Download the dataset and place it in the train/ and test/ folders.

## Usage
To train the CNN model on the plant disease dataset:

Open the PlantDiseases -10.ipynb notebook.
Follow the steps in the notebook to preprocess the data and train the model.
The notebook includes sections for training set preprocessing, validation set preprocessing, and model evaluation.
Results
The CNN model was trained on the PlantVillage dataset and achieved high accuracy on the validation set. Detailed results and evaluation metrics can be found in the notebook under the Results section.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or create a pull request.

## License
This project is licensed under the MIT License.


