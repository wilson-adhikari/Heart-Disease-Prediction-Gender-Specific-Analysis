# Heart Disease Prediction: Gender-Specific Analysis

This project predicts heart disease using patient health metrics, with a focus on gender-specific analysis. The workflow includes data cleaning, preprocessing, exploratory analysis, machine learning, and deep learning models, allowing for both male and female datasets to be analyzed separately. Trained models are saved for future predictions.

---
## Neural Network Details

- Input Layer: Matches the number of features in the dataset.
- Hidden Layers:
    - Dense layer with 64 neurons and ReLU activation.
    - Dropout layer (0.2) for regularization.
    - Dense layer with 32 neurons and ReLU activation.
    -  Dropout layer (0.2).
- Output Layer: Single neuron with sigmoid activation.
- Optimizer: Adam with learning rate 0.0001.
- Loss Function: Binary crossentropy.
- Metrics: Accuracy.
---
## Features

- **Data Cleaning & Preprocessing:**  
  - Removes duplicates and missing values.  
  - Standardizes categorical features (e.g., `Sex` column).  
  - Splits the dataset into male and female subsets with valid age and cholesterol values.  

- **Exploratory Data Analysis (EDA):**  
  - Visualizes age group distributions for males and females using heatmaps.  
  - Helps understand demographic differences and data patterns.  

- **Machine Learning Models:**  
  - **Logistic Regression** for baseline predictions.  
  - Evaluates model performance using accuracy, confusion matrix, and classification reports.  

- **Neural Network Models:**  
  - Feedforward neural network with two hidden layers and dropout layers.  
  - Sigmoid activation in output for binary classification (heart disease: yes/no).  
  - Adam optimizer with low learning rate (0.0001) and binary crossentropy loss.  
  - Supports training, validation, and evaluation on gender-specific datasets.  

- **Model Saving:**  
  - Neural network models are saved as `.keras` files (`male_model.keras` and `female_model.keras`) for future use.  

---
## Dependencies
- Python 3.9+
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow / keras
---
## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
pip install pandas matplotlib seaborn scikit-learn tensorflow
