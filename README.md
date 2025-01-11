# K-Nearest Neighbors Classifier App

## Project Overview
This project is an interactive **K-Nearest Neighbors (KNN) Classifier** implemented in Python using **Streamlit**. It allows users to visualize data, train a KNN model on a dataset, and make predictions for new inputs. The app also provides performance metrics like a confusion matrix and classification report.

## Features
- **Data Preview**: Displays the dataset used for training and testing.
- **KNN Classifier**: Uses the K-Nearest Neighbors algorithm for classification.
- **Interactive Input**: Users can input data to predict classifications in real-time.
- **Performance Metrics**:
  - Confusion Matrix
  - Classification Report
  - Accuracy Score

## Requirements
- Python 3.x
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

# Dataset
The dataset is a cleaned CSV file used for training and testing the model. It includes:

## Features: Numeric columns for model input.
## Target: Binary classification labels (M for Malignant and B for Benign).
# Usage
## Run the Streamlit app:
streamlit run knn.py
Explore the dataset by viewing the preview in the app.
## Enter new data points for prediction:
Input numerical values for each feature.
Click Predict Diagnosis to get the result.
## Use the checkboxes to view:
Confusion Matrix (Heatmap)
Classification Report and Accuracy Score
# Key Components
## Data Preprocessing:

Splits the dataset into training and testing subsets.
Applies feature scaling using StandardScaler.
## Model Training:

Implements a K-Nearest Neighbors Classifier with n_neighbors=5.
Trained on the scaled training dataset.
## Prediction:

Accepts user input for new data points.
Predicts whether the input is Malignant or Benign with confidence scores.
## Visualization:

Generates a confusion matrix as a heatmap using Seaborn.
Displays classification metrics.
## Example Input and Output
Input: Numerical values for all features (e.g., mean radius, texture, perimeter, etc.).
Output: Predicted classification (Malignant or Benign) with confidence.
## Contributions
Contributions are welcome! Feel free to fork the repository, make enhancements, and submit a pull request.


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hamasakram/K-Near-Algo.git
