import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# Load dataset
@st.cache  # Use Streamlit's cache to load data only once
def load_data():
    data = pd.read_csv("KNNAlgorithmDataset.csv")
    data = data.drop(columns=['Unnamed: 32'])  # Cleaning dataset
    return data

dt = load_data()

# Display the first few rows of the dataset
st.write("Data Preview:", dt.head())

# Prepare the data for training and prediction
X = dt.drop(['id', 'diagnosis'], axis=1)
y = dt['diagnosis'].replace(['M', 'B'], [1, 0])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# User input for new data prediction
st.write("### Enter new data for prediction")
input_data = {}
for column in X.columns:
    input_data[column] = st.number_input(f"Enter {column}:", value=float(X[column].mean()))

# Button to make prediction
if st.button("Predict Diagnosis"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = knn.predict(input_scaled)
    prediction_proba = knn.predict_proba(input_scaled)
    
    st.write(f"The prediction for the entered data is: {'Malignant' if prediction[0] == 1 else 'Benign'}")
    st.write(f"Confidence: {np.max(prediction_proba) * 100:.2f}%")

# Plotting functions
def plot_confusion_matrix():
    y_pred = knn.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    return fig

if st.checkbox("Show Confusion Matrix"):
    st.pyplot(plot_confusion_matrix())

# Showing accuracy and classification report
if st.checkbox("Show Classification Report and Accuracy"):
    y_pred = knn.predict(X_test_scaled)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

