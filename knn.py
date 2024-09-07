# %%
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve


#load dataset
dt=pd.read_csv("KNNAlgorithmDataset.csv")
print('Successfully read! ')

# %%
# Display the first few rows of the dataset
print(dt.head())

# Summary of the dataset
print(dt.info())

# Summary statistics for numeric columns
print(dt.describe())

# Check for missing values
print(dt.isnull().sum())


# %%
#cleaning the dataset
dt=dt.drop(columns=['Unnamed: 32'])

# %%
#Preparing the data for further training
X=dt.drop(['id','diagnosis'],axis=1)
y=dt['diagnosis']



# %%
# Display the first few rows of the features and target
print("Feature Set:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

# Basic stats of the features
print("\nFeature Summary:")
print(X.describe())


# %%
# Checking for missing values in the features
print("\nMissing Values in Features:")
print(X.isnull().sum())

# Checking for missing values in the target
print("\nMissing Values in Target:")
print(y.isnull().sum())


# %%
# Histograms of the features
X.hist(bins=20, figsize=(20, 15))
plt.show()


# %%
# Box plots for each feature to identify outliers
for column in X.columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=X[column])
    plt.title(f'Box Plot of {column}')
    plt.show()


# %%
sns.countplot(x=y)
plt.title('Distribution of Diagnosis')
plt.show()


# %%
#Converting M and B to 1 and 0
y.replace(['M','B'],[1,0],inplace=True)



# %%


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # Using 5 neighbors for KNN
knn.fit(X_train, y_train)

# Predicting the test set results
y_pred = knn.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%

# Range of 'k' to test
k_range = range(1, 31)

# List to store the average accuracy for each value of 'k'
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')  # 10-fold cross-validation
    k_scores.append(scores.mean())

# Find the value of 'k' that has the maximum score
best_k = k_range[k_scores.index(max(k_scores))]
print("Best k:", best_k)


# %%
# Rebuilding the KNN model with K=7
knn_optimized = KNeighborsClassifier(n_neighbors=7)
knn_optimized.fit(X_train, y_train)
y_pred_optimized = knn_optimized.predict(X_test)

# Evaluating the optimized model
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
print("Optimized Accuracy:", accuracy_optimized)
print("\nOptimized Classification Report:\n", classification_report(y_test, y_pred_optimized))


# %%


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Configuring the KNeighborsClassifier with specified parameters
knn = KNeighborsClassifier(n_neighbors=11, weights='distance', algorithm='ball_tree', metric='euclidean')
knn.fit(X_train, y_train)

# Predicting the test set results
y_pred = knn.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with specified parameters:", accuracy)

# Since you know the parameters, there's no need for grid search or training score calculations



# %%
#Calculating F1-Score now
from sklearn.metrics import f1_score
f1_score(y_test, y_pred_best, average='weighted')

# %%

# Generate the confusion matrix from the true labels and predicted labels
conf_matrix = confusion_matrix(y_test, y_pred_best)

# Plotting the confusion matrix using Seaborn's heatmap function
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix for KNN Classifier')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# %%
train_sizes, train_scores, validation_scores = learning_curve(
    estimator = KNeighborsClassifier(n_neighbors=best_k), # Use the best k from previous findings
    X = X_train, 
    y = y_train, 
    train_sizes = np.linspace(0.1, 1.0, 10), # Use 10 increments from 10% to 100% of the training data
    cv = 5, # Cross-validation splitting strategy
    scoring = 'accuracy', # Evaluation metric
    n_jobs = -1 # Use all available cores
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
validation_mean = np.mean(validation_scores, axis=1)
validation_std = np.std(validation_scores, axis=1)


# %%
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, color='green', alpha=0.1)

plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, validation_mean, 'o-', color='green', label='Validation score')

plt.title('Learning Curve for KNN Classifier')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# %%


# %%



