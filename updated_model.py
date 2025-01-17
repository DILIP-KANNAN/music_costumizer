import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the scaled feature data from CSV
df_scaled = pd.read_csv('scaled_features.csv')

# Separate features (X) and target labels (y)
X = df_scaled.drop('Genre', axis=1).values  # Features (everything except Genre column)
y = df_scaled['Genre'].values  # Target labels (Genre)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features (though they are already scaled, it's good practice to scale again during model training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a classification model (using SVC as an example)
model = SVC(kernel='linear')  # Using Support Vector Classifier with linear kernel

# Fit the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optionally, save the trained model and scaler for future use
with open('music_genre_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved as 'music_genre_model.pkl' and 'scaler.pkl'.")
