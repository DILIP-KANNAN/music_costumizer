import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import pickle

# Load the scaled feature data from CSV
df_scaled = pd.read_csv('scaled_features.csv')

# Separate features (X) and target labels (y)
X = df_scaled.drop('Genre', axis=1).values  # Features (everything except Genre column)
y = df_scaled['Genre'].values  # Target labels (Genre)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features (although they are already scaled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **Model 1: SVM with rbf Kernel**
svm_model = SVC(kernel='rbf', random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid_svm = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 1]}
grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5)
grid_search_svm.fit(X_train_scaled, y_train)

print("Best SVM model parameters:", grid_search_svm.best_params_)

# Evaluate the SVM model
svm_best_model = grid_search_svm.best_estimator_
y_pred_svm = svm_best_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Model Accuracy: {accuracy_svm:.2f}")

# **Model 2: Random Forest Classifier**
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5)
grid_search_rf.fit(X_train_scaled, y_train)

print("Best Random Forest model parameters:", grid_search_rf.best_params_)

# Evaluate the Random Forest model
rf_best_model = grid_search_rf.best_estimator_
y_pred_rf = rf_best_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {accuracy_rf:.2f}")

# Cross-validation score for more reliable results
cv_scores_svm = cross_val_score(svm_best_model, X, y, cv=5)
cv_scores_rf = cross_val_score(rf_best_model, X, y, cv=5)

print("\nSVM Cross-validation Scores:", cv_scores_svm)
print(f"Average SVM Cross-validation Accuracy: {cv_scores_svm.mean():.2f}")

print("\nRandom Forest Cross-validation Scores:", cv_scores_rf)
print(f"Average Random Forest Cross-validation Accuracy: {cv_scores_rf.mean():.2f}")

# Save the best model (Random Forest or SVM)
model_to_save = rf_best_model if accuracy_rf > accuracy_svm else svm_best_model
with open('music_genre_model.pkl', 'wb') as model_file:
    pickle.dump(model_to_save, model_file)

# Save the scaler for future use
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Best Model and scaler saved as 'music_genre_model.pkl' and 'scaler.pkl'.")
