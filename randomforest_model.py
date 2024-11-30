import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load preprocessed data (assuming X and y are already prepared)
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.array(data['features'])  # Features
y = np.array(data['labels'])    # Labels

# Step 2: Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y) # One-hot encode labels for CNN


# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# Step 3: Initialize and train RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 4: Evaluate model on test data
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")
joblib.dump(rf_model, 'random_forest_model.joblib')
print("Model saved successfully!")