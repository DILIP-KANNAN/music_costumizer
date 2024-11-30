import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# Step 1: Load Preprocessed Data
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load()

X = np.array(data['features'])  # Features
y = np.array(data['labels'])    # Labels

# Step 2: Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)  # One-hot encode labels for CNN

# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Step 4: Reshape Data for CNN
# Add a "channel" dimension for CNN input shape (samples, timesteps, channels)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Step 5: Build the CNN Model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

# Step 6: Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)

# Step 8: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Step 9: Save the Model
model.save('music_genre_cnn_model.h5')

# Step 10: Predict on New Samples
# Example: Predict the genre of the first test sample
sample_features = X_test[0].reshape(1, X_test.shape[1], 1)  # Reshape for prediction
prediction = model.predict(sample_features)
predicted_genre = label_encoder.inverse_transform([np.argmax(prediction)])
print("Predicted Genre:", predicted_genre[0])