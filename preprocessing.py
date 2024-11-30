import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Function to preprocess audio
def preprocess_audio(file_path, sr=22050, duration=30):
    # Load audio file
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    # Normalize audio length
    y = librosa.util.fix_length(y, size=sr * duration)
    return y, sr

# Function to extract features
def extract_features(y, sr):
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    
    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = np.mean(chroma.T, axis=0)
    
    # Extract Spectrogram features
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = np.mean(spectrogram.T, axis=0)
    
    # Combine features
    features = np.concatenate((mfccs, chroma, spectrogram))
    return features

# Paths and genres
dataset_path = r"C:\Users\user\Downloads\Dataset"
genres = ['beats', 'classical', 'hip-hop', 'melody', 'western']

# Prepare data
data = []

for genre in genres:
    genre_folder = os.path.join(dataset_path, genre)
    print(f"Processing genre: {genre}")
    for file in os.listdir(genre_folder):
        file_path = os.path.join(genre_folder, file)
        try:
            # Preprocess audio
            y, sr = preprocess_audio(file_path)
            # Extract features
            features = extract_features(y, sr)
            # Append features and label
            data.append([features, genre])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Create a DataFrame
df = pd.DataFrame(data, columns=['Features', 'Genre'])

# Separate features and labels
X = np.array(df['Features'].tolist())
y = np.array(df['Genre'].tolist())

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save preprocessed data
preprocessed_data = {'features': X_scaled, 'labels': y}
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(preprocessed_data, f)

print("Preprocessing complete. Data saved as 'preprocessed_data.pkl'.")
