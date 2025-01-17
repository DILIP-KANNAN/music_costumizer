import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
            features_with_label = np.concatenate((features, [genre]))  # Add genre as label
            data.append(features_with_label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Ensure data is not empty
print(f"Total entries in data: {len(data)}")

# Create a DataFrame
df = pd.DataFrame(data)

# Separate features and labels
X = np.array(df.iloc[:, :-1])  # All columns except the last one (features)
y = np.array(df.iloc[:, -1])  # The last column (genre/label)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a DataFrame with the scaled features and labels
df_scaled = pd.DataFrame(X_scaled)
df_scaled['Genre'] = y

# Save to CSV
df_scaled.to_csv('audio_features.csv', index=False)

print("Preprocessing complete. Data saved as 'audio_features.csv'.")
