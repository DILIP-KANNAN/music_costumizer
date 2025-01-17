
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Function to preprocess audio
def preprocess_audio(file_path, sr=22050, duration=30):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        # Normalize audio length
        y = librosa.util.fix_length(y, size=sr * duration)
        return y, sr
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return None, None

# Function to extract features
def extract_features(y, sr):
    try:
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
        
        # Return the extracted features
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

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
            if y is not None and sr is not None:
                # Extract features
                features = extract_features(y, sr)
                if features is not None:
                    print(f"Extracted features for {file}: {features.shape}")
                    data.append([features, genre])
                else:
                    print(f"No features extracted for {file}")
            else:
                print(f"Skipping file {file} due to loading issues.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Create a DataFrame
df = pd.DataFrame(data, columns=['Features', 'Genre'])

# Check the shape of the feature matrix
X = np.array(df['Features'].tolist())
print(f"Shape of X before scaling: {X.shape}")

# Ensure X is not empty
if X.size == 0:
    print("No valid features found. Exiting process.")
else:
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Shape of X after scaling: {X_scaled.shape}")

    # Save the scaled data to a CSV (or as needed)
    df_scaled = pd.DataFrame(X_scaled)
    df_scaled['Genre'] = np.array(df['Genre'].tolist())
    df_scaled.to_csv('scaled_features.csv', index=False)
    print("Scaled features saved to 'scaled_features.csv'.")
