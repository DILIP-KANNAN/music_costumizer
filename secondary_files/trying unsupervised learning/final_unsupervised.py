import os
import pandas as pd
import pickle
import numpy as np
import librosa

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


# Define the input (either a directory or a single file)
new_music_input = r'C:\Users\user\Music\test-maari.mp3'


audio_files = new_music_input

# Load the saved model and scaler
with open(r"C:\Users\user\Downloads\music_genre_model_using_clusters.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open(r"C:\Users\user\Downloads\scaler.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)



try:
    # Extract features for the audio file
    y, sr = preprocess_audio(audio_files)
    features = extract_features(y, sr)
    print(len(features))
    if features is not None:
        # Reshape the features array to be 2D (single sample)
        features = features.reshape(1, -1)  # Reshaped to 2D for StandardScaler

        # Scale the features
        scaled_features = scaler.transform(features)

        # Predict the cluster using the model
        predicted_cluster = model.predict(scaled_features)

        # Append the results
        results ={
            "File": audio_files,
            "Cluster": predicted_cluster[0]
        }
    else:
        print(f"Failed to process file: {audio_files}")
except Exception as e:
    print(f"Error processing {audio_files}: {e}")
print(results)