from flask import Flask, request, jsonify
import joblib
import librosa
import numpy as np
import pickle
from flask_cors import CORS

from sklearn.discriminant_analysis import StandardScaler

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load the trained model and the scaler
model = joblib.load('C:/Users/user/OneDrive/Desktop/My Project 1/random_forest_model.joblib')

# Load the scaler used during training
with open('C:/Users/user/OneDrive/Desktop/My Project 1/music_costumizer/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

if isinstance(scaler, StandardScaler):
    print("Scaler loaded successfully!")
else:
    print("Loaded object is not a StandardScaler.")
    print(type(scaler))

# Function to preprocess audio (same as training preprocessing)
def preprocess_audio(file, sr=22050, duration=30):
    # Load audio file
    y, sr = librosa.load(file, sr=sr, duration=duration)
    # Normalize audio length
    y = librosa.util.fix_length(y, size=sr * duration)
    return y, sr

# Function to extract features (same as training feature extraction)
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

@app.route('/')
def home():
    return 'Welcome to Music Genre Classifier API'

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an audio file is provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Process the audio file (example using librosa to extract features)
    try:
        # Preprocess the audio
        audio_data, sr = preprocess_audio(file)
        
        # Extract features
        features = extract_features(audio_data, sr)
        
        # Reshape features to match model input
        features = features.reshape(1, -1)
        
        # Normalize the features using the pre-trained scaler
        features_scaled = scaler.transform(features)
        
        # Predict the genre
        prediction = model.predict(features_scaled)
        genres = ['beats', 'classical', 'hip-hop', 'melody', 'western']

        # Get the genre name using the predicted index
        genre = genres[prediction[0]]

        return jsonify({'genre': genre})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').lower()

    responses = {
        "what does this app do?": "This app classifies music genres from audio files you upload.",
        "how do i upload a file?": "Click the upload button and select an audio file in MP3 or WAV format.",
        "what format should the audio file be in?": "Supported formats are MP3 and WAV with a maximum duration of 30 seconds.",
        "what genre should i listen to?": "I recommend trying classical music for a relaxing experience.",
        "why am i getting an error?": "Make sure your audio file is in the correct format and under 30 seconds long."
    }

    response = responses.get(user_message, "I'm sorry, I don't understand that question.")
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
