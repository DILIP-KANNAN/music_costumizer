from flask import Flask, request, jsonify
import joblib
import librosa
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('random_forest_model.joblib')

# Route for handling audio file upload and prediction
@app.route('/predict', methods=['POST'])
def predict_audio():
    try:
        # Check if an audio file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if the file is actually an audio file
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Load the audio file using librosa
        audio, sr = librosa.load(file, sr=None)
        
        # Extract features from the audio file (e.g., MFCCs)
        mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)  # Take the mean across time axis to get a fixed-length vector
        
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Predict the genre using the trained model
        prediction = model.predict(features)
        
        # Send the prediction as a response (e.g., predicted genre class)
        return jsonify({'genre': int(prediction[0])})  # Return the predicted genre class
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)