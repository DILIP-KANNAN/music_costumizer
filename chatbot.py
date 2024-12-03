from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from any origin

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
    app.run(debug=True)
