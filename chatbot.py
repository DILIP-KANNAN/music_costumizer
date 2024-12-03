from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/chat": {"origins": "*"}})

@app.route('/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Ensure the request is in JSON format
        data = request.get_json()
        user_message = data.get('message')
        responses = {
        "What does this app do?": "This app classifies music genres from audio files you upload.",
        "How do I upload a file?": "Click the upload button and select an audio file in MP3 or WAV format.",
        "What format should the audio file be in?": "Supported formats are MP3 and WAV with a maximum duration of 30 seconds.",
        "What genre should I listen to?": "I recommend trying classical music for a relaxing experience.",
        "Why am I getting an error?": "Make sure your audio file is in the correct format and under 30 seconds long."
    }

        response = responses.get(user_message, "I'm sorry, I don't understand that question.")
        print(f"User message: {user_message}\n")
        print(f"Available keys: {list(responses.keys())}\n")
        print(user_message)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
