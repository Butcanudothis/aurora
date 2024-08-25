from flask import Flask, request, jsonify
import os
from faiss_service import load_index, get_response

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']

    load_index()  # Load FAISS index from S3
    response = get_response(question)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8084)
