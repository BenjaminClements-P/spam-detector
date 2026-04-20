"""
Flask Web API for the Email Spam Detection System
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template_string
from src.spam_detector import SpamDetector

app = Flask(__name__)
detector = SpamDetector()

# Auto-train if no model found
if not detector.is_trained:
    from train import SPAM_SAMPLES, HAM_SAMPLES
    texts  = SPAM_SAMPLES + HAM_SAMPLES
    labels = ["spam"] * len(SPAM_SAMPLES) + ["ham"] * len(HAM_SAMPLES)
    detector.train(texts, labels)

HTML = open(os.path.join(os.path.dirname(__file__), "templates/index.html"), encoding="utf-8").read()
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data.get("subject", "") + " " + data.get("body", "")).strip()
    if not text:
        return jsonify({"error": "Empty email content"}), 400
    result = detector.predict(text)
    stats  = detector.email_index.stats()
    return jsonify({**result, "stats": stats})

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    text  = (data.get("subject", "") + " " + data.get("body", "")).strip()
    label = data.get("correct_label")
    if label not in ("spam", "ham"):
        return jsonify({"error": "Invalid label"}), 400
    detector.learn_from_feedback(text, label)
    return jsonify({"message": f"Thanks! Model updated with your feedback ({label}).",
                    "stats": detector.email_index.stats()})

@app.route("/stats")
def stats():
    return jsonify(detector.email_index.stats())

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
