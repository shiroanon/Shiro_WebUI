from flask import Flask, render_template, request, jsonify
import utils
import json

import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/easyrun")
def easyrun():
    return render_template("easyrun.html")

@app.route("/gallery")
def gallery():
    return render_template("gallery.html")

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.get_json()
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
