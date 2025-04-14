from flask import Flask, render_template, request, jsonify , send_file
import utils.utils as utils
import json
import requests
from urllib.parse import urlencode
from flask_cors import CORS
import os 
from PIL import Image
import io
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
