from flask import Flask, render_template, request, jsonify , send_file
import utils.base_utils as base_utils
import json
import requests
from urllib.parse import urlencode
from flask_cors import CORS
import os 
from PIL import Image
import io
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
URL="http://127.0.0.1:9191/"
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/easyrun", methods=["POST"])
def easy():
    data = request.get_json()
    response = {
        "prompt": data.get("prompt"),
        "parameters": data.get("parameters")
    }
    workflow=open("workflows/kk.shiro","r").read()
    imgs=base_utils.execute_easy(URL,response["prompt"],response["parameters"],workflow)
    return jsonify(imgs)

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
