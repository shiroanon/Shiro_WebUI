from flask import Flask, render_template, request, jsonify
import utils
import json
import requests

from flask_cors import CORS


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
url1="https://ccc4f9ca5c632abf6a9c5bedcd453855.loophole.site/z"

@app.route("/url")
def url():
    return jsonify({"url":url1})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/easyrun")
def easyrun():
    return render_template("easyrun.html")

@app.route("/gallery")
def gallery():
    return render_template("gallery.html")

@app.route("/gen", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt=data["positive_prompt"]
    o=utils.clean_prompt(prompt)
    
    response = requests.post(url1+"generate", json=o)
    lit=response.json()
    print(lit)
    kk=[]
    for i in lit:
        kk.append(url1+i)
    return jsonify({"img":kk})    

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
