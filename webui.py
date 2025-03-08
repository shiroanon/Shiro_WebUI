from flask import Flask, render_template, request, jsonify
import utils
import json
import requests
from urllib.parse import urlencode
from flask_cors import CORS


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
url1="https://b83e809a5ad63227a030f6632813c3fe.loophole.site/"

@app.route("/url")
def url():
    return jsonify({"url":url1})

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/easyrun")
def easyrun():
    return render_template("easyrun.html")

@app.route('/gallery')
def gallery():
    # Fetch images from Civitai API directly
    params = {
        'limit': request.args.get('limit', 100),
        'nsfw': request.args.get('nsfw', 'None'),
        'sort': request.args.get('sort', 'Newest'),
        'period': request.args.get('period', 'AllTime'),
        'page': request.args.get('page', 1)
    }
    
    response = requests.get('https://civitai.com/api/v1/images', params=params)
    image_data = response.json()
    
    return render_template('gallery.html', 
                           images=image_data['items'], 
                           metadata=image_data.get('metadata', {}),
                           params=params)
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
