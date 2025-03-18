from flask import Flask, render_template, request, jsonify , send_file
import utils
import json
import requests
from urllib.parse import urlencode
from flask_cors import CORS
import os 
from PIL import Image
import io


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
url1="https://cfd688fb5f212da237becf5a8ecca8b8.loophole.site/"
API_URL = "https://civitai.com/api/v1/models"
def fetch_models(query=None, limit=9, cursor=None, sort="Newest", types=None, nsfw=None):
    params = {
        "limit": limit,
        "sort": sort
    }
    if query:
        params["query"] = query
    if types:
        params["types"] = types
    if nsfw is not None:
        params["nsfw"] = nsfw
    if cursor:
        params["cursor"] = cursor

    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("items", []), data.get("metadata", {}).get("nextCursor", None)
    return [], None
@app.route("/get_blocks")
def get_blocks():
    blocks = [
        {
            "container": "Prompts",
            "elements": [
                {"type": "textarea", "id": "positive_prompt", "name": "Positive Prompt"},
                {"type": "textarea", "id": "negative_prompt", "name": "Negative Prompt"},
            ],
        },
        {
            "container": "Configuration",
            "elements": [
                {"type": "float-range", "id": "cfg", "name": "CFG Scale"},
                {"type": "range", "id": "steps", "name": "Steps"},
                {"type": "wid-hei", "id": "wid", "name": "Width"},
                {"type": "wid-hei", "id": "hei", "name": "Height"},
                {"type": "range", "id": "batch_size", "name": "Batch Size"},
                {"type": "select", "id": "sampler", "name": "Sampler", "options": ["euler"]},
                {"type": "select", "id": "schedular", "name": "Scheduler", "options": ["AYS"]},
            ],
        },
        {
            "container": "Uploads",
            "elements": [
                {"type": "file", "id": "image_upload", "name": "Upload Image"},
            ],
        },
        {
            "container": "Actions",
            "elements": [
                {"type": "button","id": "submit", "value": "Submit", "name": "sumbit"},
            ],
        },
    ]
    return jsonify(blocks)

@app.route("/modal")
def home():
    query = request.args.get("query", "")
    sort = request.args.get("sort", "Newest")
    types = request.args.get("types", "")
    nsfw = request.args.get("nsfw", None)
    nsfw = nsfw.lower() == "true" if nsfw else None
    cursor = request.args.get("cursor", None)

    models, next_cursor = fetch_models(query=query, cursor=cursor, sort=sort, types=types, nsfw=nsfw)

    return render_template("modal.html", models=models, query=query, sort=sort, types=types, nsfw=nsfw, next_cursor=next_cursor)
@app.route("/url")
def url():
    return jsonify({"url":url1})

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/adv")
def adv():
    return render_template("adv.html")

@app.route("/easyrun")
def easyrun():
    return render_template("easyrun.html")

@app.route("/stream")
def img():
    return send_file("static/1.png")

@app.route("/queue")
def queue():
    return jsonify(["1"])


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
@app.route('/gener', methods=["POST"])
def jj():
    data = request.get_json()
    response = requests.post(url1+"generate", json=data)
    lit=response.json()
    kk=[]
    for i in lit:
        kk.append(url1+i)
    return jsonify(kk)


IMAGE_DIR = "static"


@app.route('/compress', methods=['GET'])
def compress_image():
    img_name = request.args.get('name')
    if not img_name:
        return jsonify({"error": "Image name is required"}), 400

    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.isfile(img_path):
        return jsonify({"error": "Image not found"}), 404

    img = Image.open(img_path)
    compressed_bytes = io.BytesIO()
    img.save(compressed_bytes, format="WEBP", quality=30)
    compressed_bytes.seek(0)

    return send_file(compressed_bytes, mimetype='image/webp', as_attachment=True, download_name=f"{img_name}.webp")

@app.route('/list', methods=['GET'])
def list_images():
    if not os.path.isdir(IMAGE_DIR):
        return jsonify({"error": "Invalid directory"}), 400

    images = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    return jsonify({"images": images})
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
