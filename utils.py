import re
import random
import string
import re
from collections import defaultdict
import json
from grammer_m import grammar_map
import requests 
from PIL import Image
import time
import sys


def round_to_multiple_of_8(n):
    return int(round(n / 8.0) * 8)


def generate_random_string(length=16):
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


def parse_parameters(param_str_input):
    defaults = {
        "positive_prompt": "",
        "steps": 25,
        "cfg": 5.5,
        "batch_size": 1,
        "neg": "low quality",
        "width": 1024,
        "height": 1024,
        "seed": None,
    }
    result_dict = defaults.copy()

    PORTRAIT_WIDTH, PORTRAIT_HEIGHT = 768, 1024
    LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT = 1024, 768
    SQUARE_WIDTH, SQUARE_HEIGHT = 1024, 1024  # Can be same as default, or different

    param_str_processed = param_str_input

    def remove_flag_from_string(text, flag):
        pattern = rf"(?<!\w){re.escape(flag)}(?!\w)\s*"
        return re.sub(
            pattern, "", text, 1
        )  # Remove only the first recognized occurrence

    if "--portrait" in param_str_processed:
        # Check if it's a standalone flag before applying
        if re.search(rf'(?<!\w){re.escape("--portrait")}(?!\w)', param_str_processed):
            result_dict["width"] = PORTRAIT_WIDTH
            result_dict["height"] = PORTRAIT_HEIGHT
            param_str_processed = remove_flag_from_string(
                param_str_processed, "--portrait"
            )

    if "--landscape" in param_str_processed:
        if re.search(rf'(?<!\w){re.escape("--landscape")}(?!\w)', param_str_processed):
            result_dict["width"] = LANDSCAPE_WIDTH
            result_dict["height"] = LANDSCAPE_HEIGHT
            param_str_processed = remove_flag_from_string(
                param_str_processed, "--landscape"
            )

    if "--square" in param_str_processed:
        if re.search(rf'(?<!\w){re.escape("--square")}(?!\w)', param_str_processed):
            result_dict["width"] = SQUARE_WIDTH
            result_dict["height"] = SQUARE_HEIGHT
            param_str_processed = remove_flag_from_string(
                param_str_processed, "--square"
            )

    # --- Main parameter parsing ---
    pattern = r"--(\w+)\s+(\"[^\"]*\"|\'[^\']*\'|[^\s]+)"

    parsed_params = {}
    prompt_segments = []
    current_pos = 0

    for match in re.finditer(pattern, param_str_processed):
        param_name = match.group(1)
        param_value_raw = match.group(2)

        prompt_segments.append(param_str_processed[current_pos : match.start()])

        value_stripped = param_value_raw.strip()
        if (value_stripped.startswith("'") and value_stripped.endswith("'")) or (
            value_stripped.startswith('"') and value_stripped.endswith('"')
        ):
            if len(value_stripped) > 1:
                value_stripped = value_stripped[1:-1]

        parsed_params[param_name] = value_stripped.strip()
        current_pos = match.end()

    prompt_segments.append(param_str_processed[current_pos:])
    result_dict["positive_prompt"] = " ".join(
        s.strip() for s in prompt_segments if s.strip()
    ).strip()

    try:
        if "steps" in parsed_params:
            result_dict["steps"] = int(parsed_params["steps"])
        if "cfg" in parsed_params:
            result_dict["cfg"] = float(parsed_params["cfg"])
        if "batch_size" in parsed_params:
            result_dict["batch_size"] = int(parsed_params["batch_size"])
        if "neg" in parsed_params:
            result_dict["neg"] = parsed_params["neg"]

        if "seed" in parsed_params:
            result_dict["seed"] = int(parsed_params["seed"])
        elif result_dict["seed"] is None:
            result_dict["seed"] = random.randint(0, 2**32 - 1)

        if "size" in parsed_params:
            size_str = parsed_params["size"]
            parts = size_str.lower().split("x")
            if len(parts) == 2:
                w_str, h_str = parts
                width_raw = int(w_str.strip())
                height_raw = int(h_str.strip())
                result_dict["width"] = (
                    width_raw  # Temporarily set, rounding happens below
                )
                result_dict["height"] = height_raw
            else:
                raise ValueError("Size format must be INTxINT (e.g., 1024x768)")

        # Round final width and height to multiple of 8, regardless of source
        result_dict["width"] = round_to_multiple_of_8(result_dict["width"])
        result_dict["height"] = round_to_multiple_of_8(result_dict["height"])

        return result_dict

    except (ValueError, TypeError) as e:
        if "size" in parsed_params and (
            not isinstance(result_dict.get("width"), int)
            or not isinstance(result_dict.get("height"), int)
        ):
            result_dict["width"], result_dict["height"] = (
                defaults["width"],
                defaults["height"],
            )

        if "steps" in parsed_params and not isinstance(result_dict.get("steps"), int):
            result_dict["steps"] = defaults["steps"]

        if "cfg" in parsed_params and not isinstance(result_dict.get("cfg"), float):
            result_dict["cfg"] = defaults["cfg"]

        if "batch_size" in parsed_params and not isinstance(
            result_dict.get("batch_size"), int
        ):
            result_dict["batch_size"] = defaults["batch_size"]

        if "seed" in parsed_params and not isinstance(result_dict.get("seed"), int):
            result_dict["seed"] = random.randint(0, 2**32 - 1)
        elif result_dict["seed"] is None:
            result_dict["seed"] = random.randint(0, 2**32 - 1)

        result_dict["width"] = round_to_multiple_of_8(
            result_dict.get("width", defaults["width"])
        )
        result_dict["height"] = round_to_multiple_of_8(
            result_dict.get("height", defaults["height"])
        )

        return result_dict


def load_booru_tags(file_path):
    
    booru_tags = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tags = [
                t.strip().lower().replace(" ", "_") for t in line.strip().split(",")
            ]
            booru_tags.update(tags)
    return booru_tags


def clean_and_tokenize(prompt):
    cleaned = re.sub(r"[^\w\s]", "", prompt.lower())
    return cleaned.split()


def extract_tags(tokens, valid_tags):
    matched_tags = []
    i = 0
    while i < len(tokens):
        matched = False
        for j in range(3, 0, -1):
            if i + j <= len(tokens):
                phrase = "_".join(tokens[i : i + j])
                if phrase in valid_tags:
                    matched_tags.append(phrase)
                    i += j
                    matched = True
                    break
        if not matched:
            matched_tags.append(tokens[i])
            i += 1
    return matched_tags

def normalize_token(token):
    return grammar_map.get(token, token)


def process_prompt(prompt, booru_tags):
    tokens = clean_and_tokenize(prompt)
    matched_tags = extract_tags(tokens, booru_tags)
    normalized_tags = [normalize_token(t) for t in matched_tags]
    return ", ".join(normalized_tags)


def find_positive_negative_nodes(workflow):

    sampler_node = next(
        node for node in workflow.values() if node.get("class_type") == "SamplerCustom"
    )

    positive_id = sampler_node["inputs"]["positive"][0]

    negative_id = sampler_node["inputs"]["negative"][0]

    return positive_id, negative_id

def find_all_node_ids_by_class(workflow, node_class):
    matching_node_ids = [
        node_id
        for node_id, node_data in workflow.items()
        if node_data.get("class_type") == node_class
    ]
    return matching_node_ids

def find_sampler_node_id(workflow,node_class):
    sampler_node_id = next(
        (node_id for node_id, node_data in workflow.items()
        if node_data.get("class_type") == node_class),
        None
    )
    return sampler_node_id


def get_request_with_retries(url, retries=5):
    for _ in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed with error: {e}. Retrying...")


def know_your_setup(url):
    object_info=get_request_with_retries(url+"/object_info")
    Samplers=object_info["KSampler"]["input"]["required"]["sampler_name"][0]
    Scheduler=object_info["KSampler"]["input"]["required"]["scheduler"][0]
    Checkpoints=object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    setup_info={"Samplers":Samplers,"Scheduler":Scheduler,"Checkpoints":Checkpoints}
    return setup_info

def send_prompt(prompt_input):

    filename=generate_random_string()
    url = "https://584fd67c2614ab1c65322cb37a0e9585.loophole.site"
    setup_info=know_your_setup(url) 
    prompt_info=parse_parameters(prompt_input)
    with open ("jj.json","r") as file :
        data=json.load(file)
        data["1"]["inputs"]["ckpt_name"] = setup_info["Checkpoints"][0]
        positive_id,negetive_id=find_positive_negative_nodes(data)
        data[positive_id]["inputs"]["text"] = prompt_info["positive_prompt"]
        data[negetive_id]["inputs"]["text"] = prompt_info["neg"]
        data["2"]["inputs"]["noise_seed"]= prompt_info["seed"]
        data["7"]["inputs"]["width"] = prompt_info["width"]
        data["7"]["inputs"]["height"] = prompt_info["height"]
        data["7"]["inputs"]["batch_size"] = prompt_info["batch_size"]
        data["9"]["inputs"]["filename_prefix"] = filename
        prompt={"prompt":data}
        request = requests.post(url+"/prompt" , json=prompt)
        print(request.json())  
        response=get_request_with_retries(url+"/queue")
        print()
        while response["queue_running"] != []:
            time.sleep(1)
            response=get_request_with_retries(url+"/queue")
            print()
        images_names=[]
        print(images_names)
        im_res=get_request_with_retries(url+"/shiro/image") 
        print()
        for i in im_res["file_list"]:
            if i.startswith(filename):
                images_names.append(i)
        import term_image.image  
        print(images_names)           
        for i in images_names:
            image = term_image.image.from_url(url+"/shiro/image/webp/"+i)
            image.draw()
            
       
          
        

send_prompt(input("Enter your prompt: "))