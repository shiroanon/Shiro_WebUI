import json
import random
import re
from flask import jsonify
import re
import random

import re
import random

def clean_prompt(prompt):
    params = {
        'cfg': 5.0,
        'steps': 25,
        'batch_size': 1,
        'seed': random.randint(1, 0xFFFFFFFF),
        'width': 512,
        'height': 512,
        'scale': 1.0
    }

    # Regex patterns
    param_pattern = re.compile(r'(?i)(cfg|steps|batch_size|seed)\s*[:=]?\s*([\d.]+)')
    p_pattern = re.compile(r'(?i)-p\s+(\d+)x(\d+)')
    s_pattern = re.compile(r'(?i)-s\s+([\d.]+)')
    portrait_pattern = re.compile(r'(?i)-portrait')
    landscape_pattern = re.compile(r'(?i)-landscape')

    clean_prompt = prompt

    # Process parameters
    for match in param_pattern.finditer(prompt):
        key = match.group(1).lower()
        value = match.group(2)
        try:
            if key == 'cfg':
                params[key] = float(value)
            elif key in ['steps', 'batch_size']:
                params[key] = max(1, int(float(value)))
            elif key == 'seed':
                params['seed'] = int(float(value))
            clean_prompt = clean_prompt.replace(match.group(0), '')
        except ValueError as e:
            print(f"Invalid parameter value: {match.group(0)} - {str(e)}")

    # Process -p parameter (width x height)
    p_match = p_pattern.search(clean_prompt)
    if p_match:
        try:
            params['width'] = int(p_match.group(1))
            params['height'] = int(p_match.group(2))
        except ValueError as e:
            print(f"Invalid width/height in -p flag {p_match.group(0)}: {str(e)}")
        clean_prompt = clean_prompt.replace(p_match.group(0), '', 1)

    # Process -s parameter (scale)
    s_match = s_pattern.search(clean_prompt)
    if s_match:
        try:
            params['scale'] = float(s_match.group(1))
        except ValueError as e:
            print(f"Invalid scale in -s flag {s_match.group(0)}: {str(e)}")
        clean_prompt = clean_prompt.replace(s_match.group(0), '', 1)

    # Set predefined sizes for -portrait and -landscape
    if portrait_pattern.search(clean_prompt):
        params['width'], params['height'] = 768, 512
        clean_prompt = re.sub(portrait_pattern, '', clean_prompt, 1)

    if landscape_pattern.search(clean_prompt):
        params['width'], params['height'] = 512, 768
        clean_prompt = re.sub(landscape_pattern, '', clean_prompt, 1)

    # Apply scale factor
    params['width'] = int(params['width'] * params['scale'])
    params['height'] = int(params['height'] * params['scale'])

    # Clean up prompt
    clean_prompt = re.sub(r'\s+', ' ', clean_prompt).strip()

    return {
        "prompt": clean_prompt,
        "cfg": params["cfg"],
        "steps": params["steps"],
        "batch_size": params["batch_size"],
        "seed": params["seed"],
        "wid": params["width"],
        "hei": params["height"]
    }

def replace_with_json_values(text):
    """Replace specific words with values from a JSON file"""
    with open("template.json", "r", encoding="utf-8") as f:
        replacements = json.load(f)

    for key, value in replacements.items():
        text = text.replace(key, value.strip())

    return text