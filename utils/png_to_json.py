import os
import json
from PIL import Image

def read_exif_and_create_json(image_path):

    img = Image.open(image_path)



    metadata = img.info
    return metadata

 
print(read_exif_and_create_json("/home/shiro/Desktop/Shiro_WebUI/test.png")     )       
