from typing import Union
from PIL import Image
import os
import requests


def load_image(image: Union[str, "PIL.Image.Image"]) -> "PIL.Image.Image":
    
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
    elif isinstance(image, Image.Image):
        image = image
    
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL Image."
        )
    
    image = image.convert("RGB")
    return image