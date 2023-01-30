from typing import Any, Union, Dict
from PIL import Image

import os
import yaml
import requests
import addict


def load_image(image: Union[str, "Image.Image"]) -> "Image.Image":
    """_summary_

    Args:
        image (Union[str, Image.Image]): _description_

    Raises:
        ValueError: _description_

    Returns:
        Image.Image: _description_
    """
    
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

def load_yaml(cfg: Union[str, Dict[str, Any]], attr: bool=True) -> Dict[str, Any]:
    """
    

    Args:
        cfg (Union[str, Dict[str, Any]]): _description_

    Returns:
        Dict[str, Any]: _description_
    """

    config = None

    if not os.path.exists(cfg):
        raise FileNotFoundError(
            "Incorret path used for config file. Should be in a local path."
        )

    with open(cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return addict.Dict(config) if attr else config
    