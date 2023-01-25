import abc
from typing import Optional, List, Tuple, Union
from PIL import Image
import requests
import os
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch.transforms import ToTensorV2

class Preprocess(abc.ABC):
    """Represent a preprocess class."""
    
    @abc.abstractmethod
    def __call__(self):
        pass


class ImagePreprocess(Preprocess):

    def __init__(
        self,
        resize: Optional[Tuple[int, int]] = (256, 256), 
        std: Optional[Tuple[int, int, int]] = (0.229, 0.224, 0.225),
        mean: Optional[Tuple[int, int, int]] = (0.485, 0.456, 0.406),
    ) -> None:

        self.resize = resize
        self.std = std
        self.mean = mean
        
        self.transform = Compose(
            [
                Resize(height=self.resize[0], width=self.resize[1]),
                Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )

    def __call__(self, image):

        return self.transform(image=image)["image"]
        

    
class TextPreprocess(Preprocess):

    def __init__(
        self,
    ) -> None:
        pass

