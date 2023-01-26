from abc import ABC, abstractmethod
from typing import Optional, Tuple
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch.transforms import ToTensorV2
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

class Preprocess(ABC):
    """Represent a preprocess class."""
    
    @abstractmethod
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

    def __call__(self, image) -> Tensor:

        return self.transform(image=image)["image"]
        

class TextPreprocess(Preprocess):

    def __init__(
        self,
        tokenizer: (PreTrainedTokenizer | PreTrainedTokenizerFast),
        tokenizer_config:dict,
    ) -> None:
        super().__init__()
        
        self.tokenizer = tokenizer
        self.tokenizer_config = tokenizer_config
    
    def __call__(self, text:str) -> Tuple[Tensor, Tensor]:
        
        inputs = self.tokenizer(text, **self.tokenizer_config)

        return inputs["input_ids"], inputs["attention_mask"]

