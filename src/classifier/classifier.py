from src.loader.loader import ModelLoader
from src.preprocess.preprocess import Preprocess

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Optional, Tuple
from PIL import Image


class Classifier(ABC):
    """
    Represent a Classifier class

    Args:
        preprocessor (Preprocess):
        loader (ModelLoader):
        id2label (dict):
        device (str):
    """
    def __init__(
        self,
        preprocessor: Preprocess,
        loader: ModelLoader,
        id2label:dict,
        device: Optional[str] = None,
    ) -> None:

        self.preprocessor: Preprocess = preprocessor
        self.loader: ModelLoader = loader
        self.id2label = id2label

        self.device = device
        if self.device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = self.loader.get_model()
        self.model.to(self.device)

    def __call__(self, inputs, **kwargs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """

        is_list = isinstance(inputs, list)

        preprocess_parms = {}
        forward_params = {}
        postprocess_params = {}

        if is_list:
            return self.run_multi(inputs, preprocess_parms, forward_params, postprocess_params)
        
        return self.run_single(inputs, preprocess_parms, forward_params, postprocess_params)

    @abstractmethod
    def forward(self, inputs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        return NotImplementedError("forward not implemented")

    @abstractmethod
    def preprocess(self, inputs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        return NotImplementedError("preprocess not implemented")
    
    @abstractmethod
    def postprocess(self, inputs):
        """
        Postprocess will receive the raw outpus of the `forward` method, generally tensors, and reformat them into
        something more friendly.

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        return NotImplementedError("postprocess not implemented")

    def run_multi(self, inputs, preprocess_parms, forward_params, postprocess_params):
        return [self.run_single(item, preprocess_parms, forward_params, postprocess_params) for item in inputs]

    def run_single(self, inputs, preprocess_parms, forward_params, postprocess_params):
        model_inputs = self.preprocess(inputs, **preprocess_parms)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs


class ImageClassifier(Classifier):
    """
    This pipeline predicts the class of an image.

    Example:
    ```python
    >>> from src.classifier import classifier
    >>> from src.loader import loader
    >>> from src.preprocess import preprocess

    >>> test_loader = loader.JitScriptLoader(model_path="PATH/TO/MODEL")
    >>> test_preprocess = preprocess.ImagePreprocess()
    >>> test_classifier = classifier.ImageClassifier(loader=test_loader, image_preprocess=test_preprocess)
    >>> test_classifier("sample.jpg")
    {'score': 0.9872, 'label': True}
    ```
    """
    def __init__(
        self,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)

    def __call__(self, images: Union[torch.Tensor, List[torch.Tensor], "Image.Image", List["Image.Image"]], **kwargs):
        """_summary_

        Args:
            images (Union[torch.Tensor, List[torch.Tensor], Image.Image, List[Image.Image]]):
                The pipeline handles four type of images:

                - An image 


        Returns:
            _type_: _description_
        """
        return super().__call__(images, **kwargs)
    
    def forward(self, inputs:torch.Tensor, **kwargs):
        """_summary_

        Args:
            inputs (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)

        predictions = None

        with torch.inference_mode():
            inputs = inputs.to(self.device)
            predictions = self.model(inputs)

        return predictions
    
    def preprocess(self, inputs, **kwargs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not self.preprocessor:
            return ValueError("`preprocessor: Preprocess` cannot be `None`.")
        
        if isinstance(inputs, torch.Tensor):
            return inputs

        if isinstance(inputs, Image.Image):
            inputs = np.array(inputs)

        return self.preprocessor(inputs)
    
    def postprocess(self, model_outputs:torch.Tensor, **kwargs):
        """_summary_

        Args:
            model_outputs (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        outputs = torch.nn.functional.softmax(model_outputs, dim=-1)
        values, indices = torch.max(outputs, dim=-1)
        values = values.tolist()
        indices = indices.tolist()

        outputs = outputs.cpu().numpy()

        if self.id2label:
            return [{self.id2label[index]:round(value, 4)} for value, index in zip(values, indices)]

        return [{index:round(value, 4)} for value, index in zip(values, indices)]


class TextClassifier(Classifier):
    """
    This pipeline predicts the class of a text.

    Example:

        >>> from src.classifier import classifier
        >>> from src.loader import loader
        >>> from src.preprocess import preprocess

        >>> test_loader = loader.
        >>> test_preprocess = preprocess.TextPreprocess()
        >>> test_classifier = classifier.TextClassifier()

    Args:
        text_preprocess (Prerocess): _description_
        id2label (dict): _description_
        loader (ModelLoader): _dsecription_
    """
    
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def __call__(self, text: Union[str, List[str]], **kwargs):
        """_summary_

        Args:
            text (Union[str, List[str]]): _description_

        Returns:
            _type_: _description_
        """
        return super().__call__(text, **kwargs)

    def forward(self, inputs:Tuple[torch.Tensor, torch.Tensor], **kwargs) -> torch.Tensor:
        """_summary_

        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor]): _description_

        Returns:
            torch.Tensor: _description_
        """

        predictions = None

        with torch.inference_mode():
            token_ids, attantion_masks = inputs
            token_ids = token_ids.to(self.device)
            attantion_masks = attantion_masks.to(self.device)
            predictions = self.model(token_ids, attantion_masks)[0]

        return predictions.cpu()
    
    def preprocess(self, inputs:str, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            inputs (str): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        if not isinstance(inputs, str):
            return inputs
            
        return self.preprocessor(inputs)
    
    def postprocess(self, model_outputs:torch.Tensor, **kwargs):
        """
            postprocess in TextClassification will make model output 

        Args:
            model_outputs (torch.Tensor): model output from `forward`

        Returns:
            list: 
        """
        outputs = torch.nn.functional.softmax(model_outputs, dim=-1)
        values, indices = torch.topk(outputs, k=3, dim=-1)
        values = values.tolist()[0]
        indices = indices.tolist()[0]

        if self.id2label:
            return [{self.id2label[index]:round(value, 4)} for value, index in zip(values, indices)]

        return [{index:round(value, 4)} for value, index in zip(values, indices)]

