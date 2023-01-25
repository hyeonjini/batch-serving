from src.loader.loader import ModelLoader
from src.preprocess.preprocess import Preprocess

import torch
from abc import ABC, abstractmethod
from typing import Union, List


class Classifier(ABC):
    def __init__(
        self,
        loader: ModelLoader,
    ) -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        self.loader: ModelLoader = loader
        self.model = self.loader.get_model()
        self.model.to(self.device)

    def __call__(self, inputs, **kwargs):

        is_list = isinstance(inputs, list)

        preprocess_parms = {}
        forward_params = {}
        postprocess_params = {}

        if is_list:
            return self.run_multi(inputs, preprocess_parms, forward_params, postprocess_params)
        
        return self.run_single(inputs, preprocess_parms, forward_params, postprocess_params)

    @abstractmethod
    def forward(self, inputs):
        return NotImplementedError("forward not implemented")

    @abstractmethod
    def preprocess(self, inputs):
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

        >>> from src.classifier import classifier
        >>> from src.loader import loader
        >>> from src.preprocess import preprocess

        >>> test_loader = loader.JitScriptLoader(model_path="PATH/TO/MODEL")
        >>> test_preprocess = preprocess.ImagePreprocess()
        >>> test_classifier = classifier.ImageClassifier(loader=test_loader, image_preprocess=test_preprocess)

    Args:
        Classifier (_type_): _description_
    """
    def __init__(
        self,
        image_preprocess:Preprocess, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.image_preprocess = image_preprocess

    # def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
    #     pass

    def __call__(self, inputs:torch.Tensor, **kwargs):
        """_summary_

        Args:
            inputs (torch.Tensor): _description_

        Returns:

            The dictionaries contain the following keys:
            label:`str` -- The label identified by the model.
            score: `float` -- The score attributed by the model for that label. 
            _type_: _description_
        """
        return super().__call__(inputs, **kwargs)
    
    def forward(self, inputs, **kwargs):
        from datetime import datetime
        start_time = datetime.now()
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
        predictions = None
        # with torch.no_grad():
        with torch.inference_mode():
            inputs = inputs.to(self.device)
            # predictions = torch.nn.functional.softmax(self.model(inputs), dim=-1)
            # predictions = predictions.cpu().numpy()
            predictions = self.model(inputs)
        print("ImageClassifier.forward:", datetime.now()-start_time)
        return predictions
    
    def preprocess(self, inputs, **kwargs):
        from datetime import datetime
        start_time = datetime.now()
        if not self.image_preprocess:
            return ValueError("`image_preprocess` cannot be None.")
        preprocess = self.image_preprocess(inputs)
        if len(preprocess.shape) == 3:
            preprocess = preprocess.unsqueeze(0)
        print("ImageClassifier.preprocess:", datetime.now()-start_time)
        return preprocess
    
    def postprocess(self, outputs, **kwargs):
        from datetime import datetime
        start_time = datetime.now()
        outputs = torch.nn.functional.softmax(outputs, dim=-1)
        outputs = outputs.detach().cpu().numpy()
        print("ImageClassifier.postprocess:", datetime.now()-start_time)
        return [output for output in outputs] 


class TextClassifier(Classifier):
    def __init__(self) -> None:
        super().__init__()