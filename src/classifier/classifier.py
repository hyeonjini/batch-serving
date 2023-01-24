from src.loader.loader import ModelLoader
import torch

class Classifier:
    def __init__(
        self,
        model_path:str,
        loader: ModelLoader,
    ) -> None:

        self.model_path: str = model_path
        self.loader: ModelLoader = loader
        self.model = self.loader.get_model()

    def inference(self):
        return NotImplementedError()
    
    def __str__(self) -> str:
        return str(type(self))


class ImageClassifier(Classifier):
    def __init__(self) -> None:
        pass

    def inference(self, data):
        outputs = {}
        with torch.no_grad():
            print("inference code")

        return outputs

class TextClassifier(Classifier):
    def __init__(self) -> None:
        super().__init__()
    
    def inference(self, data):

        with torch.no_grad():

            print("inference code")

        return super().inference()

class BrothyClassifier(ImageClassifier):
    def __init__(self) -> None:
        pass

    def inference(self):
        return super().inference()

class NoodleClassifier(ImageClassifier):
    def __init__(self) -> None:
        pass

    def inference(self):
        return super().inference()

class RiceClassifier(ImageClassifier):
    def __init__(self) -> None:
        super().__init__()

    def inference(self):
        return super().inference()