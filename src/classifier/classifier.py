from collections import abc


class Classifier:
    def __init__(self, model_path:str) -> None:
        pass

    def inference(self):
        return NotImplementedError()
    
    def __str__(self) -> str:
        return str(type(self))


class ComputerVisionClassifier(Classifier):
    def __init__(self) -> None:
        pass

    def inference(self):
        return super().inference()

class NLPClassifier(Classifier):
    def __init__(self) -> None:
        super().__init__()
    
    def inference(self):
        return super().inference()

class BrothyClassifier(ComputerVisionClassifier):
    def __init__(self) -> None:
        pass

    def inference(self):
        return super().inference()

class NoodleClassifier(ComputerVisionClassifier):
    def __init__(self) -> None:
        pass

    def inference(self):
        return super().inference()

class RiceClassifier(ComputerVisionClassifier):
    def __init__(self) -> None:
        super().__init__()

    def inference(self):
        return super().inference()

class TextClassifier(NLPClassifier):
    def __init__(self) -> None:
        super().__init__()

    def inference(self):
        return super().inference()
