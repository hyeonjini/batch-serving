import abc
from typing import Optional, List, Tuple

class Preprocess(abc.ABC):
    """Represent a preprocess class."""
    
    @abc.abstractmethod
    def preprocess(self):
        pass


class ImagePreprocess(Preprocess):

    def __init__(
        self, 
        resize: Tuple(int, int),
    ) -> None:
        pass
    
class TextPreprocess(Preprocess):

    def __init__(
        self,
    ) -> None:
        pass

    