import abc
import torch
from importlib import import_module
import torchvision

class ModelLoader:
    """Represent a ModelLoader class."""
    
    def __init__(
        self,
        model_path: str,
    ) -> None: 

        self.model_path: str = model_path

    def load_model(self):
        return NotImplementedError()
    
    def get_model(self):
        return NotImplementedError()


class JitScriptLoader(ModelLoader):
    """Pytorch JitScript"""

    def __init__(
        self,
        model_path: str,
    ) -> None:
        super().__init__(model_path)

        self.model: torch.jit.ScriptModule = None
        self.load_model()

    def load_model(self) -> None:
        self.model = torch.jit.load(self.model_path)
        self.model.eval()

    def get_model(self) -> torch.jit.ScriptModule:
        return self.model


class StateDictLoader(ModelLoader):
    """ Pytorch Ditionary State loader"""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        model_arc: str,
        num_classes: int,
    ) -> None:
        super().__init__(model_path)

        self.model_name: str = model_name
        self.model_arc: str = model_arc
        self.num_classes: int = num_classes

        self.model: torch.nn.Module = None

        self.load_model()

    def load_model(self) -> None:
        """ load a model using torch.load function """

        self.model = getattr(
            import_module("torchvision.models"),
            self.model_name,
        )

        self.model.fc = torch.nn.Linear(
            in_features=self.net.fc.in_features,
            out_features=self.num_classes,
            bias=True,
        )

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
    
    def get_model(self) -> torch.nn.Module:
        return self.model

        


class HuggingfaceLoader(ModelLoader):
    """ """

    def __init__(self) -> None:
        super().__init__()
    