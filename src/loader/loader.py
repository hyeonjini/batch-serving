import abc
import torch
from importlib import import_module

class ModelLoader:
    """Represent a ModelLoader class."""
    
    def __init__(
        self,
        model_path: str,
    ) -> None: 

        self.model_path: str = model_path

    def _load_model(self):
        return NotImplementedError()
    
    def get_model(self):
        return NotImplementedError()


class JitScriptLoader(ModelLoader):
    """Pytorch JitScript"""
    warm_up = 3
    def __init__(
        self,
        model_path: str,
    ) -> None:
        super().__init__(model_path)

        self.model: torch.jit.ScriptModule = None
        self._load_model()

    def _load_model(self) -> None:
        self.model = torch.jit.load(self.model_path)
        self.model.eval()

        # warm up
        with torch.no_grad():
            for _ in range(self.warm_up):
                self.model(torch.randn(3, 3, 224, 224))

    def get_model(self) -> torch.jit.ScriptModule:
        return self.model


class StateDictLoader(ModelLoader):
    """ PyTorch Dictionary State loader """

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

        self._load_model()

    def _load_model(self) -> None:
        """ load a model using torch.load function """

        self.model = getattr(
            import_module("src.model"),
            self.model_name,
        )(self.model_arc, self.num_classes)

        self.model.load_state_dict(torch.load(self.model_path))        
        self.model.eval()
    
    def get_model(self) -> torch.nn.Module:
        return self.model


class HuggingfaceLoader(ModelLoader):
    """ """

    def __init__(self) -> None:
        super().__init__()
    