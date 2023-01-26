from abc import ABC, abstractmethod
from typing import Optional
import torch
from importlib import import_module

class ModelLoader(ABC):
    """Represent a ModelLoader class."""
    
    def __init__(
        self,
        model_path: str,
    ) -> None: 

        self.model_path: str = model_path

    @abstractmethod
    def _load_model(self):
        return NotImplementedError()
    
    @abstractmethod
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


class HuggingfacePreTrainedModelLoader(ModelLoader):
    """ Model loader in Huggingface """

    def __init__(
        self,
        model_name:str,
        model_path:str,
    ) -> None:

        super().__init__(model_path)

        self.model_name = model_name

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        self.model = getattr(
            import_module("transformers"),
            self.model_name,
        ).from_pretrained(
            self.model_path
        )
        self.model.eval()
    
    def get_model(self) -> torch.nn.Module:
        return self.model
    
    def get_id2label(self) -> Optional[dict]:
        if self.model.config.id2label:
            return self.model.config.id2label

        return None

    def get_label2id(self) -> Optional[dict]:
        if self.model.config.label2id:
            return self.model.config.label2id
        
        return None
    