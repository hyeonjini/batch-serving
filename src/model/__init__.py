import torch
import torch.nn as nn
from importlib import import_module


def resnet(name:str, num_classes:int) -> torch.nn.Module:
    model = getattr(
        import_module("torchvision.models"),
        name,
    )()

    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes,
        bias=True
    )
    return model

def densenet(name:str, num_classes: int) -> torch.nn.Module:
    
    model = getattr(
        import_module("torchvision.models"),
        name,
    )()

    num_features = model.features[-1].num_features
    model.classifier = torch.nn.Linear(
        in_features=num_features,
        out_features=num_classes,
        bias=True
    )
    return model


class DenseNet(nn.Module):
    
    def __init__(
        self,
        name: str = "densenet121",
        num_classes: int = 10,
    ) -> None:

        nn.Module.__init__(self)
        self.net = getattr(
            import_module("torchvision.models"),
            name,
        )()

        num_features = self.net.features[-1].num_features
        self.net.classifier = nn.Linear(in_features=num_features, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    def __init__(
        self,
        name: str = "resnet18",
        num_classes: int = 10,
    ) -> None:

        nn.Module.__init__(self)
        self.net = getattr(
            import_module("torchvision.models"),
            name,
        )()

        self.net.fc = nn.Linear(
            in_features=self.net.fc.in_features,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, x):
        return self.net(x)

