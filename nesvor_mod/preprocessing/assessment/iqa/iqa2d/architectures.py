from torch import nn
import torch
import torchvision


"""pretrained models"""


class DualFC(nn.Module):
    def __init__(
        self, num_ftrs: int, num_classes: int, p: float = 0, is_dual: bool = True
    ) -> None:
        super().__init__()
        self.is_dual = is_dual
        self.fc1 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor):
        x1 = self.fc1(x)
        if self.is_dual:
            return x1, x
        else:
            return x1


class PretrainedModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, p: float = 0) -> None:
        super().__init__()
        self.model = getattr(torchvision.models, model_name)()
        if model_name.startswith("vgg"):
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = DualFC(num_ftrs, num_classes, p)
        elif model_name.startswith("res"):
            num_ftrs = self.model.fc.in_features
            self.model.fc = DualFC(num_ftrs, num_classes, p)
            # self.model.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.expand(-1, 3, -1, -1)
        x = self.model(x)
        return x


def resnet34(**kwargs) -> PretrainedModel:
    return PretrainedModel("resnet34", num_classes=kwargs["num_classes"])
