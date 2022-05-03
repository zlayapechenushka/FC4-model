import torch
import torchvision.models as models
import math

from torch import nn
from torch.nn.functional import normalize as norma
class FC4(torch.nn.Module):

    def __init__(self):
        super().__init__()

        #Alexnet
        alexnet = models.alexnet(pretrained=True)
        self.backbone = nn.Sequential(*list(alexnet.children())[0])

        #Additional layers
        self.final_convs = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
            nn.Conv2d(256, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 4, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, image):
        image = self.backbone(image)
        out = self.final_convs(image)
 
        # Per-patch color estimates (first 3 dimensions)
        rgb = norma(out[:, :3, :, :], dim=1)

        # Confidence (last dimension)
        confidence = out[:, 3:4, :, :]

        # Confidence-weighted pooling
        pred = norma(torch.sum(torch.sum(rgb * confidence, 2), 2), dim=1)

        return pred, rgb, confidence


class ModelFC4:

    def __init__(self):
        self._device = "cuda:0"
        self._optimizer = None
        self._network = FC4().to(self._device)

    def predict(self, image):
        """
        Performs inference on the input image using the FC4 method.
        @param image: the image for which an illuminant colour has to be estimated
        @param return_steps: whether or not to also return the per-patch estimates and confidence weights. When this
        flag is set to True, confidence-weighted pooling must be active)
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """

        pred, rgb, confidence = self._network(image)
        return pred

    def optimize(self, image, true):
        self._optimizer.zero_grad()
        pred = self.predict(image)
        loss = self.get_loss(pred, true)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def get_loss(self, pred, true, safe_v = 0.999999):
        dot = torch.clamp(torch.sum(norma(pred, dim=1) * norma(true, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle).to(self._device)

    def train_mode(self):
        self._network = self._network.train()

    def evaluation_mode(self):
        self._network = self._network.eval()

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "adam"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}
        self._optimizer = optimizers_map[optimizer_type](self._network.parameters(), lr=learning_rate)