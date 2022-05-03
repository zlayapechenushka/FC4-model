import numpy as np
import torchvision.transforms.functional as F
import torch

from torch.nn.functional import interpolate
from torch import Tensor

def normalize(image):
    return np.clip(image, 0.0, 65535.0) * (1.0 / 65535.0)

def bgr_to_rgb(image):
    return image[:, :, ::-1]

def linear_to_nonlinear(image):
    if isinstance(image, np.ndarray):
        return np.power(image, (1.0 / 2.2))
    if isinstance(image, Tensor):
        return torch.pow(image, 1.0 / 2.2)
    return F.to_pil_image(torch.pow(F.to_tensor(image), 1.0 / 2.2).squeeze(), mode="RGB")

def hwc_to_chw(image):
    return image.transpose(2, 0, 1)

def scale(image):
    image = image - image.min()
    image = image / image.max()
    return image

def rescale(image, size):
    return interpolate(image, size, mode='bilinear')

def correct(image,illuminant):
    image = F.to_tensor(image).to(DEVICE)

    #Correcting image
    correction = illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(Tensor([3])).to(DEVICE)
    corrected_img = torch.div(image, correction + 1e-10)

    #Normalization
    max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    normalized_img = torch.div(corrected_img, max_img)

    return F.to_pil_image(linear_to_nonlinear(normalized_img).squeeze(), mode="RGB")

def percentile(errors, procents):
    return np.percentile(errors, procents * 100)

def compute_metrics(errors):
    errors = sorted(errors)
    metrics = {
        "mean": np.mean(errors),
        "median": percentile(errors, 0.5),
        "trimean": 0.25 * (percentile(errors, 0.25) + 2 * percentile(errors, 0.5) + percentile(errors, 0.75)),
        "bst25": np.mean(errors[:int(0.25 * len(errors))]),
        "wst25": np.mean(errors[int(0.75 * len(errors)):]),
        "wst5": percentile(errors, 0.95)}
    return metrics

def print_metrics(current_metrics):
    print(" Mean ......... : {:.4f} ".format(current_metrics["mean"]))
    print(" Median ....... : {:.4f} ".format(current_metrics["median"]))
    print(" Trimean ...... : {:.4f} ".format(current_metrics["trimean"]))
    print(" Best 25% ..... : {:.4f} ".format(current_metrics["bst25"]))
    print(" Worst 25% .... : {:.4f} ".format(current_metrics["wst25"]))
    print(" Worst 5% ..... : {:.4f} ".format(current_metrics["wst5"]))

def normalize(image):
    max_int = 65535.0
    return np.clip(image, 0.0, max_int) * (1.0 / max_int)