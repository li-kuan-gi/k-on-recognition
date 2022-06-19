import torch
from torchvision import models


def get_model(output_size):
    model = models.mobilenet_v3_large(
        pretrained=True, width_mult=1.0,  reduced_tail=False, dilated=False)

    n_feats = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(n_feats, output_size)

    return model
