import torch
from torchvision import models


def get_model(output_size):
    model = models.densenet121(pretrained=True)

    n_feats = model.classifier.in_features
    model.classifier = torch.nn.Linear(n_feats, output_size)

    return model
