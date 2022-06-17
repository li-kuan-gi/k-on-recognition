from pre_process import get_transforms
from custom_datasets import ClassifiedDatasets
from model import get_model
from train_and_test import train_model

import torch
from torch.backends import cudnn

cudnn.benchmark = True

train_transform, eval_transform = get_transforms()

k_on_datasets = ClassifiedDatasets('data', train_transform, eval_transform,
                                   validation_split=0.2)
train_loader, val_loader, test_loader = torch.utils.data.DataLoader(
    k_on_datasets.test, batch_size=4, shuffle=True, num_workers=4)

model = get_model(output_size=len(k_on_datasets.classes))

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=7, gamma=0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_model(model, train_loader, val_loader,
            criterion, optimizer, exp_lr_scheduler,
            device=device)

torch.save(model.state_dict(), 'k-on.pth')
