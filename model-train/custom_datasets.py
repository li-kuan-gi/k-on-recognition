import os

from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import datasets


class DatasetFromLabeledList(Dataset):
    def __init__(self, data, transform=None) -> None:
        super().__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item, label = self.data[index]

        if self.transform:
            item = self.transform(item)

        return item, label


class ClassifiedDatasets:
    def __init__(self, root, train_transform, eval_transform, validation_split):
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.validation_split = validation_split

        self.train_src = os.path.join(root, 'train')
        self.train, self.val = self._split_train_dataset()

        self.test_src = os.path.join(root, 'test')
        self.test = self._get_test_dataset()
        self.classes = self.test.classes

    def _get_test_dataset(self):
        return datasets.ImageFolder(self.test_src, self.eval_transform)

    def _split_train_dataset(self):
        origin_train_dataset = datasets.ImageFolder(self.train_src)
        n_total = len(origin_train_dataset)
        n_val = int(n_total * self.validation_split)
        n_train = n_total - n_val
        train_subset, val_subset = random_split(
            origin_train_dataset, [n_train, n_val])
        return DatasetFromLabeledList(train_subset, self.train_transform),\
            DatasetFromLabeledList(val_subset, self.eval_transform)


def get_dataloaders(calssified_datasets: ClassifiedDatasets):
    return DataLoader(getattr(calssified_datasets, 'train'), batch_size=4,
                      shuffle=True, num_workers=4),\
        DataLoader(getattr(calssified_datasets, 'val'), batch_size=4,
                   shuffle=True, num_workers=4),\
        DataLoader(getattr(calssified_datasets, 'test'),
                   batch_size=4, shuffle=True, num_workers=4)
