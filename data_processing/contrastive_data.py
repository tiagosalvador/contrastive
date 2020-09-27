import math
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class UnlabeledDataset(Dataset):
    '''simple class that takes a dataset and strips its labels'''

    def __init__(self, input_dataset):
        self.input_dataset = input_dataset

    def __len__(self):
        return len(self.input_dataset)

    def __getitem__(self, idx):
        item, label = self.input_dataset.__getitem__(idx)
        return item


class ContrastiveData:
    '''Takes care of data for contrastive purposes'''

    def __init__(self, args, dataset_name="MNIST", **kwargs):
        # Import train data
        self.args = args
        self.kwargs = kwargs
        train_data = None
        test_data = None
        if dataset_name == "MNIST":
            train_data = datasets.MNIST(args.data_dir, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                            # Where do these numbers come from?
                                        ]))
            test_data = datasets.MNIST(args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        else:
            raise ValueError("Dataset name is not supported")

        # split into labeled and unlabled training sets
        labeled_train_data, unlabeled_train_data = torch.utils.data.random_split(train_data, [
            math.floor(args.frac_labeled * len(train_data)), math.floor((1 - args.frac_labeled) * len(train_data))])
        self.labeled_train_data = labeled_train_data
        self.unlabeled_train_data = UnlabeledDataset(unlabeled_train_data)
        self.test_data = test_data

    def get_data_loaders(self):
        '''Get data loaders'''

        labeled_loader = torch.utils.data.DataLoader(self.labeled_train_data, batch_size=self.args.batch_size, shuffle=True,
                                                     **self.kwargs)
        unlabeled_loader = torch.utils.data.DataLoader(self.unlabeled_train_data, batch_size=self.args.batch_size,
                                                       shuffle=True,
                                                       **self.kwargs)
        test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=1000, shuffle=True, **self.kwargs)
        return {'labeled': labeled_loader, 'unlabeled': unlabeled_loader, 'test': test_loader}
