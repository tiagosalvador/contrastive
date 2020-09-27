import torch
import argparse
from torchvision import datasets, transforms
import math
from torch.utils.data import Dataset, DataLoader

# Parse arguments
parser = argparse.ArgumentParser(description='Constrastive Learning Experiment')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--dropout', type=float, default=0.25, metavar='P',
                    help='dropout probability (default: 0.25)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='heavy ball momentum in gradient descent (default: 0.9)')
parser.add_argument('--frac-labeled', type=float, default=0.99, metavar='FL',
                    help='Fraction of labeled data (default 0.99))')
parser.add_argument('--data-dir', type=str, default='./data',metavar='DIR')
args = parser.parse_args()
args.cuda =  torch.cuda.is_available()

# Print out arguments to the log
print('Constrastive Learning Run')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# simple class that takes a dataset and strips its labels
class UnlabeledDataset(Dataset):
    def __init__(self,input_dataset):
        self.input_dataset = input_dataset

    def __len__(self):

        return len(self.input_dataset)

    def __getitem__(self,idx):
        item,label = self.input_dataset.__getitem__(idx)
        return item

# Import train data
train_data = datasets.MNIST(args.data_dir, train=True,download = True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

# split into labeled and unlabled training sets
labeled_train_data,unlabeled_train_data = torch.utils.data.random_split(train_data,[math.floor(args.frac_labeled*len(train_data)),math.floor((1-args.frac_labeled)*len(train_data))])
unlabeled_train_data = UnlabeledDataset(unlabeled_train_data)

# Get data loaders
labeled_loader = torch.utils.data.DataLoader(labeled_train_data,batch_size=1000,shuffle=True,**kwargs)
unlabeled_loader = torch.utils.data.DataLoader(unlabeled_train_data,batch_size=1000,shuffle=True,**kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=True, **kwargs)
