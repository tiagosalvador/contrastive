import torch
import argparse
import matplotlib.pyplot as plot
from data_processing.contrastive_data import ContrastiveData
import torch.nn as nn

# Parse arguments
parser = argparse.ArgumentParser(description='Constrastive Learning Experiment')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--dropout', type=float, default=0.25, metavar='P',
                    help='dropout probability (default: 0.25)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='heavy ball momentum in gradient descent (default: 0.9)')
parser.add_argument('--frac-labeled', type=float, default=0.01, metavar='FL',
                    help='Fraction of labeled data (default 0.01))')
parser.add_argument('--data-dir', type=str, default='./data',metavar='DIR')
args = parser.parse_args()
args.cuda =  torch.cuda.is_available()

# Print out arguments to the log
print('Constrastive Learning Run')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

### Let's define the simplest network I can
class SimpleNet(nn.Module):

    def __init__(self,num_clusters):
        super(SimpleNet,self).__init__()
        self.num_clusters = num_clusters

    def forward(self,x): # No activation function, just one map of weights
        net = nn.Sequential(
            nn.Linear(2*num_clusters,2*num_clusters)
        )

        return net(x)

def SemiSupervisedLoss(labeled_estimate,target,unlabled_input = None,unlabled_centers = None):
    return None

num_clusters = 2
data = ContrastiveData(args.batch_size,args.frac_labeled,args.data_dir,dataset_name = 'Projection',num_clusters = num_clusters, **kwargs)
data_loaders = data.get_data_loaders()
