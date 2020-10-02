import math
import torch
import os
import os.path
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ContrastiveData:
    '''Takes care of data for contrastive purposes'''

    def __init__(self, batch_size,fraction_labeled,data_directory, dataset_name="MNIST",num_clusters=5, **kwargs):
        # Import train data
        self.batch_size = batch_size
        self.fraction_labeled = fraction_labeled
        self.data_directory = data_directory
        self.kwargs = kwargs
        self.num_clusters = num_clusters
        train_data = None
        test_data = None
        if dataset_name == "MNIST":
            train_data = datasets.MNIST(self.data_directory, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))            
            test_data = datasets.MNIST(self.data_directory, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        elif dataset_name == "Projection":
            train_data = ProjectionData(self.data_directory,train = True,num_clusters=self.num_clusters)
            test_data = ProjectionData(self.data_directory,train = False,num_clusters=self.num_clusters)
        else:
            raise ValueError("Dataset name is not supported")

        # split into labeled and unlabled training sets
        labeled_train_data, unlabeled_train_data = torch.utils.data.random_split(train_data, [
            math.floor(self.fraction_labeled * len(train_data)), math.floor((1 - self.fraction_labeled) * len(train_data))])
        self.labeled_train_data = labeled_train_data
        self.unlabeled_train_data = UnlabeledDataset(unlabeled_train_data)
        self.test_data = test_data

    def get_data_loaders(self):
        '''Get data loaders'''

        labeled_loader = torch.utils.data.DataLoader(self.labeled_train_data, batch_size=self.batch_size, shuffle=True,
                                                     **self.kwargs)
        unlabeled_loader = torch.utils.data.DataLoader(self.unlabeled_train_data, batch_size=self.batch_size,
                                                       shuffle=True,
                                                       **self.kwargs)
        test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=1000, shuffle=True, **self.kwargs)
        return {'labeled': labeled_loader, 'unlabeled': unlabeled_loader, 'test': test_loader}

    def getItemSize(self):
        data,label = self.labeled_train_data.__getitem__(0)
        return data.size()



class UnlabeledDataset(Dataset):
    '''simple class that takes a dataset and strips its labels'''

    def __init__(self, input_dataset: Dataset):
        self.input_dataset = input_dataset

    def __len__(self):
        return len(self.input_dataset)

    def __getitem__(self, idx:int):
        item, label = self.input_dataset.__getitem__(idx)
        return item

class ProjectionData(Dataset):
    ''' Toy dataset with data seperated into [x,y] where x is perfectly clustered and y is noise
        Projection refers to optimal map to learn, which is a projection onto x
    '''
    def __init__(self,root:str, train: bool = True,num_clusters: int = 5):
        self.train = train
        self.root = root
        self.num_clusters = num_clusters
        self.datafolder = os.path.join(self.root,'projdata')
        self.training_file = 'training_k{num_clusters}.pt'.format(num_clusters = self.num_clusters)
        self.test_file = 'test_k{num_clusters}.pt'.format(num_clusters= self.num_clusters)

        if not self._check_exists():
            print('Dataset not found: Generating new data')
            self.generate()


        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.labels = torch.load(os.path.join(self.datafolder, data_file))

    def __getitem__(self,idx:int):
        item,label = self.data[idx],self.labels[idx]
        return item,label

    def __len__(self):
        return len(self.data)

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.datafolder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.datafolder,
                                            self.test_file)))


    def generate(self):
        ''' generates 60000 training examples and 10000 testing examples with labels'''
        # TODO Generate labels
        k = self.num_clusters
        path = os.path.join(self.datafolder)
        os.makedirs(path, exist_ok=True)

        samples_per_category = 70000//k
        eyes = torch.eye(k)

        data = []
        labels = []
        for i in range(k): # Generate vectors that are half standard basis half random noise
            ei = eyes[:,i].repeat(samples_per_category,1)
            noise = torch.randn([samples_per_category,k])

            one_label = torch.cat((eyes[:,i],torch.zeros(k)))
            labels.append(one_label.repeat(samples_per_category,1))
            data.append(torch.cat((ei,noise),1))


        data = torch.cat(data)
        labels = torch.cat(labels)
        training_data,test_data = torch.split(data,[60000,len(data) - 60000])
        training_labels,test_labels = torch.split(labels,[60000,len(data) - 60000])

        with open(os.path.join(self.datafolder, self.training_file), 'wb') as f:
            torch.save((training_data,training_labels), f)
        with open(os.path.join(self.datafolder, self.test_file), 'wb') as f:
            torch.save((test_data,test_labels), f)
