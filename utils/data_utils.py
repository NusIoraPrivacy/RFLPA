import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import random

def iid_partition(dataset, clients):
    """
    I.I.D paritioning of data over clients
    Shuffle the data
    Split it between clients

    params:
      - dataset (torch.utils.Dataset): Dataset containing the Images
      - clients (int): Number of Clients to split the data between

    returns:
      - Dictionary of image indexes for each client
    """

    num_items_per_client = int(len(dataset) / clients)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(clients):
        client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])

    return client_dict

def non_iid_partition(dataset, n_nets, alpha):
    """
        :param dataset: dataset name
        :param n_nets: number of clients
        :param alpha: beta parameter of the Dirichlet distribution
        :return: dictionary containing the indexes for each client
    """
    y_train = np.array(dataset.targets)
    min_size = 0
    K = 10
    N = y_train.shape[0]
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            # print(len(proportions))
            # print(sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])
    # print(len(net_dataidx_map))
    # print(net_dataidx_map[0].shape)
    return net_dataidx_map

class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def get_dataset(data_root, args):
    if args.dataset == "Cifar":
        stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        transforms_cifar_train = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.Normalize(*stats)])
        transforms_cifar_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(*stats)])

        train_data = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms_cifar_train)
        test_data = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transforms_cifar_test)
    
    elif args.dataset == "Cifar100":
        stats = ((0.50707515, 0.48654887, 0.44091784), (0.26733428, 0.256438462, 0.2761504713))

        transforms_cifar_train = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.Normalize(*stats)])
        transforms_cifar_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(*stats)])

        train_data = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms_cifar_train)
        test_data = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transforms_cifar_test)

    elif args.dataset == "Mnist":
        train_data = datasets.MNIST(
            root = data_root, train = True,  transform = transforms.ToTensor(),  download = True,            
        )
        test_data = datasets.MNIST(
            root = data_root, train = False,  transform = transforms.ToTensor()
        )
    
    elif args.dataset == "Fashion_mnist":
        transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = datasets.FashionMNIST(
            root = data_root, train = True,  transform = transform,  download = True,            
        )
        test_data = datasets.FashionMNIST(
            root = data_root,  train = False,  transform = transform,  download = True, 
        )
    return train_data, test_data

def add_trigger(data, add_all=True, args=None):
    if args.attack_type == "badnet":
        d_size, channels, width, height = data.shape
        trig_size = d_size if add_all else 1
        for idx in range(trig_size):
            for c in range(channels):
                data[idx, c, width-3, height-3] = 255
                data[idx, c, width-3, height-2] = 255
                data[idx, c, width-2, height-3] = 255
                data[idx, c, width-2, height-2] = 255
    
    elif args.attack_type == "scaling":
        d_size, channels, width, height = data.shape
        trig_size = d_size if add_all else 1
        for idx in range(trig_size):
            x_rand = random.randrange(-2,20)
            y_rand = random.randrange(-23, 2)
            data[idx][0][ x_rand + 2][ y_rand + 25] = 2.5 + (random.random()-0.5)
            data[idx][0][ x_rand + 2][ y_rand + 24] = 2.5 + (random.random()-0.5)
            data[idx][0][ x_rand + 2][ y_rand + 23] = 2.5 + (random.random()-0.5)

            data[idx][0][ x_rand + 6][ y_rand + 25] = 2.5 + (random.random()-0.5)
            data[idx][0][ x_rand + 6][ y_rand + 24] = 2.5 + (random.random()-0.5)
            data[idx][0][ x_rand + 6][ y_rand + 23] = 2.5 + (random.random()-0.5)

            data[idx][0][ x_rand + 5][ y_rand + 24] = 2.5 + (random.random()-0.5)
            data[idx][0][ x_rand + 4][ y_rand + 23] = 2.5 + (random.random()-0.5)
            data[idx][0][ x_rand + 3][ y_rand + 24] = 2.5 + (random.random()-0.5)

    return data