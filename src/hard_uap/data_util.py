import torch
import torchvision
import torchvision.transforms as transforms

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device, transform = None):
        self.dl = dl
        self.device = device
        self.transform = transform
    def __iter__(self):
        if self.transform is None:
            for b in self.dl:
                yield to_device(b, self.device)
        else:
            for b in self.dl:
                a = to_device(b, self.device)
                yield [self.transform(a[0]), a[0], a[1]]
    def __len__(self):
        return len(self.dl)

def load_cifar(dataset_root, val_size = 5000):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    cifar10_train = torchvision.datasets.CIFAR10(root=dataset_root, train=True, transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root=dataset_root, train=False, transform=transform)
    train_size = len(cifar10_train) - val_size
    cifar10_train, cifar10_val = torch.utils.data.random_split(cifar10_train, [train_size, val_size])
    return cifar10_train, cifar10_val, cifar10_test

def toDeviceDataLoader(*args, device = torch.device('cuda:0'), batch_size = 16, transform = None):
    dls = [torch.utils.data.DataLoader(d, batch_size = batch_size, shuffle = True) for d in args]
    return [DeviceDataLoader(d, device = device, transform = transform) for d in dls]

def cifar_class_idx():
    return {i:n for i,n in enumerate(['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])}

def load_mnist(batch_size_train, batch_size_test, device, dataset_path="/share/datasets/mnist/"):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(dataset_path, download=True, train=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(dataset_path, download=True, train=False, transform=transform)
    train_loader = toDeviceDataLoader(train_dataset, device = device, batch_size = batch_size_train)[0]
    test_loader = toDeviceDataLoader(test_dataset, device = device, batch_size = batch_size_test)[0]
    return train_loader, test_loader