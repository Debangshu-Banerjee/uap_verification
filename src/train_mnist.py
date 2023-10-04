import torch
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

torch.manual_seed(123)
random.seed(123)
np.random.seed(123)

class MNet(torch.nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        self.l1 = torch.nn.Linear(28*28, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l2 = torch.nn.Linear(256, 2)
    
    def forward(self, x):
        out = torch.relu(self.l1(x))
        return self.l2(out)

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

def toDeviceDataLoader(*args, device = torch.device('cuda:0'), batch_size = 16, transform = None):
    dls = [torch.utils.data.DataLoader(d, batch_size = batch_size, shuffle = True, drop_last = True) for d in args]
    return [DeviceDataLoader(d, device = device, transform = transform) for d in dls]

def load_mnist(batch_size_train, batch_size_test, device, dataset_path="/share/datasets/mnist/"):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(dataset_path, download=True, train=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(dataset_path, download=True, train=False, transform=transform)
    train_loader = toDeviceDataLoader(train_dataset, device = device, batch_size = batch_size_train)[0]
    test_loader = toDeviceDataLoader(test_dataset, device = device, batch_size = batch_size_test)[0]
    return train_loader, test_loader

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def dsa_01(ds, k, verbose = False):
    tot = 0
    acc = 0
    for batch in ds:
        idx_01 = torch.nonzero((batch[1] == 0) + (batch[1] == 1))
        if idx_01.shape[0] == 0:
            continue
        batch[0] = batch[0][idx_01.flatten()]
        batch[1] = batch[1][idx_01.flatten()]
        out = k(batch[0].view(-1, 28*28))
        acc += accuracy(out, batch[1]) * len(batch[1])
        tot += len(batch[1])
    if verbose:
        print(acc, tot)
    return (acc/tot).item()

train_loader, test_loader = load_mnist(64, 10, device)

# mdl = to_device(MNet(), device)
# mdl.load_state_dict(torch.load('mnist_01.pth'))
# mdl = mdl.eval()

mdl = to_device(MNet(), device)
crit = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mdl.parameters(), lr=0.01) 
for epoch in range(2):
    pbar = tqdm(train_loader)
    for batch_id, (images, labels) in enumerate(pbar):
        idx_01 = torch.nonzero((labels == 0) + (labels == 1))
        if idx_01.shape[0] == 0:
            continue
        images = images[idx_01.flatten()]
        labels = labels[idx_01.flatten()]
        outputs = mdl(images.view(-1, 28*28))
        loss = F.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if batch_id == len(train_loader) - 1:
            pbar.set_postfix({"Test Accuracy":dsa_01(test_loader, mdl)})
torch.save(mdl.state_dict(), "nets/mnist_01.pth")
print(f"Test Accuracy: {dsa_01(test_loader, mdl, True)}")
print(f"Train Accuracy: {dsa_01(train_loader, mdl, True)}")

import torch.onnx

x = torch.randn(1, 28* 28, requires_grad=True, device = device)
torch_out = mdl(x)

# Export the model
torch.onnx.export(mdl,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "nets/binary.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})