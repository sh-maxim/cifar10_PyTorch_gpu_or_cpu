#!/usr/bin/env python

import torch
import torchvision
import torchvision.transforms as transforms

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Press Ctrl+F8 to toggle the breakpoint.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#train nn
import torch.optim as optim
import time
def train_nn_model(_train_on_device, _trainloader, _batch_size, _nu_channels):
    print(f"> Attempting to train on '{_train_on_device}'.")

    if _train_on_device == 'gpu':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"> Using device of '{device}'")
    myNet = define_nn(_nu_channels)
    net = myNet()
    if _train_on_device == 'gpu':
        net.to(device)

    #defining loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #train NN
    tenth_iteration = trainset.data.shape[0] / _batch_size / 10
    start = time.time()
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if _train_on_device == 'gpu':
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print 10 times during training
            if i % tenth_iteration ==  tenth_iteration - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / tenth_iteration))
                running_loss = 0.0

    end = time.time()
    print('GPU Finished Training')
    print("%.3f seconds passed" % (end - start))

    #save NN to hdd
    PATH = './cifar10_gpu_net.pth'
    torch.save(net.state_dict(), PATH)

# functions to show an image
import matplotlib.pyplot as plt
import numpy as np
def imshow(img):
    # unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define CNN
import torch.nn as nn
import torch.nn.functional as F
def define_nn(_nu_channels):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            #number of output channels was 6
            self.conv1 = nn.Conv2d(3, _nu_channels, 5)
            self.pool = nn.MaxPool2d(2, 2)
            #number of input channels was 6
            self.conv2 = nn.Conv2d(_nu_channels, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    return Net

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import sys
    print(f"> Executing on {sys.executable}.")

    #{'gpu','cpu'}
    train_on_device = 'cpu'
    # {4, 40}
    batch_size = 4
    #{6, 60, 600, 6000}
    nu_channels = 6
    show_example_image = False

    # data loading
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/home/max/main/gpu/cifar10/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    if show_example_image:
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
        # show images
        imshow(torchvision.utils.make_grid(images))

    train_nn_model(train_on_device, trainset, batch_size, nu_channels)
