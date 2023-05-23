import torch
import torchvision
import torchvision.transforms as transforms
import random

def contaminate_data(dataset, rate):
    for i in range(int(rate*len(dataset.targets))):
        current = dataset.targets[i]
        other = list(range(1,current)) + list(range(current+1, 10))
        dataset.targets[i] = random.choice(other)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

trainset_corrupt = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

contaminate_data(trainset_corrupt, 0.4)

trainloader_corrupt = torch.utils.data.DataLoader(
    trainset_corrupt, batch_size=100, shuffle=True, num_workers=0)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

