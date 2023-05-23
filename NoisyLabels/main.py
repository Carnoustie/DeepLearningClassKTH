# Library imports
from matplotlib import pyplot as plt
import torch
import torch.optim as optim

# Custom module imports
from loss import *
from util import *
from VGG import *

# Parameters
eta = 0.01
momentum = 0.9
weight_decay = 5e-4
gamma = 0.1
milestones = [40, 80]
alpha = 0.1
beta = 1
nbr_classes = 10
nbr_epochs = 10

# Set up device and net
net = VGG()
net = net.to(net.device)

# Initialize Pytorch learning functionality
optimizer = optim.SGD(net.parameters(), lr=eta,
                      momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

# Select criterion
criterion = SCELoss(alpha=alpha, beta=beta, num_classes=nbr_classes)
#criterion = nn.CrossEntropyLoss()

# Train network
loss_list = []
loss_list = net.fit(optimizer, scheduler, criterion, trainloader, testloader, nbr_epochs, loss_list)

# Plot results
plt.plot(loss_list)
plt.xlabel("Epochs")
plt.ylabel("Average Training Loss")
plt.show()