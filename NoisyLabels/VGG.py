import torch.nn as nn
import tqdm
import torch

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self._make_layers([32, 'MaxPool', 32, 'MaxPool', 64, 64, 'MaxPool', 128, 128, 'MaxPool'])
        self.classifier = nn.Linear(512, 10)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, model):
        seq_layers = []
        channels = 3
        for layer in model:
            if layer == 'MaxPool':
                seq_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                seq_layers += [nn.Conv2d(channels, layer, kernel_size=3, padding=1),
                           nn.BatchNorm2d(layer),
                           nn.ReLU(inplace=True)]
                channels = layer
        seq_layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*seq_layers)
    
    def fit(self, optimizer, scheduler, criterion, trainloader, testloader, nbr_epochs, loss_list):
        for epoch in tqdm.tqdm(range(nbr_epochs)):
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (images, labels) in enumerate((trainloader)):
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.forward(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            loss_list.append(train_loss/len(trainloader))
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate((testloader)):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.forward(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            acc = correct/total
            print("\nAccuracy is " + str(acc))
            scheduler.step()
        return loss_list