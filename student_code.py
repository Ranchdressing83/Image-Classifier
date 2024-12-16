import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        
        # Define the convolutional and fully connected layers as specified
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # Dictionary to store intermediate shapes
        shape_dict = {}
        
        # Layer 1: Convolution + ReLU + Max Pooling
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        shape_dict[1] = list(x.size())
        
        # Layer 2: Convolution + ReLU + Max Pooling
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        shape_dict[2] = list(x.size())
        
        # Flatten layer
        x = x.view(x.size(0), -1)
        shape_dict[3] = list(x.size())
        
        # Fully connected layers with ReLU activations
        x = torch.relu(self.fc1(x))
        shape_dict[4] = list(x.size())
        
        x = torch.relu(self.fc2(x))
        shape_dict[5] = list(x.size())
        
        out = self.fc3(x)
        shape_dict[6] = list(out.size())
        
        return out, shape_dict

def count_model_params():
    model = LeNet()  # Create an instance of LeNet
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
