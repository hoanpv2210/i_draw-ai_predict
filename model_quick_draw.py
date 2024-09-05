import torch
import torch.nn as nn

class quick_draw_CNN(nn.Module):
    def __init__(self, num_classes = 17):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels =4, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels =8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(in_features=392, out_features=128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(in_features=128, out_features=56),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features = 56, out_features = num_classes)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
if __name__ == '__main__':
    model = quick_draw_CNN()
    fake_data = torch.rand(16,1,28,28)
    output= model(fake_data)
    print(output.shape)