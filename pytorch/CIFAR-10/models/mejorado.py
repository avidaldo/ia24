import torch.nn as nn
import torch.nn.functional as F

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Primera capa convolucional: entrada 3 canales, salida 32 filtros
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm mejora la estabilidad
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Segunda capa convolucional: entrada 32, salida 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.dropout1 = nn.Dropout(0.25)  # Previene sobreajuste

        # Tercera capa convolucional: entrada 64, salida 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Cuarta capa convolucional: mantiene salida en 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Capa totalmente conectada: input 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout2 = nn.Dropout(0.5)  # Más agresivo para evitar sobreajuste
        self.fc2 = nn.Linear(512, 10)  # CIFAR-10: 10 clases

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 32, 16, 16]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 64, 8, 8]
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))             # [B, 128, 8, 8]
        x = self.pool2(F.relu(self.bn4(self.conv4(x)))) # [B, 128, 4, 4]
        x = x.view(x.size(0), -1)                       # Flatten: [B, 2048]
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x