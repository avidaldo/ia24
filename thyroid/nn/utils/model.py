from torch import nn

class ThyroidNet(nn.Module):
    
    def __init__(self, input_size, num_classes=3, dropout_rate=0.3):
        super(ThyroidNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Capa 1
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Capa 2
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Capa 3
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Capa de salida (sin activación, se aplica en la función de pérdida)
        x = self.fc4(x)
        
        return x
