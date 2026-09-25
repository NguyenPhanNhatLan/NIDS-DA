import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    def __init__(self, input_dim):
        super(BaselineMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256,168)
        self.bn2 = nn.BatchNorm1d(168)
        
        self.dropout = nn.Dropout(0.3)
        
        self.classifier = nn.Sequential(
            nn.Linear(168,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.bn1(x)
       
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        features = self.bn2(x)
        features = F.relu(features)
        
        logits = self.classifier(features)
        return features, logits
    
    