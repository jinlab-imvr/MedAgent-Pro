import torch.nn as nn
import torch
import timm

class EVA02(nn.Module):
    def __init__(self, model):
        super(EVA02, self).__init__()
        self.model = timm.create_model(model, pretrained=True, num_classes=0) # feture size=768
        self.adapter = nn.Linear(768, 4084)
        self.head_1 = nn.Linear(768, 2)
        self.head_2 = nn.Linear(768, 2)
        self.head_3 = nn.Linear(768, 2)
        self.head_4 = nn.Linear(768, 2)
        self.head_5 = nn.Linear(768, 2)
        self.head_6 = nn.Linear(768, 2)
    def forward(self, x):
        x = self.model(x)
        y = self.adapter(x)
        return y