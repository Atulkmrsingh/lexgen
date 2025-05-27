import torch
# import torch.onnx.operators
# from fairseq import utils
from torch import  nn


class DomainGatingNetwork(nn.Module):
    def __init__(self, dec_hid_dim, mid_dim, out_feats=1):
        super().__init__()
        self.linear = self.fc1_linear(dec_hid_dim, mid_dim)
        self.linear2 = self.fc1_linear(mid_dim, out_feats)
        self.relu = nn.ReLU()

    def fc1_linear(self, in_features, out_features, bias=True):
            m = nn.Linear(in_features, out_features, bias)
            nn.init.xavier_uniform_(m.weight)
            if bias:
                nn.init.constant_(m.bias, 0.)
            return m

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        
        x= self.linear2(x)
        # if self.training:
        #     noise = torch.randn_like(x) * 0.001
        #     x = x + noise
        x = torch.sigmoid(x)
        return x
