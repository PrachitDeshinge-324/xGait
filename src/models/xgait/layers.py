"""
Custom layers for XGait model
"""

import torch
import torch.nn as nn


class SetBlockWrapper(nn.Module):
    """Wrapper for handling set/sequence data"""
    
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block
        
    def forward(self, x, *args, **kwargs):
        """
        x: [N, C, S, H, W] where S is sequence length
        """
        if len(x.size()) == 4:
            return self.forward_block(x, *args, **kwargs)
            
        N, C, S, H, W = x.size()
        x = x.view(N*S, C, H, W)
        output = self.forward_block(x, *args, **kwargs)
        
        # Reshape back to sequence format
        _, new_C, new_H, new_W = output.size()
        output = output.view(N, new_C, S, new_H, new_W)
        return output


class SeparateFCs(nn.Module):
    """Separate fully connected layers for each part - Official Structure"""
    
    def __init__(self, parts_num, in_channels, out_channels):
        super(SeparateFCs, self).__init__()
        self.parts_num = parts_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Official structure: tensor parameter instead of ModuleList
        self.fc_bin = nn.Parameter(torch.randn(parts_num, in_channels, out_channels))
        
    def forward(self, x):
        """
        x: [N, C, P] where P is parts_num
        """
        N, C, P = x.shape
        outputs = torch.zeros(N, self.out_channels, P, device=x.device)
        
        for p in range(P):
            outputs[:, :, p] = torch.matmul(x[:, :, p], self.fc_bin[p])  # [N, out_channels]
            
        return outputs  # [N, out_channels, P]


class SeparateBNNecks(nn.Module):
    """Separate Batch Normalization necks for each part - Official Structure"""
    
    def __init__(self, class_num, in_channels, parts_num):
        super(SeparateBNNecks, self).__init__()
        self.parts_num = parts_num
        self.in_channels = in_channels
        self.class_num = class_num
        
        # Official structure: unified BN and FC
        self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        self.fc_bin = nn.Parameter(torch.randn(parts_num, in_channels, class_num))
        
    def forward(self, x):
        """
        x: [N, C, P]
        returns: (features, logits)
        """
        N, C, P = x.shape
        
        # Flatten for unified BN: [N, C*P]
        x_flat = x.view(N, C * P)
        features_flat = self.bn1d(x_flat)
        
        # Reshape back: [N, C, P]
        features = features_flat.view(N, C, P)
        
        # Compute logits using fc_bin: [P, C, class_num]
        logits = torch.zeros(N, self.class_num, P, device=x.device)
        for p in range(P):
            logits[:, :, p] = torch.matmul(features[:, :, p], self.fc_bin[p])  # [N, class_num]
        
        return features, logits
