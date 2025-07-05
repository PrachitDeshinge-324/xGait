"""
Pooling modules for XGait model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizontalPoolingPyramid(nn.Module):
    """Horizontal Pooling Pyramid"""
    
    def __init__(self, bin_num=[16]):
        super(HorizontalPoolingPyramid, self).__init__()
        self.bin_num = bin_num
        
    def forward(self, x):
        """
        x: [N, C, H, W]
        output: [N, C, P] where P is sum(bin_num)
        """
        n, c = x.size()[:2]
        features = []
        
        for num_bin in self.bin_num:
            z = F.adaptive_avg_pool2d(x, (num_bin, 1))  # [N, C, num_bin, 1]
            z = z.view(n, c, num_bin)  # [N, C, num_bin]
            features.append(z)
            
        return torch.cat(features, dim=2)  # [N, C, P]


class PackSequenceWrapper(nn.Module):
    """Temporal pooling wrapper"""
    
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func
        
    def forward(self, seqs, seqL, options=None, *args, **kwargs):
        """
        seqs: [N, C, S, H, W]
        seqL: sequence lengths
        """
        if options is None:
            options = {"dim": 2}
        
        dim = options.get('dim', 2)
        
        if self.pooling_func == torch.max:
            out, _ = torch.max(seqs, dim=dim)
        else:
            out = self.pooling_func(seqs, dim=dim)
            
        return [out]
