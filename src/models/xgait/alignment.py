"""
Cross-granularity alignment modules for XGait model
"""

import torch
import torch.nn as nn


class CALayers(nn.Module):
    """Global Cross-granularity Alignment Module"""
    
    def __init__(self, channels, reduction):
        super(CALayers, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, sil_feat, par_feat):
        """
        Global cross-granularity alignment
        """
        N, C, S, H, W = sil_feat.shape
        
        # Flatten for processing
        sil_flat = sil_feat.contiguous().view(N*S, C, H, W)
        par_flat = par_feat.contiguous().view(N*S, C, H, W)
        
        # Concatenate silhouette and parsing features
        concat_feat = torch.cat([sil_flat, par_flat], dim=1)  # [N*S, 2*C, H, W]
        
        # Generate attention from concatenated features
        attention = self.avg_pool(concat_feat).view(N*S, 2*C)  # [N*S, 2*C]
        attention = self.fc(attention).view(N*S, 2*C, 1, 1)   # [N*S, 2*C, 1, 1]
        
        # Apply attention to concatenated features
        aligned_feat = concat_feat * attention
        
        # Split back to individual streams and combine
        sil_aligned, par_aligned = torch.chunk(aligned_feat, 2, dim=1)
        combined_feat = sil_aligned + par_aligned  # [N*S, C, H, W]
        
        # Reshape back
        combined_feat = combined_feat.view(N, C, S, H, W)
        
        return combined_feat


class CALayersP(nn.Module):
    """Part-based Cross-granularity Alignment Module"""
    
    def __init__(self, channels, reduction, choosed_part='up', with_max_pool=True):
        super(CALayersP, self).__init__()
        self.choosed_part = choosed_part
        self.with_max_pool = with_max_pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if with_max_pool:
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, sil_feat, par_feat, mask_resize):
        """
        Part-based cross-granularity alignment
        """
        N, C, S, H, W = sil_feat.shape
        
        # Determine part region
        if self.choosed_part == 'up':
            part_h = H // 4
            sil_part = sil_feat[:, :, :, :part_h, :]
            par_part = par_feat[:, :, :, :part_h, :]
        elif self.choosed_part == 'middle':
            start_h, end_h = H // 4, 3 * H // 4
            sil_part = sil_feat[:, :, :, start_h:end_h, :]
            par_part = par_feat[:, :, :, start_h:end_h, :]
        else:  # down
            part_h = 3 * H // 4
            sil_part = sil_feat[:, :, :, part_h:, :]
            par_part = par_feat[:, :, :, part_h:, :]
        
        # Flatten for processing
        sil_flat = sil_part.contiguous().view(N*S, C, -1, W)
        par_flat = par_part.contiguous().view(N*S, C, -1, W)
        
        # Concatenate silhouette and parsing features
        concat_feat = torch.cat([sil_flat, par_flat], dim=1)  # [N*S, 2*C, part_h, W]
        
        # Generate attention from concatenated features
        if self.with_max_pool:
            attention_avg = self.avg_pool(concat_feat).view(N*S, 2*C)
            attention_max = self.max_pool(concat_feat).view(N*S, 2*C)
            attention = self.fc(attention_avg + attention_max).view(N*S, 2*C, 1, 1)
        else:
            attention = self.avg_pool(concat_feat).view(N*S, 2*C)
            attention = self.fc(attention).view(N*S, 2*C, 1, 1)
        
        # Apply attention to concatenated features
        aligned_feat = concat_feat * attention
        
        # Split back to individual streams and combine
        sil_aligned, par_aligned = torch.chunk(aligned_feat, 2, dim=1)
        aligned_part = sil_aligned + par_aligned  # [N*S, C, part_h, W]
        
        # Reshape back to part dimensions
        _, _, _, part_h, part_w = sil_part.shape
        aligned_part = aligned_part.view(N, C, S, part_h, part_w)
        
        return aligned_part
