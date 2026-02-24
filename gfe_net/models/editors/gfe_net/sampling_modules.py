# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

class DownSample(nn.Module):

    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2,
                 method: str = 'conv',
                 kernel_size: int = 3,
                 padding: int = 1,
                 bias: bool = True):
        super(DownSample, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.method = method
        
        if method == 'conv':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            self.pool = nn.MaxPool2d(scale_factor)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
        elif method == 'strided_conv':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                stride=scale_factor, padding=padding, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
        elif method == 'maxpool':
            self.pool = nn.MaxPool2d(scale_factor)
            if in_channels != out_channels:
                self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
            else:
                self.conv = nn.Identity()
                
        elif method == 'avgpool':
            self.pool = nn.AvgPool2d(scale_factor)
            if in_channels != out_channels:
                self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
            else:
                self.conv = nn.Identity()
        else:
            raise ValueError(f"Unsupported downsample method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)
            
        Returns:
            torch.Tensor: 下采样后的张量，形状为 (B, out_channels, H//scale_factor, W//scale_factor)
        """
        if self.method == 'conv':
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.pool(x)
        elif self.method == 'strided_conv':
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        elif self.method in ['maxpool', 'avgpool']:
            x = self.pool(x)
            x = self.conv(x)
            
        return x

class UpSample(nn.Module):
    """上采样模块，支持多种上采样方法
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        scale_factor (int): 上采样倍数，默认为2
        method (str): 上采样方法，支持 'transpose_conv', 'bilinear', 'nearest', 'pixel_shuffle', 'sub_pixel'
        kernel_size (int): 卷积核大小，默认为3
        padding (int): 填充大小，默认为1
        bias (bool): 是否使用偏置，默认为True
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2,
                 method: str = 'transpose_conv',
                 kernel_size: int = 3,
                 padding: int = 1,
                 bias: bool = True):
        super(UpSample, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.method = method
        
        if method == 'transpose_conv':
            self.conv_transpose = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=scale_factor, 
                stride=scale_factor, bias=bias
            )
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
        elif method == 'bilinear':
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=method, align_corners=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
        elif method == 'nearest':
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=method)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
        elif method == 'pixel_shuffle':
            self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 
                                kernel_size, padding=padding, bias=bias)
            self.pixel_shuffle = nn.PixelShuffle(scale_factor)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
        elif method == 'sub_pixel':
            self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size, padding=padding, bias=bias)
            self.conv2 = nn.Conv2d(in_channels * 2, out_channels * (scale_factor ** 2), 
                                 kernel_size, padding=padding, bias=bias)
            self.pixel_shuffle = nn.PixelShuffle(scale_factor)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported upsample method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)
            
        Returns:
            torch.Tensor: 上采样后的张量，形状为 (B, out_channels, H*scale_factor, W*scale_factor)
        """
        if self.method == 'transpose_conv':
            x = self.conv_transpose(x)
            x = self.bn(x)
            x = self.relu(x)
        elif self.method in ['bilinear', 'nearest']:
            x = self.upsample(x)
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        elif self.method == 'pixel_shuffle':
            x = self.conv(x)
            x = self.pixel_shuffle(x)
            x = self.bn(x)
            x = self.relu(x)
        elif self.method == 'sub_pixel':
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = self.pixel_shuffle(x)
            x = self.bn(x)
            x = self.relu(x)
            
        return x

class AdaptiveSampling(nn.Module):
    """自适应采样模块，可以根据输入自动选择上采样或下采样
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        target_size (tuple): 目标尺寸 (H, W)
        method (str): 采样方法
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 target_size: tuple,
                 method: str = 'bilinear'):
        super(AdaptiveSampling, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_size = target_size
        self.method = method
        
        if in_channels != out_channels:
            self.channel_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.channel_conv = nn.Identity()
            
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)
            
        Returns:
            torch.Tensor: 采样后的张量，形状为 (B, out_channels, target_H, target_W)
        """
        x = self.channel_conv(x)
        
        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode=self.method, align_corners=False)
        
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class MultiScaleSampling(nn.Module):
    """多尺度采样模块，同时输出多个不同尺度的特征图
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        scales (list): 尺度列表，如 [0.5, 1.0, 2.0] 表示0.5倍、1倍、2倍尺度
        method (str): 采样方法
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scales: list = [0.5, 1.0, 2.0],
                 method: str = 'bilinear'):
        super(MultiScaleSampling, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.method = method
        
        self.samplers = nn.ModuleList()
        for scale in scales:
            if scale < 1.0:
                scale_factor = int(1.0 / scale)
                sampler = DownSample(in_channels, out_channels, scale_factor, method='avgpool')
            elif scale > 1.0:
                scale_factor = int(scale)
                sampler = UpSample(in_channels, out_channels, scale_factor, method='bilinear')
            else:
                if in_channels != out_channels:
                    sampler = nn.Conv2d(in_channels, out_channels, 1, bias=False)
                else:
                    sampler = nn.Identity()
            self.samplers.append(sampler)
    
    def forward(self, x: torch.Tensor) -> list:
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)
            
        Returns:
            list: 多尺度特征图列表
        """
        outputs = []
        for sampler in self.samplers:
            outputs.append(sampler(x))
        return outputs