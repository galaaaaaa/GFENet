import torch
import os
import torch.nn as nn
from .common import *
from mmengine.model import BaseModule
from mmagic.registry import MODELS
from .modules import DEBlockTrain,mgab, DEABlock

@MODELS.register_module()
class GFENet(BaseModule):

    def __init__(
        self,
        in_channels: int,
        num_feats: int = 32,
        kernel_size: int = 3,
        scale: int = 2,
        norm_flag=0,
        hr_block = 'res',
        norm_dict={"mean": None, "std": None, "min": None, "max": None},
        deploy=False,
        ablation_variant=None,
    ):
        super(GFENet, self).__init__()
        self.hr_block = hr_block
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.ablation_variant = ablation_variant
        
        Block = mgab

        self.conv_rgb1 =nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_feats, kernel_size=kernel_size, padding=1),
                                      Block(default_conv, num_feats, 3))
                                     
        if self.hr_block == 'res':
            self.rgb_rb2 = ResBlock(
                default_conv,
                num_feats,
                kernel_size,
                bias=True,
                bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                res_scale=1,
            )
            self.rgb_rb3 = ResBlock(
                default_conv,
                num_feats,
                kernel_size,
                bias=True,
                bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                res_scale=1,
            )
            self.rgb_rb4 = ResBlock(
                default_conv,
                num_feats,
                kernel_size,
                bias=True,
                bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                res_scale=1,
            )
        if self.hr_block == 'DEB':
            base_dim = num_feats
            # self.rgb_rb2 = DEBlockTrain(default_conv, base_dim, 3)
            self.rgb_rb2 = Block(default_conv, base_dim, 3)

            self.rgb_rb3 = Block(default_conv, base_dim, 3)
            self.rgb_rb4 = Block(default_conv, base_dim, 3)

        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats, kernel_size=kernel_size, padding=1)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg2 = ResidualGroup(default_conv, 64, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg3 = ResidualGroup(default_conv, 96, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg4 = nn.Sequential(Block(default_conv, 128, kernel_size, reduction=16),
                                    Block(default_conv, 128, kernel_size, reduction=16),
                                    Block(default_conv, 128, kernel_size, reduction=16),
                                    Block(default_conv, 128, kernel_size, reduction=16))

        self.bridge1 = deim(dp_feats=32, add_feats=32, scale=scale, ablation_variant=ablation_variant)
        self.bridge2 = deim(dp_feats=64, add_feats=32, scale=scale, ablation_variant=ablation_variant)
        self.bridge3 = deim(dp_feats=96, add_feats=32, scale=scale, ablation_variant=ablation_variant)

        self.bridge4 = g2fr(dp_feats=128, add_feats=32, scale=scale)
        self.bridge5 = g2fr(dp_feats=128, add_feats=32, scale=scale)

        # self.downsample = default_conv(1, 128, kernel_size=kernel_size)

        self.tail_1= nn.Sequential(Block(default_conv, 128, kernel_size, reduction=16),
                                Block(default_conv, 128, kernel_size, reduction=16),
                                Block(default_conv, 128, kernel_size, reduction=16),
                                Block(default_conv, 128, kernel_size, reduction=16))
        self.tail_2 = nn.Sequential(Block(default_conv, 128, kernel_size, reduction=16),
                        Block(default_conv, 128, kernel_size, reduction=16),
                        Block(default_conv, 128, kernel_size, reduction=16),
                        Block(default_conv, 128, kernel_size, reduction=16))
        
        self.upsampler = DenseProjection(128, 128, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(128, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True),
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode="bicubic")
        self.conv_fuse=default_conv(2, 1, kernel_size=1, bias=True)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # def forward(self, depth, image):
    def forward(self, x, hr_guidance, **kwargs):
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1:  # z-score
            assert self.norm_dict["mean"] is not None and self.norm_dict["std"] is not None
            x = (x - self.norm_dict["mean"]) / self.norm_dict["std"]
        elif self.norm_flag == 2:  # min-max
            assert self.norm_dict["min"] is not None and self.norm_dict["max"] is not None
            x = (x - self.norm_dict["min"]) / (self.norm_dict["max"] - self.norm_dict["min"])

        lst = x
        dp_in = self.act(self.conv_dp1(x))
        dp1 = self.dp_rg1(dp_in)

        rgb1 = self.act(self.conv_rgb1(hr_guidance))
        rgb2 = self.rgb_rb2(rgb1)
        ca1_in = self.bridge1(dp1, rgb2)

        dp2 = self.dp_rg2(ca1_in)
        rgb3 = self.rgb_rb3(rgb2)
        ca2_in = self.bridge2(dp2, rgb3)

        dp3 = self.dp_rg3(ca2_in)
        rgb4 = self.rgb_rb4(rgb3)
        ca3_in = self.bridge3(dp3, rgb4)
       
        dp4 = self.dp_rg4(ca3_in)
        tail_in = self.upsampler(dp4)
        
        # g2fr Connection Ablations
        if self.ablation_variant == 'symmetric':
            # Symmetric / Level-wise Alignment
            # rgb2 -> first g2fr, rgb3 -> second g2fr
            # Note: The original implementation was:
            # bridge4 (first g2fr) <- rgb3
            # bridge5 (second g2fr) <- rgb2
            # So 'symmetric' here might mean swapping them or keeping them aligned with depth.
            # Assuming "level-wise alignment" means:
            # deeper features (rgb3) -> deeper g2fr (bridge5)?
            # OR shallower features (rgb2) -> shallower g2fr (bridge4)?
            # Let's look at the original code:
            # bridge4([tail_in, rgb3])
            # bridge5([out, rgb2])
            # This IS essentially symmetric in terms of skip connections (U-Net style).
            # rgb2 is early, rgb3 is later.
            # bridge4 is early in decoder, bridge5 is later in decoder.
            # Wait, tail_1 is AFTER bridge4.
            # So bridge4 is "deeper" in the decoder sense (closer to bottleneck), bridge5 is "shallower" (closer to output).
            # rgb3 is "deeper" in encoder, rgb2 is "shallower".
            # So original: bridge4(rgb3), bridge5(rgb2) matches Deep-Deep, Shallow-Shallow.
            # This IS the "Symmetric" configuration described as "Strawman".
            # So Baseline IS Symmetric.
            
            # Original code: bridge4 uses rgb3, bridge5 uses rgb2.
            # User wants A: bridge4 uses rgb2, bridge5 uses rgb3.
            # This seems to be "Asymmetric" or "Crossed" relative to U-Net skip connections?
            # Or maybe user considers bridge4 as "first" and bridge5 as "second".
            # And rgb2 is "first" (after rgb1), rgb3 is "second".
            # So "first to first, second to second" = rgb2 -> bridge4, rgb3 -> bridge5.
            
            tail_in = self.bridge4([tail_in, rgb2])
            out = self.tail_1(tail_in)
            out = self.bridge5([out, rgb3])
            
        elif self.ablation_variant == 'single_fused':
            # Fuse rgb2 and rgb3
            # Simple summation or concat? Since channels match (num_feats), sum is easiest.
            rgb_fused = rgb2 + rgb3
            tail_in = self.bridge4([tail_in, rgb_fused])
            out = self.tail_1(tail_in)
            # Skip bridge5
            # out = self.bridge5([out, ...]) 
            # But wait, tail_2 expects input from bridge5 output?
            # tail_2 is just blocks. bridge5 output shape same as input.
            # So we can just pass out directly to tail_2.
            pass

        elif self.ablation_variant == 'rgb2_only':
             tail_in = self.bridge4([tail_in, rgb2])
             out = self.tail_1(tail_in)
             out = self.bridge5([out, rgb2])
             
        else:
            # Baseline (Original)
            # bridge4 uses rgb3
            # bridge5 uses rgb2
            tail_in = self.bridge4([tail_in, rgb3])
            out = self.tail_1(tail_in)
            out = self.bridge5([out, rgb2])

        out = self.tail_2(out)
        
        out = self.last_conv(out)

        out = out + self.bicubic(lst)
        
        if self.norm_flag == 1:
            out = out * self.norm_dict["std"] + self.norm_dict["mean"]
        elif self.norm_flag == 2:
            out = out * (self.norm_dict["max"] - self.norm_dict["min"]) + self.norm_dict["min"]

        return out

if __name__ == "__main__":
    hr_guidance = torch.randn((1, 3, 256, 256)).cuda()
    lst = torch.randn((1, 1, 128, 128)).cuda()
    net = lst(num_feats=32, kernel_size=3, scale=2).cuda()
    output = net(lst, hr_guidance)
    print(output.shape)
    pass
