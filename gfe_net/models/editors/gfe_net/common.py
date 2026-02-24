import torch.nn as nn
import torch
from torchvision import transforms
import torch.nn.init as init
from .sampling_modules import DownSample,UpSample
import numpy as np
import scipy.linalg
from .import thops
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {2: (6, 2, 2), 4: (8, 4, 2), 8: (12, 8, 2), 16: (20, 16, 2)}[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d

    return conv_f(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                res_scale=1,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thops.pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out
class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out
class Freprocess(nn.Module):
    def __init__(self, channels, coupled=False):
        super(Freprocess, self).__init__()
        self.coupled = coupled
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        
        if not coupled:
            self.amp_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                          nn.Conv2d(channels,channels,1,1,0))
            self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                          nn.Conv2d(channels,channels,1,1,0))
        else:
            self.coupled_fuse = nn.Sequential(
                nn.Conv2d(4*channels, channels, 1, 1, 0),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(channels, 2*channels, 1, 1, 0)
            )

        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        
        if not self.coupled:
            msF_amp = torch.abs(msF)
            msF_pha = torch.angle(msF)
            panF_amp = torch.abs(panF)
            panF_pha = torch.angle(panF)
            amp_fuse = self.amp_fuse(torch.cat([msF_amp,panF_amp],1))
            pha_fuse = self.pha_fuse(torch.cat([msF_pha,panF_pha],1))

            real = amp_fuse * torch.cos(pha_fuse)+1e-8
            imag = amp_fuse * torch.sin(pha_fuse)+1e-8
            out = torch.complex(real, imag)+1e-8
        else:
            cat_input = torch.cat([msF.real, msF.imag, panF.real, panF.imag], dim=1)
            coupled_out = self.coupled_fuse(cat_input)
            out_real, out_imag = torch.chunk(coupled_out, 2, dim=1)
            out = torch.complex(out_real, out_imag) + 1e-8

        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
class deim(nn.Module):
    def __init__(self, dp_feats, add_feats, scale, ablation_variant=None):
        super(deim, self).__init__()
        self.ablation_variant = ablation_variant
        
        if ablation_variant != 'no_spatial':
            self.gfe_net = SUFT_adaptive(dp_feats, add_feats, scale=scale)
        else:
            self.gfe_net = None

        self.conv_skip=nn.Conv2d(dp_feats,dp_feats+add_feats,1,1,0)
        
        self.upsample = DenseProjection(dp_feats, add_feats, scale, up=True)
        self.downsample = DenseProjection(dp_feats+add_feats, dp_feats+add_feats, scale, up=False)

        self.spa_process = nn.Sequential(InvBlock(DenseBlock, 2*add_feats, add_feats),
                                         nn.Conv2d(2*add_feats,add_feats,1,1,0))
        
        if ablation_variant != 'no_freq':
            coupled = (ablation_variant == 'coupled_freq')
            self.fre_process = Freprocess(add_feats, coupled=coupled)
        else:
            self.fre_process = None

        self.spa_att = nn.Sequential(nn.Conv2d(add_feats, add_feats // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(add_feats // 2, add_feats, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        
        cat_feats = 0
        if ablation_variant != 'no_spatial':
            cat_feats += dp_feats
        if ablation_variant != 'no_freq':
            cat_feats += add_feats

        self.cha_att = nn.Sequential(nn.Conv2d(cat_feats, dp_feats // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(dp_feats // 2, cat_feats, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(cat_feats, dp_feats+add_feats, 3, 1, 1)

    def forward(self, msf, pan):
          #, i
        msf_or=self.conv_skip(msf)
        
        feats = []
        if self.gfe_net is not None:
            spafuse = self.gfe_net(msf,pan)
            feats.append(spafuse)
        
        if self.fre_process is not None:
            msf_up=self.upsample(msf)
            frefuse = self.fre_process(msf_up,pan)
            feats.append(frefuse)
        
        cat_f = torch.cat(feats,1)
        cha_res =  self.post(self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f))*cat_f)
        out = cha_res
        out=self.downsample(out)
        out=out+msf_or

        return out
class g2fr(nn.Module):
    def __init__(self, dp_feats,add_feats,scale, height=2, reduction=8):
        super(g2fr, self).__init__()
        self.dp_feats=dp_feats
        self.add_feats=add_feats
        
        self.height = height
        
        d = max(int(dp_feats/reduction), 4)
       
        self.conv1=nn.Conv2d(add_feats,dp_feats,kernel_size=1)
      
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dp_feats, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dp_feats*height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)
        self.conv4=nn.Conv2d(2*dp_feats,dp_feats,kernel_size=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats[1]=self.conv1(in_feats[1])
        x=in_feats[0]

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats*attn, dim=1)
        out = self.conv4(torch.cat([out, x], dim=1))

        return out

class SUFT(nn.Module):
    def __init__(self, dp_feats, add_feats, scale):
        super(SUFT, self).__init__()
        self.fliper = transforms.RandomHorizontalFlip(1)
        self.dp_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)
        self.dpf_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)
        self.total_down = DenseProjection(dp_feats + add_feats, dp_feats + add_feats, scale, up=False, bottleneck=False)
        self.conv_du = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.dif_layer = nn.Identity()

    def forward(self, depth, rgb):
        dpf = self.fliper(depth)
        dp_h = self.dp_up(depth)
        dpf_h = self.dpf_up(dpf)
        dif = torch.abs(dp_h - self.fliper(dpf_h))
        dif = self.dif_layer(dif)

        dif_avg = torch.mean(dif, dim=1, keepdim=True)
        dif_max, _ = torch.max(dif, dim=1, keepdim=True)
        attention = self.conv_du(torch.cat([dif_avg, dif_max], dim=1))
        max = torch.max(torch.max(attention, -1)[0], -1)[0].unsqueeze(1).unsqueeze(2)
        min = torch.min(torch.min(attention, -1)[0], -1)[0].unsqueeze(1).unsqueeze(2)

        attention = (attention - min) / (max - min + 1e-12)
        rgb_h = rgb * attention
        total = torch.cat([dp_h, rgb_h], dim=1)
        out = self.total_down(total)
        return out
class SUFT_adaptive(nn.Module):
    def __init__(self, dp_feats, add_feats, scale):
        super(SUFT_adaptive, self).__init__()
        self.fliper = transforms.RandomHorizontalFlip(1)
        self.dp_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)
        self.dpf_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)
        self.total_down = DenseProjection(dp_feats + add_feats, dp_feats + add_feats, scale, up=False, bottleneck=False)
        self.conv_du = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.conv_last = nn.Conv2d(dp_feats + add_feats, dp_feats , kernel_size=1, padding=0, bias=True)
        self.dif_layer = nn.Identity()

    def forward(self, depth, rgb):
        dpf = self.fliper(depth)
        dp_h = self.dp_up(depth)
        dpf_h = self.dpf_up(dpf)
        dif = torch.abs(dp_h - self.fliper(dpf_h))
        dif = self.dif_layer(dif)

        dif_avg = torch.mean(dif, dim=1, keepdim=True)
        dif_max, _ = torch.max(dif, dim=1, keepdim=True)
        attention = self.conv_du(torch.cat([dif_avg, dif_max], dim=1))
        max = torch.max(torch.max(attention, -1)[0], -1)[0].unsqueeze(1).unsqueeze(2)
        min = torch.min(torch.min(attention, -1)[0], -1)[0].unsqueeze(1).unsqueeze(2)

        attention = (attention - min) / (max - min + 1e-12)
        rgb_h = rgb * attention
        total = torch.cat([dp_h, rgb_h], dim=1)
        total = self.conv_last(total)
        
        return total

class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        self.up = up
        if bottleneck:
            self.bottleneck = nn.Sequential(*[nn.Conv2d(in_channels, nr, 1), nn.PReLU(nr)])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(*[projection_conv(inter_channels, nr, scale, up), nn.PReLU(nr)])
        self.conv_2 = nn.Sequential(*[projection_conv(nr, inter_channels, scale, not up), nn.PReLU(inter_channels)])
        self.conv_3 = nn.Sequential(*[projection_conv(inter_channels, nr, scale, up), nn.PReLU(nr)])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)

        out = a_0.add(a_1)
        return out
