#!/usr/bin/env python
# coding: utf-8
#


from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplabv3 import _ASPP
from resnet import _ConvBnReLU, _ResLayer, _Stem
from pdb import set_trace as bp
from torchsummary import summary
class DeepLabV3Plus(nn.Module):
    """
    DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decoder
    """

    def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
        super(DeepLabV3Plus, self).__init__()

        # Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # Encoder
        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0])
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
        self.aspp = _ASPP(ch[5], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))

        # Decoder
        self.reduce = _ConvBnReLU(256, 48, 1, 1, 0, 1)
        self.fc2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", _ConvBnReLU(304, 256, 3, 1, 1, 1)),
                    ("conv2", _ConvBnReLU(256, 256, 3, 1, 1, 1)),
                    ("conv3", nn.Conv2d(256, n_classes, kernel_size=1)),
                ]
            )
        )

    def forward(self, x):#in: [1, 3, 513, 513]
        #bp()
        h = self.layer1(x)#out: [1, 64, 129, 129]
        h = self.layer2(h)#out: [1, 256, 129, 129]
        h_ = self.reduce(h)# h_ shape: [1, 48, 129, 129]
        h = self.layer3(h)#out:[1, 512, 65, 65]
        h = self.layer4(h)#out:[1, 1024, 33, 33]
        h = self.layer5(h)#out:[1, 2048, 33, 33]
        h = self.aspp(h)#out:[1, 1280, 33, 33]
        h = self.fc1(h)#out:[1, 256, 33, 33]
        h = F.interpolate(h, size=h_.shape[2:], mode="bilinear", align_corners=False)#[1, 256, 129, 129]
        h = torch.cat((h, h_), dim=1)#[1, 304, 129, 129]
        h = self.fc2(h)#[1, 21, 129, 129]
        h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)#[1, 21, 513, 513]
        return h
"""
AAD layer Adaptive attention denormalized 
Input:
s: [-1,ch_s,H,W]任意不同scale的atrous得到的feature
w: [-1,ch_w,H,W]整体图像的low level feature
h_in: [-1,ch_h,H,W]前面一级输入的activation map
Outpu:
h_out: [-1,ch_h,H,W] : shape same as input feature map
"""
class AAD(nn.Module):
    def __init__(self,ch_h,ch_s,ch_w):
        super().__init__()

        
        self.s_conv_gamma=nn.Conv2d(ch_s,ch_h,kernel_size=1,stride=1,padding=0,bias=True)
        self.s_conv_beta=nn.Conv2d(ch_s,ch_h,kernel_size=1,stride=1,padding=0,bias=True)
        self.w_conv_gamma=nn.Conv2d(ch_w,ch_h,kernel_size=1,stride=1,padding=0,bias=True)
        self.w_conv_beta=nn.Conv2d(ch_w,ch_h,kernel_size=1,stride=1,padding=0,bias=True)
        self.norm=nn.InstanceNorm2d(ch_h,affine=False)
        self.h_conv=nn.Conv2d(ch_h,1,kernel_size=1,stride=1,padding=0,bias=True)
        

        
    def forward(self,h_in,s,w):
        #bp()
        h_norm=self.norm(h_in)
        
        gamma_s=self.s_conv_gamma(s)
        beta_s=self.s_conv_beta(s)
        S=h_norm*gamma_s+beta_s

        gamma_w=self.w_conv_gamma(w)
        beta_w=self.w_conv_beta(w)
        W=h_norm*gamma_w+beta_w
    
        M=self.h_conv(h_norm)
        M=torch.sigmoid(M)

        #out=(1-M)*A+M*I
        out=(torch.ones_like(M).to(M.device)-M)*S+M*W
        return out
"""
AAD resBLK:
Input:
h_in: [-1,ch_hin,H,W]前面一级输入的activation map
s:[-1,ch_s,W,W]任意不同scale的atrous得到的feature
w: [-1,ch_s,W,W]整体图像的low level feature
Output:
h_out:[-1,ch_hout,n,n]下面一级输入的activation map
"""
class  AAD_ResBLK(nn.Module):
    def __init__(self,ch_hin,ch_hout,ch_s,ch_w):
        super().__init__()
        self.ch_hin=ch_hin
        self.ch_hout=ch_hout
        self.AAD1=AAD(ch_hin,ch_s,ch_w)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(ch_hin,ch_hin,kernel_size=3,padding=1,stride=1,bias=False)
        
        self.AAD2=AAD(ch_hin,ch_s,ch_w)
        self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(ch_hin,ch_hout,kernel_size=3,padding=1,stride=1,bias=False)

        self.AAD3=AAD(ch_hin,ch_s,ch_w)
        self.relu3=nn.ReLU(inplace=True)
        self.conv3=nn.Conv2d(ch_hin,ch_hout,kernel_size=3,padding=1,stride=1,bias=False)
        

    def forward(self,h_in,s,w):
        y=self.AAD1(h_in,s,w)
        y=self.relu1(y)
        y=self.conv1(y)

        y=self.AAD2(y,s,w)
        y=self.relu2(y)
        y=self.conv2(y)

        if self.ch_hin!=self.ch_hout:
            h_in=self.AAD3(h_in,s,w)
            h_in=self.relu3(h_in)
            h_in=self.conv3(h_in)
            
        out=y+h_in
        return out
"""
DeepLabv3+ Decoder part, assume low level features output from Conv2 is 
[-1, 48, 129, 129]
scaled features from ASPP is
s1/s2/s3/s4/s5 : [-1, 256, 129, 129]

ResBlk1(48,100):ResBlk1输入48 ch，输出100个ch
ResBlk2(100,152):ResBlk2输入100 ch，输出152个ch
ResBlk3(152,204):ResBlk3输入152 ch，输出204个ch
ResBlk4(204,256):ResBlk4输入204 ch，输出256个ch
ResBlk5(256,256):ResBlk5输入256 ch，输出256个ch---这个是输入输出ch数相同的一个AAResBlk

"""
class AADecoder(nn.Module):
    def __init__(self,ch_w,ch_s1,ch_s2,ch_s3, ch_s4, ch_s5, ch_o1,ch_o2,ch_o3, ch_o4, ch_o5):
        """
        ch_w:low level feature ch 数
        ch_s1,ch_s2,ch_s3, ch_s4, ch_s5: scale featuer ch 数
        ch_o1:希望的ResBlk1 输出ch数
        ch_o2:希望的ResBlk2 输出ch数
        ch_o3:希望的ResBlk3 输出ch数
        ch_o4:希望的ResBlk4 输出ch数
        ch_o5:希望的ResBlk5 输出ch数,就是最终输出ch数
        """
        super().__init__()
        self.AAResBlk1=AAD_ResBLK(ch_w,ch_o1,ch_s1,ch_w)
        self.AAResBlk2=AAD_ResBLK(ch_o1,ch_o2,ch_s2,ch_w)
        self.AAResBlk3=AAD_ResBLK(ch_o2,ch_o3,ch_s3,ch_w)
        self.AAResBlk4=AAD_ResBLK(ch_o3,ch_o4,ch_s4,ch_w)
        self.AAResBlk5=AAD_ResBLK(ch_o4,ch_o5,ch_s5,ch_w)
        
    def forward(self,w,s1,s2,s3,s4,s5):
        h1=self.AAResBlk1(w,s1,w)#输出 [1, 100, 129, 129]
        h2=self.AAResBlk2(h1,s2,w)#输出 [1, 152, 129, 129]
        h3=self.AAResBlk3(h2,s3,w)#输出[1, 204, 129, 129]
        h4=self.AAResBlk4(h3,s4,w)#输出[1, 256, 129, 129]
        h5=self.AAResBlk5(h4,s5,w)#输出[1, 256, 129, 129]
        return h5
        
        
    
if __name__ == "__main__":
    #---AAD
    ch_h,ch_s,ch_w=256,256,256
    H,W=129,129
    aad=AAD(ch_h,ch_s,ch_w)
    s=torch.randn(1, ch_s, H, W)
    w=torch.randn(1, ch_w, H, W)
    h=torch.randn(1, ch_h, H, W)
    out=aad(h,s,w)#out.shape should be [1, 256, 129, 129]
    summary(aad,[(ch_h,H,W),(ch_s,H,W),(ch_w,H,W)],device="cpu")
    """
    ch_h,ch_s,ch_w=256,256,256 : 参数total params: 263,425
    """
    #bp()
    #---AAD ResBlk
    ch_hin,ch_hout=256,256
    aad_ResBLK=AAD_ResBLK(ch_hin,ch_hout,ch_s,ch_w)
    h_out=aad_ResBLK(h,s,w)
    summary(aad_ResBLK,[(ch_h,H,W),(ch_s,H,W),(ch_w,H,W)],device="cpu")
    """
    ch_h,ch_s,ch_w=256,256,256 : 参数total params: 1,706,498
    """
    
    #---AAD Decoder
    """
    ch_w=48
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=100,152,204,256,256
    
    ch_w=32
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=88,144,200,256,256

    ch_w=64
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=112,160,208,256,256

    ch_w=16
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=76,136,196,256,256
           
    ch_w=8
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=70,132,194,256,256
    """
    ch_w=80
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=124,168,212,256,256
    
    ch_s1,ch_s2,ch_s3, ch_s4, ch_s5=256,256,256,256,256
    
    
    aad_Decoder=AADecoder(ch_w,ch_s1,ch_s2,ch_s3, ch_s4, ch_s5, ch_o1,ch_o2,ch_o3, ch_o4, ch_o5)
    w=torch.randn(1, ch_w, H, W)
    s1,s2,s3,s4,s5=torch.randn(1, ch_s1, H, W),torch.randn(1, ch_s2, H, W),torch.randn(1, ch_s3, H, W),torch.randn(1, ch_s4, H, W),torch.randn(1, ch_s5, H, W)
    
    h_out=aad_Decoder(w,s1,s2,s3,s4,s5)
    summary(aad_Decoder, [(ch_w,H,W), (ch_s1,H,W), (ch_s2,H,W),(ch_s3,H,W),(ch_s4,H,W),(ch_s5,H,W)],device="cpu")
    bp()
    """
    ch_w=48
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=100,152,204,256,256
    Total params: 4,971,766

    ch_w=32
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=88,144,200,256,256
    Total params: 4,630,206

    ch_w=64
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=112,160,208,256,256
    Total params: 5,341,166

    ch_w=16
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=76,136,196,256,256
    Total params: 4,316,486

    ch_w=8
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=70,132,194,256,256
    Total params: 4,170,066

    ch_w=80
    ch_o1,ch_o2,ch_o3, ch_o4, ch_o5=124,168,212,256,256
    Total params: 5,738,406
    """
    bp()
    #---deeplabv3plus
    model = DeepLabV3Plus(
        n_classes=21,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=16,
    )
    bp()
    model.eval()
    image = torch.randn(1, 3, 513, 513)
    out=model(image)
    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
