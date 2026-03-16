# -*- coding: utf-8 -*-
"""
Attention U-Net model for static IR drop prediction.
@author: Lizi Zhang

Architecture matches the paper (Table 1):
  PreConv: 2×2 kernel, 12 filters → per-channel spatial filtering
  Encoder: C1(32) → C2(64) → C3(128) → C4(256) → Bottleneck(512)
  Decoder: U1(256) → U2(128) → U3(64) → U4(32) → 1×1 Conv → 1 output
  Attention gates on all skip connections
  Dropout in encoder + bottleneck (0.3-0.5 pretrain, 0.1 finetune)

  Input:  (B, 12, 512, 512) — 12 circuit feature maps
  Output: (B, 1, 512, 512)  — predicted IR drop heatmap
"""

import torch
import torch.nn as nn


class conv_block(nn.Module):
    """Two consecutive 3×3 convolution layers, each followed by BatchNorm and ReLU.
    This is the basic building block used in all encoder/decoder levels."""
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    """Upsample spatial resolution by 2x, then apply convolution.
    Used in the decoder path to restore resolution."""
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class Attention_block(nn.Module):
    """Attention gate: learns which spatial regions of the encoder features
    are relevant, suppressing irrelevant areas (e.g., zero-current regions)."""
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi, (x, psi)


def set_dropout_rate(model, rate):
    """Change dropout rate for all Dropout2d layers in the model.
    Called between pretrain (high dropout) and finetune (low dropout)."""
    for module in model.modules():
        if isinstance(module, nn.Dropout2d):
            module.p = rate


class VCAttUNet(nn.Module):
    """Attention U-Net matching the paper's architecture (Table 1).

    Filter sizes: 32, 64, 128, 256, 512 (n1=32).
    PreConv: single 2×2 conv with 12 output channels.
    Dropout applied in encoder + bottleneck.

    Args:
        in_ch:        number of input channels (12 circuit feature maps)
        out_ch:       number of output channels (1 IR drop prediction)
        dropout_rate: dropout probability (0.3-0.5 for pretrain, 0.1 for finetune)
    """
    def __init__(self, in_ch=12, out_ch=1, dropout_rate=0.5):
        super(VCAttUNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # [32, 64, 128, 256, 512]

        # PreConv: single 2×2 conv, 12 filters (paper Table 1)
        # Processes each input feature channel with a small spatial filter
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=1, padding='same'),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

        # Encoder: double 3×3 conv blocks with dropout after each
        self.Conv1 = conv_block(in_ch, filters[0])       # 12→32
        self.Conv2 = conv_block(filters[0], filters[1])   # 32→64
        self.Conv3 = conv_block(filters[1], filters[2])   # 64→128
        self.Conv4 = conv_block(filters[2], filters[3])   # 128→256
        self.Conv5 = conv_block(filters[3], filters[4])   # 256→512 (bottleneck)

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout: applied in encoder and bottleneck
        # Rate changes between pretrain (0.3-0.5) and finetune (0.1)
        self.dropout = nn.Dropout2d(dropout_rate)

        # Decoder with attention gates
        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 4)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        # Final 1×1 conv: map to single output channel
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # PreConv: per-channel spatial filtering
        x = self.pre(x)

        # Encoder path with dropout
        e1 = self.dropout(self.Conv1(x))
        e2 = self.Maxpool1(e1)

        e2 = self.dropout(self.Conv2(e2))
        e3 = self.Maxpool2(e2)

        e3 = self.dropout(self.Conv3(e3))
        e4 = self.Maxpool3(e3)

        e4 = self.dropout(self.Conv4(e4))
        e5 = self.Maxpool4(e4)

        e5 = self.dropout(self.Conv5(e5))  # bottleneck

        # Decoder path with attention skip connections (no dropout)
        d5 = self.Up5(e5)
        e4, att5 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        e3, att4 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        e2, att3 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e1, att2 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out, x
