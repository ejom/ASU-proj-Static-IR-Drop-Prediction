# -*- coding: utf-8 -*-
"""
Attention U-Net model for static IR drop prediction.
@author: Lizi Zhang

Architecture overview (VCAttUNet):
  This is a U-Net with attention gates. U-Net is an encoder-decoder architecture:

  ENCODER (left side): progressively shrinks spatial resolution while increasing
  the number of feature channels, capturing high-level patterns.
    512x512 → 256x256 → 128x128 → 64x64 → 32x32

  DECODER (right side): progressively upsamples back to original resolution,
  combining high-level features with fine-grained details from the encoder
  via skip connections.
    32x32 → 64x64 → 128x128 → 256x256 → 512x512

  ATTENTION GATES: at each skip connection, an attention mechanism learns
  which spatial regions of the encoder features are most relevant, suppressing
  irrelevant areas. This helps the model focus on IR drop hotspot regions.

  Input:  (B, 12, 512, 512) — 12 circuit feature maps
  Output: (B, 1, 512, 512)  — predicted IR drop heatmap
"""

import torch
import torch.nn as nn


class conv_block(nn.Module):
    """Two consecutive convolution layers, each followed by BatchNorm and ReLU.

    This is the basic building block of U-Net. Two conv layers allow the network
    to learn more complex features at each resolution level.

    Args:
        ch_in:  number of input channels
        ch_out: number of output channels
    """
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            # First conv: ch_in → ch_out, kernel 3x3, padding=1 keeps spatial size unchanged
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),  # normalize activations (stabilizes training)
            nn.ReLU(inplace=True),   # activation function: max(0, x)
            # Second conv: ch_out → ch_out (refine features)
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """Upsample spatial resolution by 2x, then apply convolution.

    Used in the decoder path to progressively increase resolution back
    to the original input size.

    Args:
        ch_in:  number of input channels
        ch_out: number of output channels
    """
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # double H and W (nearest neighbor interpolation)
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    """Attention gate that learns which spatial regions of the encoder features
    are relevant for the current decoder level.

    Takes two inputs:
      g: gating signal from the decoder (lower resolution, high-level context)
      x: encoder features from the skip connection (higher resolution, fine details)

    Produces an attention map (0 to 1) that multiplies x, suppressing irrelevant
    regions and amplifying important ones (like IR drop hotspots).

    Args:
        F_g:   channels in gating signal g
        F_l:   channels in encoder features x
        F_int: intermediate channels (compression for efficiency)
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        # Project gating signal to intermediate space (1x1 conv = per-pixel linear transform)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Project encoder features to same intermediate space
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Combine and produce single-channel attention map (sigmoid → values in [0, 1])
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # squash to [0, 1] — 1 means "pay attention here"
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)           # project gating signal
        x1 = self.W_x(x)           # project encoder features
        psi = self.relu(g1 + x1)   # combine (element-wise add + ReLU)
        psi = self.psi(psi)         # produce attention weights (B, 1, H, W)
        return x * psi, (x, psi)    # multiply encoder features by attention, also return attention map


class VCAttUNet(nn.Module):
    """Attention U-Net for IR drop prediction.

    Channel progression through the network:
      Encoder: 12 → 8 → 16 → 32 → 64 → 128 → 256  (spatial: 512 → 256 → 128 → 64 → 32)
      Decoder: 256 → 128 → 64 → 32 → 16 → 1        (spatial: 32 → 64 → 128 → 256 → 512)

    Args:
        in_ch:  number of input channels (12 circuit feature maps)
        out_ch: number of output channels (1 IR drop prediction)
    """
    def __init__(self, in_ch=12, out_ch=1):
        super(VCAttUNet, self).__init__()

        n1 = 16  # base number of filters
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # [16, 32, 64, 128, 256]

        # Pre-processing: reduce 12 input channels to 8
        self.pre = conv_block(ch_in=in_ch, ch_out=8)

        # --- ENCODER: each level doubles channels, MaxPool halves spatial size ---
        self.Conv1 = conv_block(8, filters[0])          # 8→16,   512x512
        self.Conv2 = conv_block(filters[0], filters[1])  # 16→32,  256x256
        self.Conv3 = conv_block(filters[1], filters[2])  # 32→64,  128x128
        self.Conv4 = conv_block(filters[2], filters[3])  # 64→128, 64x64
        self.Conv5 = conv_block(filters[3], filters[4])  # 128→256, 32x32 (bottleneck)

        # MaxPool layers: reduce spatial size by 2x at each level
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- DECODER: each level halves channels, upsamples to double spatial size ---
        # Each decoder level: upsample → attention gate → concatenate → conv block
        self.Up5 = up_conv(filters[4], filters[3])       # 256→128
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])  # 256→128 (after concat)

        self.Up4 = up_conv(filters[3], filters[2])       # 128→64
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])       # 64→32
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])       # 32→16
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=8)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        # Final 1x1 conv: reduce to single output channel (IR drop prediction)
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Pre-process input
        x = self.pre(x)  # (B, 12, 512, 512) → (B, 8, 512, 512)

        # --- ENCODER PATH (top-down) ---
        e1 = self.Conv1(x)          # (B, 16, 512, 512)
        e2 = self.Maxpool1(e1)      # (B, 16, 256, 256)

        e2 = self.Conv2(e2)         # (B, 32, 256, 256)
        e3 = self.Maxpool2(e2)      # (B, 32, 128, 128)

        e3 = self.Conv3(e3)         # (B, 64, 128, 128)
        e4 = self.Maxpool3(e3)      # (B, 64, 64, 64)

        e4 = self.Conv4(e4)         # (B, 128, 64, 64)
        e5 = self.Maxpool4(e4)      # (B, 128, 32, 32)

        e5 = self.Conv5(e5)         # (B, 256, 32, 32) — bottleneck

        # --- DECODER PATH (bottom-up) with attention skip connections ---
        d5 = self.Up5(e5)                           # upsample: (B, 128, 64, 64)
        e4, att5 = self.Att5(g=d5, x=e4)            # attention-weighted encoder features
        d5 = torch.cat((e4, d5), dim=1)              # concatenate: (B, 256, 64, 64)
        d5 = self.Up_conv5(d5)                       # conv: (B, 128, 64, 64)

        d4 = self.Up4(d5)                            # (B, 64, 128, 128)
        e3, att4 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)              # (B, 128, 128, 128)
        d4 = self.Up_conv4(d4)                        # (B, 64, 128, 128)

        d3 = self.Up3(d4)                            # (B, 32, 256, 256)
        e2, att3 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)              # (B, 64, 256, 256)
        d3 = self.Up_conv3(d3)                        # (B, 32, 256, 256)

        d2 = self.Up2(d3)                            # (B, 16, 512, 512)
        e1, att2 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)              # (B, 32, 512, 512)
        d2 = self.Up_conv2(d2)                        # (B, 16, 512, 512)

        # Final 1x1 conv to produce single-channel output
        out = self.Conv(d2)                           # (B, 1, 512, 512)

        return out, x  # return prediction and pre-processed features


class VCAttUNet_Large(nn.Module):
    """Larger variant of VCAttUNet with 2x the base filters (32 vs 16).
    Same architecture, more parameters for potentially better accuracy at
    higher computational cost. Not used by default."""

    def __init__(self, in_ch=12, out_ch=1):
        super(VCAttUNet_Large, self).__init__()

        n1 = 32  # doubled base filters
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # [32, 64, 128, 256, 512]

        self.pre = conv_block(ch_in=in_ch, ch_out=n1)

        self.Conv1 = conv_block(n1, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

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
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0]//4)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Identical forward pass structure to VCAttUNet, just with wider layers
        x = self.pre(x)

        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

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
