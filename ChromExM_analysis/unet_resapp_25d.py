import torch
import torch.nn as nn


class ResBlockGN(nn.Module):
    """
    Residual block with GroupNorm.
    """
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        g = max(1, min(groups, out_ch))
        while out_ch % g != 0 and g > 1:
            g -= 1

        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(g, out_ch)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        idt = self.proj(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + idt)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling block.

    GroupNorm groups are adjusted so they divide the output channel count.
    """
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()

        # Make GroupNorm safe if out_ch is not divisible by the default group count.
        g = max(1, min(groups, out_ch))
        while out_ch % g != 0 and g > 1:
            g -= 1

        self.b1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.b2 = nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False)
        self.b3 = nn.Conv2d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False)
        self.b4 = nn.Conv2d(in_ch, out_ch, 3, padding=8, dilation=8, bias=False)

        self.proj = nn.Conv2d(out_ch * 4, out_ch, 1, bias=False)
        self.gn   = nn.GroupNorm(g, out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.act(self.gn(self.proj(x)))


class UNetResASPP(nn.Module):
    """
    2D U-Net with residual blocks + ASPP.

    For 2.5D training/inference:
      - Pass in_channels=3 (e.g., [z-1, z, z+1] as channels).
      - Output remains a single-channel probability map (out_channels=1).
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        ch = [64, 128, 256, 512]

        self.down1 = ResBlockGN(in_channels, ch[0]); self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResBlockGN(ch[0], ch[1]);       self.pool2 = nn.MaxPool2d(2)
        self.down3 = ResBlockGN(ch[1], ch[2]);       self.pool3 = nn.MaxPool2d(2)

        self.middle = ResBlockGN(ch[2], ch[3])
        self.aspp   = ASPP(ch[3], ch[3])

        self.up3  = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2); self.upc3 = ResBlockGN(ch[2] + ch[2], ch[2])
        self.up2  = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2); self.upc2 = ResBlockGN(ch[1] + ch[1], ch[1])
        self.up1  = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2); self.upc1 = ResBlockGN(ch[0] + ch[0], ch[0])

        self.final = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        xm = self.aspp(self.middle(self.pool3(x3)))

        x  = self.upc3(torch.cat([self.up3(xm), x3], dim=1))
        x  = self.upc2(torch.cat([self.up2(x),  x2], dim=1))
        x  = self.upc1(torch.cat([self.up1(x),  x1], dim=1))

        # Sigmoid output for binary probability maps.
        return torch.sigmoid(self.final(x))