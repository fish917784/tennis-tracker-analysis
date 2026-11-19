"""Figure-guided TrackNetV2 + surround priors + cross-scale attention.

MSPE/CSCA equations are engineering choices, not an exact paper reproduction.
See README.md for the correspondence and assumptions.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, depth=2):
        layers = []
        for _ in range(depth):
            layers.extend([nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                           nn.BatchNorm2d(cout), nn.ReLU(inplace=True)])
            cin = cout
        super().__init__(*layers)


class MSPE(nn.Module):
    """Center-surround contrasts in eight directions at three radii.

    Replicated borders avoid artificial image-edge contrast. Absolute contrast
    preserves both bright-on-dark and dark-on-bright small target boundaries.
    """
    def __init__(self, channels, radii=(1, 2, 3)):
        super().__init__()
        self.radii = tuple(radii)
        self.project = nn.Sequential(
            nn.Conv2d(channels * len(radii), channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.fuse = ConvBlock(channels * 2, channels, depth=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        contrasts = []
        for radius in self.radii:
            padded = F.pad(x, (radius,) * 4, mode="replicate")
            neighbors = [padded[:, :, radius + dy:radius + dy + h,
                                radius + dx:radius + dx + w]
                         for dy, dx in ((-radius, -radius), (-radius, 0),
                                        (-radius, radius), (0, -radius),
                                        (0, radius), (radius, -radius),
                                        (radius, 0), (radius, radius))]
            contrasts.append(sum((x - neighbor).abs() for neighbor in neighbors) / 8)
        prior = self.project(torch.cat(contrasts, dim=1))
        return x + self.fuse(torch.cat([x, prior], dim=1))


class CSCA(nn.Module):
    """Shallow structure and deeper decoder context jointly predict gates."""
    def __init__(self, shallow_channels, context_channels, reduction=8):
        super().__init__()
        channels = shallow_channels
        self.context = nn.Sequential(nn.Conv2d(context_channels, channels, 1, bias=False),
                                     nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        hidden = max(channels // reduction, 8)
        self.channel = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Conv2d(4, 1, 7, padding=3)
        self.fuse = ConvBlock(2 * channels, channels, depth=1)

    def forward(self, shallow, context):
        context = self.context(F.interpolate(context, size=shallow.shape[-2:],
                                             mode="bilinear", align_corners=False))
        joint = shallow + context
        channel_gate = torch.sigmoid(self.channel(F.adaptive_avg_pool2d(joint, 1)) +
                                     self.channel(F.adaptive_max_pool2d(joint, 1)))
        spatial_gate = torch.sigmoid(self.spatial(torch.cat([
            shallow.mean(1, keepdim=True), shallow.amax(1, keepdim=True),
            context.mean(1, keepdim=True), context.amax(1, keepdim=True)], dim=1)))
        return shallow + self.fuse(torch.cat([
            shallow * channel_gate * spatial_gate, context], dim=1))


class TrajectoryNetPP(nn.Module):
    """Three RGB frames (B,9,H,W) -> three heatmaps (B,3,H,W).

    TrackNetV2-style widths 64/128/256/512, three downsampling stages.
    MSPE follows the first two encoder blocks. Two CSCA skip paths connect
    these enhanced features with deeper decoder features at matching sizes.
    """
    architecture = "tracknetv2_mspe_csca_v1"

    def __init__(self, in_channels=9, out_channels=3, disable_mspe=False,
                 disable_csca=False, return_logits=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.disable_mspe = disable_mspe
        self.disable_csca = disable_csca
        self.return_logits = return_logits
        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256, depth=3)
        self.bottleneck = ConvBlock(256, 512, depth=3)
        self.pool = nn.MaxPool2d(2)
        self.mspe1, self.mspe2 = MSPE(64), MSPE(128)
        self.decoder3 = ConvBlock(512, 256, depth=3)
        self.csca2 = CSCA(128, 256)
        self.decoder2 = ConvBlock(256 + 128, 128)
        self.csca1 = CSCA(64, 128)
        self.decoder1 = ConvBlock(128 + 64, 64)
        self.head = nn.Conv2d(64, out_channels, 1)

    @staticmethod
    def up(x, reference):
        return F.interpolate(x, size=reference.shape[-2:], mode="nearest")

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected (B, {}, H, W) input".format(self.in_channels))
        if min(x.shape[-2:]) < 8:
            raise ValueError("Input height and width must both be at least 8")
        s1 = self.encoder1(x)
        if not self.disable_mspe:
            s1 = self.mspe1(s1)
        s2 = self.encoder2(self.pool(s1))
        if not self.disable_mspe:
            s2 = self.mspe2(s2)
        s3 = self.encoder3(self.pool(s2))
        deep = self.bottleneck(self.pool(s3))
        d3 = self.decoder3(self.up(deep, s3))
        a2 = s2 if self.disable_csca else self.csca2(s2, d3)
        d2 = self.decoder2(torch.cat([self.up(d3, s2), a2], dim=1))
        a1 = s1 if self.disable_csca else self.csca1(s1, d2)
        d1 = self.decoder1(torch.cat([self.up(d2, s1), a1], dim=1))
        logits = self.head(d1)
        return logits if self.return_logits else torch.sigmoid(logits)


TrackNet = TrajectoryNetPP
