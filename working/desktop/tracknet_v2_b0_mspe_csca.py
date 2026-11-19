"""EfficientNetV2-B0 TrackNet with pooled MSPE and decoder-guided CSCA.

This is an implementation based on the supplied architecture description, not a
claim of reproducing an unspecified paper module exactly.
"""
from typing import List

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:
    raise ImportError("Install timm to use this model: python -m pip install timm") from exc


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int = 3):
        super().__init__(
            nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )


class MSPE(nn.Module):
    def __init__(self, channels: int, out_channels: int = 128, branch_channels: int = 32):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(channels, branch_channels, 1, bias=False), nn.SiLU(inplace=True))
            for b in (1, 2, 4)
        ])
        self.fuse = ConvBNAct(channels + 3 * branch_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        pooled = [F.interpolate(branch(x), size=size, mode="bilinear", align_corners=False) for branch in self.branches]
        return self.fuse(torch.cat([x, *pooled], dim=1))


class CSCA(nn.Module):
    def __init__(self, channels: int = 128, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.SiLU(inplace=True), nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
        self.project = ConvBNAct(channels, channels)

    def forward(self, local: Tensor, cross: Tensor) -> Tensor:
        x = local + cross
        channel = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * channel
        spatial = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return self.project(x * torch.sigmoid(self.spatial(spatial)))


class TrackNetV2B0MSPECSCA(nn.Module):
    """9-channel, three-frame input to three full-resolution probability maps."""
    def __init__(self, in_channels: int = 9, out_channels: int = 3, pretrained: bool = False,
                 disable_mspe: bool = False, disable_csca: bool = False, return_logits: bool = False):
        super().__init__()
        self.disable_mspe = disable_mspe
        self.disable_csca = disable_csca
        self.return_logits = return_logits
        self.encoder = timm.create_model(
            "tf_efficientnetv2_b0", pretrained=pretrained, features_only=True,
            in_chans=in_channels, out_indices=(0, 1, 2, 3, 4),
        )
        channels = self.encoder.feature_info.channels()
        if len(channels) != 5:
            raise RuntimeError(f"Expected five encoder features, got channels={channels}")
        c1, c2, c3, c4, cd = channels
        self.mspe1, self.mspe2 = MSPE(c1), MSPE(c2)
        self.skip1, self.skip2 = ConvBNAct(c1, 128, 1), ConvBNAct(c2, 128, 1)
        self.csca1_context = ConvBNAct(64, 128, 1)
        self.csca1, self.csca2 = CSCA(), CSCA()
        self.deep_project = ConvBNAct(cd, 128, 1)
        self.decoder4 = ConvBNAct(128 + c4, 128)
        self.decoder3 = ConvBNAct(128 + c3, 128)
        self.decoder2 = ConvBNAct(128 + 128, 64)
        self.decoder1 = ConvBNAct(64 + 128, 32)
        self.decoder0 = ConvBNAct(32, 32)
        self.head = nn.Conv2d(32, out_channels, 1)

    @staticmethod
    def resize(x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]
        s1, s2, s3, s4, deep = self.encoder(x)
        m1 = self.mspe1(s1) if not self.disable_mspe else self.skip1(s1)
        m2 = self.mspe2(s2) if not self.disable_mspe else self.skip2(s2)
        z = self.decoder4(torch.cat([self.resize(self.deep_project(deep), s4), s4], dim=1))
        z = self.decoder3(torch.cat([self.resize(z, s3), s3], dim=1))
        # CSCA2 merges the MSPE2 skip with D3 at the S2 scale.
        d3_at_s2 = self.resize(z, s2)
        if not self.disable_csca:
            c2 = self.csca2(m2, d3_at_s2)
        else:
            c2 = m2
        z = self.decoder2(torch.cat([d3_at_s2, c2], dim=1))
        # CSCA1 merges the MSPE1 skip with projected D2 at the S1 scale.
        d2_at_s1 = self.resize(z, s1)
        if not self.disable_csca:
            c1 = self.csca1(m1, self.csca1_context(d2_at_s1))
        else:
            c1 = m1
        z = self.decoder1(torch.cat([d2_at_s1, c1], dim=1))
        z = self.decoder0(F.interpolate(z, size=input_size, mode="bilinear", align_corners=False))
        logits = self.head(z)
        return logits if self.return_logits else torch.sigmoid(logits)


TrackNet = TrackNetV2B0MSPECSCA
