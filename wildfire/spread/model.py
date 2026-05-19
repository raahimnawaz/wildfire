"""
ConvLSTM next-day wildfire spread forecaster.

Architecture follows Shi et al. (2015) "Convolutional LSTM Network: A Machine
Learning Approach for Precipitation Nowcasting" — generalized for fire spread
as recommended by the project's research doc:

  Input  : (B, T, C, H, W)   T-day stack of feature maps + previous fire mask
  Output : (B, 1, H, W)      next-day per-pixel fire-presence logits

The convolutional gates preserve spatial context lost by a vanilla LSTM, and
stacking layers builds receptive field for fire-front propagation.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .config import SPREAD_CFG


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM2d cell with peephole-free gates."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        # Compute all 4 gates in one fused conv (input + hidden concatenated on channel dim)
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.hidden_channels = hidden_channels

    def forward(
        self,
        x: torch.Tensor,                # (B, C_in, H, W)
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch: int, height: int, width: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(batch, self.hidden_channels, height, width, device=device, dtype=dtype)
        return zeros, zeros.clone()


class ConvLSTMStack(nn.Module):
    """Stacked ConvLSTM cells unrolled over T timesteps."""

    def __init__(self, in_channels: int, hidden_channels: List[int], kernel_size: int = 3):
        super().__init__()
        self.cells = nn.ModuleList()
        prev = in_channels
        for h in hidden_channels:
            self.cells.append(ConvLSTMCell(prev, h, kernel_size=kernel_size))
            prev = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) -> (B, hidden_channels[-1], H, W) using only the final timestep."""
        b, t, _c, h_, w_ = x.shape
        states = [cell.init_state(b, h_, w_, x.device, x.dtype) for cell in self.cells]
        last_h = None
        for step in range(t):
            inp = x[:, step]
            for layer_idx, cell in enumerate(self.cells):
                h, c = cell(inp, states[layer_idx])
                states[layer_idx] = (h, c)
                inp = h
            last_h = inp
        assert last_h is not None
        return last_h


class ConvLSTMSpreadModel(nn.Module):
    """ConvLSTM stack + 1x1 conv head producing per-pixel logits."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Tuple[int, ...] = SPREAD_CFG.hidden_channels,
        kernel_size: int = SPREAD_CFG.kernel_size,
        use_batch_norm: bool = SPREAD_CFG.use_batch_norm,
    ):
        super().__init__()
        self.encoder = ConvLSTMStack(in_channels, list(hidden_channels), kernel_size=kernel_size)
        head_in = hidden_channels[-1]
        layers: List[nn.Module] = []
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(head_in))
        layers += [
            nn.Conv2d(head_in, head_in // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in // 2, 1, kernel_size=1),
        ]
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) -> logits (B, 1, H, W)."""
        feat = self.encoder(x)
        return self.head(feat)


def build_default_model(in_channels: int) -> ConvLSTMSpreadModel:
    return ConvLSTMSpreadModel(
        in_channels=in_channels,
        hidden_channels=SPREAD_CFG.hidden_channels,
        kernel_size=SPREAD_CFG.kernel_size,
        use_batch_norm=SPREAD_CFG.use_batch_norm,
    )


class UNetSpreadModel(nn.Module):
    """
    U-Net baseline matching the architecture used by the canonical Next Day
    Wildfire Spread paper (Huot et al., ICDM 2022). Built on top of
    segmentation_models.pytorch — battle-tested encoders, easy to swap
    backbones (resnet18/34/50, efficientnet, mit_b0, etc.).

    Wraps the SMP U-Net so it accepts the same (B, T, C, H, W) tensor shape
    as ConvLSTMSpreadModel. We reduce the time dimension by stacking T
    frames along the channel axis — equivalent to the original NDWS setup
    where T=1 per training example.
    """

    def __init__(self, in_channels: int, lookback_days: int,
                 encoder_name: str = "resnet18",
                 encoder_weights: str | None = None):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "segmentation-models-pytorch is required for UNetSpreadModel. "
                "Install with: pip install segmentation-models-pytorch"
            ) from e

        self.lookback_days = lookback_days
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels * lookback_days,
            classes=1,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        flat = x.reshape(b, t * c, h, w)
        return self.unet(flat)


def build_model(architecture: str, in_channels: int, lookback_days: int) -> nn.Module:
    """Factory dispatch on architecture name."""
    arch = architecture.lower()
    if arch == "convlstm":
        return build_default_model(in_channels)
    if arch == "unet":
        return UNetSpreadModel(in_channels=in_channels, lookback_days=lookback_days)
    raise ValueError(f"Unknown architecture '{architecture}'. Use 'convlstm' or 'unet'.")
