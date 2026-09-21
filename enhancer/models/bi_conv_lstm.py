"""Bi-directional ConvLSTM enhancer: published-LSTM baseline for VVC
post-filter comparison against Martell.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """Standard ConvLSTM (Shi et al., NeurIPS 2015): i, f, o, g gates."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 5):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels + hidden_channels,
                              4 * hidden_channels,
                              kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + 1.0)  # forget bias = 1.0 (as in paper / TF default)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

    def init_state(self, batch: int, h: int, w: int, device, dtype):
        z = torch.zeros(batch, self.hidden_channels, h, w, device=device, dtype=dtype)
        return (z, z.clone())


class TimeDistributedCNN(nn.Module):
    """Apply the same conv stack independently to each time step."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 num_layers: int = 5, kernel_size: int = 5, output_relu: bool = True):
        super().__init__()
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_channels, hidden, kernel_size, padding=padding), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(hidden, hidden, kernel_size, padding=padding), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(hidden, out_channels, kernel_size, padding=padding)]
        if output_relu and out_channels == hidden:
            layers += [nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W] -> [B*T, C, H, W] -> conv -> reshape back
        b, t, c, h, w = x.shape
        x = self.net(x.reshape(b * t, c, h, w))
        return x.view(b, t, x.shape[1], h, w)


def _scan(cell: ConvLSTMCell, seq: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """Run ConvLSTM over the time axis of a [B, T, C, H, W] sequence."""
    b, t, _, h, w = seq.shape
    state = cell.init_state(b, h, w, seq.device, seq.dtype)
    outs = []
    rng = range(t - 1, -1, -1) if reverse else range(t)
    for i in rng:
        out, state = cell(seq[:, i], state)
        outs.append(out)
    if reverse:
        outs = outs[::-1]
    return torch.stack(outs, dim=1)


class BiConvLSTMEnhancer(nn.Module):
    """
    Args (config):
      base_channels: hidden width (paper = 24)
      kernel_size: conv kernel (paper = 5)
      cnn_layers: depth of encoder/decoder CNN stacks (paper = 5)
    """

    def __init__(self, config):
        super().__init__()
        c = getattr(config, "base_channels", 24)
        k = getattr(config, "kernel_size", 5)
        layers = getattr(config, "cnn_layers", 5)

        # Encoder: per-frame CNN, 3-ch in -> c-ch
        self.encoder = TimeDistributedCNN(3, c, c, num_layers=layers,
                                          kernel_size=k, output_relu=True)
        # Bi-ConvLSTM
        self.lstm_fwd = ConvLSTMCell(c, c, kernel_size=k)
        self.lstm_bwd = ConvLSTMCell(c, c, kernel_size=k)
        # Decoder: per-frame CNN, 2c-ch (fwd+bwd) -> 3-ch
        self.decoder = TimeDistributedCNN(2 * c, c, 3, num_layers=layers,
                                          kernel_size=k, output_relu=False)

    def forward(self, curr, prev, nxt, metadata=None):
        # Stack as a 3-frame temporal sequence
        seq = torch.stack([prev, curr, nxt], dim=1)  # [B, 3, 3, H, W]

        encoded = self.encoder(seq)            # [B, 3, c, H, W]
        fwd = _scan(self.lstm_fwd, encoded, reverse=False)
        bwd = _scan(self.lstm_bwd, encoded, reverse=True)
        merged = torch.cat([fwd, bwd], dim=2)  # [B, 3, 2c, H, W]
        residual = self.decoder(merged)        # [B, 3, 3, H, W]

        # Extract the middle (current-frame) output and add residual to curr
        return (curr + residual[:, 1]).clamp(0, 1)


if __name__ == "__main__":
    class Cfg:
        base_channels = 24
        kernel_size = 5
        cnn_layers = 5

    m = BiConvLSTMEnhancer(Cfg())
    print(f"BiConvLSTMEnhancer params: {sum(p.numel() for p in m.parameters()):,}")
    prev = torch.randn(2, 3, 132, 132)
    curr = torch.randn(2, 3, 132, 132)
    nxt = torch.randn(2, 3, 132, 132)
    y = m(curr, prev, nxt, None)
    print(f"output {y.shape}")
