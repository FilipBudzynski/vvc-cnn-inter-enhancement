"""QG-ConvLSTM (Quality-Gated ConvLSTM) enhancer, Yang et al., ICME 2019
(arXiv:1903.04596) adapted to VVC.
"""

import torch
import torch.nn as nn


class QGConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 quality_channels: int, kernel_size: int = 5):
        super().__init__()
        self.hidden_channels = hidden_channels
        p = kernel_size // 2
        self.conv_xh = nn.Conv2d(in_channels + hidden_channels,
                                 4 * hidden_channels, kernel_size, padding=p)
        # Separate quality projection feeds only i and f gates (paper Eq 5-6)
        self.conv_q  = nn.Conv2d(quality_channels,
                                 2 * hidden_channels, kernel_size, padding=p)

    def forward(self, x, q, state):
        h, c = state
        gates_xh = self.conv_xh(torch.cat([x, h], dim=1))
        i_xh, f_xh, o_xh, g_xh = torch.split(gates_xh, self.hidden_channels, dim=1)
        i_q, f_q = torch.split(self.conv_q(q), self.hidden_channels, dim=1)
        i = torch.sigmoid(i_xh + i_q)
        f = torch.sigmoid(f_xh + f_q + 1.0)   # forget bias = 1.0
        o = torch.sigmoid(o_xh)
        g = torch.tanh(g_xh)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

    def init_state(self, b, h, w, device, dtype):
        z = torch.zeros(b, self.hidden_channels, h, w, device=device, dtype=dtype)
        return (z, z.clone())


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, padding=p)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TimeDistributedStack(nn.Module):
    """Time-distributed BN+ReLU conv stack."""

    def __init__(self, c_in, c_hidden, c_out, num_layers=5, k=5, final_zero_init=False):
        super().__init__()
        layers = [ConvBNReLU(c_in, c_hidden, k)]
        for _ in range(num_layers - 2):
            layers.append(ConvBNReLU(c_hidden, c_hidden, k))
        final = nn.Conv2d(c_hidden, c_out, k, padding=k//2)
        if final_zero_init:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        b, t, c, h, w = x.shape
        y = self.net(x.reshape(b * t, c, h, w))
        return y.view(b, t, y.shape[1], h, w)


def _scan_qg(cell, seq, q, reverse=False):
    b, t, c, h, w = seq.shape
    state = cell.init_state(b, h, w, seq.device, seq.dtype)
    outs = []
    rng = range(t - 1, -1, -1) if reverse else range(t)
    for ti in rng:
        h_out, state = cell(seq[:, ti], q, state)
        outs.append(h_out)
    if reverse:
        outs = outs[::-1]
    return torch.stack(outs, dim=1)


class QGConvLSTMEnhancer(nn.Module):
    """
    Args (config):
      base_channels: hidden width (paper = 24; default 64 for capacity parity)
      kernel_size: conv kernel (paper = 5)
      cnn_layers: depth of encoder/decoder (paper = 5)
      metadata_channels: VVC decoder metadata channels (we have 9)
      quality_embed: quality-feature embedding width (paper-equivalent
                     to BRISQUE -> 16-dim projection)
    """

    def __init__(self, config):
        super().__init__()
        c     = getattr(config, "base_channels", 64)
        k     = getattr(config, "kernel_size", 5)
        L     = getattr(config, "cnn_layers", 5)
        meta  = getattr(config, "metadata_channels", 9)
        q_dim = getattr(config, "quality_embed", 16)

        # Per-frame encoder
        self.encoder = TimeDistributedStack(3, c, c, num_layers=L, k=k)
        # Metadata -> quality feature
        self.meta_encoder = nn.Sequential(
            nn.Conv2d(meta, q_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(q_dim, q_dim, 3, padding=1), nn.ReLU(inplace=True),
        )
        # Bi-QG-ConvLSTM
        self.lstm_fwd = QGConvLSTMCell(c, c, q_dim, kernel_size=k)
        self.lstm_bwd = QGConvLSTMCell(c, c, q_dim, kernel_size=k)
        self.decoder = TimeDistributedStack(2 * c, c, 3,
                                            num_layers=L, k=k,
                                            final_zero_init=True)

    def forward(self, curr, prev, nxt, metadata):
        seq  = torch.stack([prev, curr, nxt], dim=1)         # [B, 3, 3, H, W]
        q    = self.meta_encoder(metadata)                    # [B, q_dim, H, W]
        feat = self.encoder(seq)                              # [B, 3, c, H, W]
        fwd  = _scan_qg(self.lstm_fwd, feat, q, reverse=False)
        bwd  = _scan_qg(self.lstm_bwd, feat, q, reverse=True)
        merged   = torch.cat([fwd, bwd], dim=2)               # [B, 3, 2c, H, W]
        residual = self.decoder(merged)                       # [B, 3, 3, H, W]
        out = curr + residual[:, 1]
        return out.clamp(0, 1) if not self.training else out


if __name__ == "__main__":
    class Cfg:
        base_channels = 64; kernel_size = 5; cnn_layers = 5
        metadata_channels = 9; quality_embed = 16
    m = QGConvLSTMEnhancer(Cfg())
    print(f"QGConvLSTM params: {sum(p.numel() for p in m.parameters()):,}")
    prev = torch.randn(2, 3, 132, 132); curr = torch.randn(2, 3, 132, 132)
    nxt  = torch.randn(2, 3, 132, 132); meta = torch.randn(2, 9, 132, 132)
    out = m(curr, prev, nxt, meta)
    print(f"output: {out.shape}")
    # Verify exact-identity at init
    m.eval()
    diff = (m(curr.clamp(0,1), prev.clamp(0,1), nxt.clamp(0,1), meta) - curr.clamp(0,1)).abs().max()
    print(f"max |output - curr| at init (should be ~0): {diff.item():.6e}")
