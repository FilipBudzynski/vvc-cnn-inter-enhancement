"""Quick diagnostic probes for QG-ConvLSTM (analogous to probe_baselines.py):
single-batch overfit with the training recipe (MSE, wd=0) at lr 1e-4 and
1e-3, plus gradient norms per module and gate statistics at init.
Small GPU footprint; safe to run alongside the ongoing training."""
import torch
import torch.nn.functional as F

import probe_baselines as P
import evaluate_bd as E


def make_qg():
    class Cfg:
        base_channels = 64
        kernel_size = 5
        cnn_layers = 5
        metadata_channels = 9
        quality_embed = 16
    from enhancer.models.qg_conv_lstm import QGConvLSTMEnhancer
    return QGConvLSTMEnhancer(Cfg()).to(E.DEVICE)


def mse_loss(name, model, batch):
    prev, curr, nxt, orig, meta = batch
    out, _aux = P.fwd(model, "qg_conv_lstm", prev, curr, nxt, meta)
    return F.mse_loss(out, orig), out


def main():
    batch = P.get_batch(8)
    prev, curr, nxt, orig, meta = batch

    m = make_qg()
    with torch.no_grad():
        out, _ = P.fwd(m, "qg_conv_lstm", prev, curr, nxt, meta)
        res = (out - curr).abs()
        print(f"init: |res| mean={res.mean():.2e} max={res.max():.2e}  "
              f"dY(out)={P.dY(out, curr, orig):+.4f}")
    # gradient norms once
    loss, _ = mse_loss("qg", m, batch)
    loss.backward()
    for n, p in m.named_parameters():
        if p.grad is not None and n.endswith("weight") and p.dim() == 4:
            g = p.grad.norm().item()
            w = p.norm().item()
            if any(t in n for t in ["encoder.net.0", "encoder.net.4", "conv_xh",
                                    "conv_q", "decoder.net.0", "decoder.net.6",
                                    "meta_encoder.0"]):
                print(f"  grad {n:42s} ||g||={g:.2e} ||w||={w:.2e}")
    for lr in (1e-4, 1e-3):
        torch.manual_seed(0)
        P.overfit("qg_conv_lstm", make_qg, batch, mse_loss, lr=lr, wd=0.0,
                  steps=400, label=f"MSE lr={lr} wd=0")


if __name__ == "__main__":
    main()
