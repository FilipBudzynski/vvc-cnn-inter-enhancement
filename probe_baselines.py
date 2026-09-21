"""Forensic probe for the two reimplemented baselines (Bi-ConvLSTM, STENet-2024)
that collapse to identity on Y during full training.
"""

import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F

from enhancer.dataset_blackfyre import BlackfyreDataset
from enhancer.models.bi_conv_lstm import BiConvLSTMEnhancer
from enhancer.models.stenet_2024 import STENet2024
from train_martell_hybrid import y_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiCfg:
    base_channels = 24
    kernel_size = 5
    cnn_layers = 5


class SteCfg:
    base_channels = 64
    metadata_channels = 9


def get_batch(n=8):
    torch.manual_seed(42)
    import random
    random.seed(123)
    ds = BlackfyreDataset("data/precomputed_martell", patch_size=132, split="val")
    idxs = np.linspace(0, len(ds) - 1, n).astype(int)
    prevs, currs, nxts, origs, metas = [], [], [], [], []
    for i in idxs:
        (p, c, nx), o, feat, _ = ds[int(i)]
        prevs.append(p); currs.append(c); nxts.append(nx); origs.append(o); metas.append(feat)
    return (torch.stack(prevs).to(DEVICE), torch.stack(currs).to(DEVICE),
            torch.stack(nxts).to(DEVICE), torch.stack(origs).to(DEVICE),
            torch.stack(metas).to(DEVICE))


def fwd(model, name, prev, curr, nxt, meta):
    out = model(curr, prev, nxt, meta)
    if isinstance(out, tuple):
        return out
    return out, None


def psnr(a, b):
    return 10 * np.log10(1.0 / max(F.mse_loss(a, b).item(), 1e-12))


def dY(out, curr, orig):
    return (psnr(out[:, 0:1], orig[:, 0:1]) - psnr(curr[:, 0:1], orig[:, 0:1]))


def report_forward(model, name, batch, tag):
    prev, curr, nxt, orig, meta = batch
    model.eval()
    with torch.no_grad():
        out, synth = fwd(model, name, prev, curr, nxt, meta)
        res = out - curr
        print(f"[{tag}] {name}: mean|res| Y={res[:,0].abs().mean():.2e} "
              f"U={res[:,1].abs().mean():.2e} V={res[:,2].abs().mean():.2e}")
        # 8-bit identity on Y
        same = (torch.round(out[:, 0] * 255) == torch.round(curr[:, 0] * 255)).float().mean()
        print(f"[{tag}] {name}: Y pixels byte-identical to input: {same*100:.2f}%")
        print(f"[{tag}] {name}: dPSNR vs input  Y={dY(out,curr,orig):+.4f} dB  "
              f"U={psnr(out[:,1:2],orig[:,1:2])-psnr(curr[:,1:2],orig[:,1:2]):+.4f}  "
              f"V={psnr(out[:,2:3],orig[:,2:3])-psnr(curr[:,2:3],orig[:,2:3]):+.4f}")
        # temporal dependence: replace nxt with zeros
        out2, _ = fwd(model, name, prev, torch.zeros_like(nxt) if False else curr, nxt * 0, meta)
        print(f"[{tag}] {name}: |out(nxt=0) - out| = {(out2-out).abs().mean():.2e} "
              f"(>0 means temporal path is wired)")
        if synth is not None:
            print(f"[{tag}] {name}: synth dPSNR-Y vs input = {dY(synth,curr,orig):+.4f} dB, "
                  f"|synth-curr| Y = {(synth-curr)[:,0].abs().mean():.2e}")


def gate_stats(model, batch):
    """LSTM gate saturation: capture pre-activations of fwd cell conv."""
    prev, curr, nxt, orig, meta = batch
    captured = []
    h = model.lstm_fwd.conv.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    with torch.no_grad():
        model(curr, prev, nxt, meta)
    h.remove()
    g = torch.cat(captured, 0)
    c = g.shape[1] // 4
    i_g, f_g, o_g, _ = torch.split(g, c, dim=1)
    i_s, f_s, o_s = torch.sigmoid(i_g), torch.sigmoid(f_g + 1.0), torch.sigmoid(o_g)
    for nm, s in [("i", i_s), ("f", f_s), ("o", o_s)]:
        sat = ((s < 0.01) | (s > 0.99)).float().mean()
        print(f"  gate {nm}: mean={s.mean():.3f}  saturated(<1% or >99%)={sat*100:.1f}%")


def own_loss(name, model, batch):
    prev, curr, nxt, orig, meta = batch
    if name == "stenet":
        enh, synth = model(curr, prev, nxt, meta)
        return F.mse_loss(enh, orig) + 0.1 * F.mse_loss(synth, orig), enh
    enh = model(curr, prev, nxt, None)
    return F.mse_loss(enh, orig), enh


def hybrid_loss(name, model, batch):
    prev, curr, nxt, orig, meta = batch
    if name == "stenet":
        enh, synth = model(curr, prev, nxt, meta)
        aux = 0.1 * F.mse_loss(synth, orig)
    else:
        enh = model(curr, prev, nxt, None)
        aux = 0.0
    ly = y_loss(enh[:, 0:1], orig[:, 0:1])
    luv = F.mse_loss(enh[:, 1:3], orig[:, 1:3])
    return ly + luv + aux, enh


def report_grads(name, make_model, batch):
    torch.manual_seed(42)
    model = make_model().to(DEVICE)
    model.train()
    loss, _ = own_loss(name, model, batch)
    loss.backward()
    print(f"grad probe ({name}, own loss={loss.item():.6f}):")
    wd = 1e-4
    key = {"bi": ["decoder.net.8", "lstm_fwd.conv", "lstm_bwd.conv", "encoder.net.0"],
           "stenet": ["pfe.out", "rfs.out", "pfe.head", "rfs.head"]}[name]
    for n, p in model.named_parameters():
        if any(n.startswith(k) for k in key) and n.endswith("weight"):
            gn = p.grad.norm().item()
            wn = p.norm().item()
            print(f"  {n:30s} ||grad||={gn:.3e}  wd*||w||={wd*wn:.3e}  ratio grad/wd={gn/(wd*wn):.2f}")
    if name == "stenet":
        # split loss terms
        for term in ["main", "synth"]:
            m2 = make_model().to(DEVICE)
            m2.load_state_dict(model.state_dict())
            prev, curr, nxt, orig, meta = batch
            enh, synth = m2(curr, prev, nxt, meta)
            l = F.mse_loss(enh, orig) if term == "main" else 0.1 * F.mse_loss(synth, orig)
            l.backward()
            gp = m2.pfe.out.weight.grad
            gr = m2.rfs.out.weight.grad
            print(f"  term={term}: ||grad pfe.out||={0 if gp is None else gp.norm().item():.3e}  "
                  f"||grad rfs.out||={0 if gr is None else gr.norm().item():.3e}")


def overfit(name, make_model, batch, loss_fn, lr, wd, steps, label):
    torch.manual_seed(42)
    model = make_model().to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    prev, curr, nxt, orig, meta = batch
    log = []
    for s in range(steps + 1):
        opt.zero_grad()
        loss, enh = loss_fn(name, model, batch)
        if s < steps:
            loss.backward()
            opt.step()
        if s % 50 == 0 or s == steps:
            log.append((s, loss.item(), dY(enh.detach(), curr, orig)))
    head = f"{name} [{label}] lr={lr:g} wd={wd:g}"
    curve = "  ".join(f"s{s}:{l:.5f}/{d:+.3f}dB" for s, l, d in log)
    print(f"{head}\n    loss/dY-> {curve}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["bi", "stenet"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--stages", nargs="+", default=["wiring", "grads", "overfit"])
    args = ap.parse_args()

    batch = get_batch(8)
    prev, curr, nxt, orig, meta = batch
    print(f"batch: {curr.shape}  input PSNR Y={psnr(curr[:,0:1],orig[:,0:1]):.3f} dB  "
          f"U={psnr(curr[:,1:2],orig[:,1:2]):.3f}  V={psnr(curr[:,2:3],orig[:,2:3]):.3f}")

    makers = {"bi": lambda: BiConvLSTMEnhancer(BiCfg()),
              "stenet": lambda: STENet2024(SteCfg())}
    ckpts = {"bi": "checkpoints/bi_conv_lstm_best.pt",
             "stenet": "checkpoints/stenet_2024_best.pt"}

    for name in args.models:
        print("=" * 78)
        if "wiring" in args.stages:
            torch.manual_seed(42)
            m = makers[name]().to(DEVICE)
            report_forward(m, name, batch, "init")
            if name == "bi":
                gate_stats(m, batch)
            try:
                m.load_state_dict(torch.load(ckpts[name], map_location=DEVICE, weights_only=True))
                report_forward(m, name, batch, "best.pt")
                if name == "bi":
                    gate_stats(m, batch)
            except FileNotFoundError:
                print(f"no checkpoint {ckpts[name]}")
            del m
        if "grads" in args.stages:
            report_grads(name, makers[name], batch)
        if "overfit" in args.stages:
            for label, fn, lr, wd in [("A own", own_loss, 1e-4, 1e-4),
                                      ("B own", own_loss, 1e-4, 0.0),
                                      ("C own", own_loss, 1e-3, 1e-4),
                                      ("D own", own_loss, 1e-3, 0.0),
                                      ("E hybrid", hybrid_loss, 1e-4, 1e-4)]:
                overfit(name, makers[name], batch, fn, lr, wd, args.steps, label)


if __name__ == "__main__":
    main()
