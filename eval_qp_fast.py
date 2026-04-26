#!/usr/bin/env python3
"""Szybka ewaluacja"""
import os, re, numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from pytorch_msssim import ssim
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_yuv(f, w, h, n=10):
    frames = []
    for _ in range(n):
        d = f.read(w*h*3//2)
        if len(d) < w*h*3//2: break
        y = np.frombuffer(d[:w*h], dtype=np.uint8).reshape(h,w).astype(np.float32)/255
        u = np.frombuffer(d[w*h:w*h+w*h//4], dtype=np.uint8).reshape(h//2,w//2).astype(np.float32)/255
        v = np.frombuffer(d[w*h+w*h//4:], dtype=np.uint8).reshape(h//2,w//2).astype(np.float32)/255
        frames.append(np.stack([y, np.repeat(np.repeat(u,2,0),2,1), np.repeat(np.repeat(v,2,0),2,1)]))
    return frames

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3,64,7,padding=3),nn.BatchNorm2d(64),nn.PReLU())
        self.b = nn.Sequential(*[nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.PReLU(),nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.PReLU()) for _ in range(4)])
        self.o = nn.Conv2d(64,3,3,padding=1)
    def forward(self,x): return x+self.o(self.b(self.f(x)))

print(f"Device: {DEVICE}")

models = {}
models["ResNet"] = Net()
try:
    ckpt = torch.load("experiments/enhancer/vtm_resnet_v6.pth", map_location=DEVICE)
    models["ResNet"].load_state_dict(ckpt, strict=False)
except: pass

try:
    from enhancer.models.snow import SnowEnhancer
    class C: base_channels=64; metadata_channels=19
    models["Snow"] = SnowEnhancer(C()).to(DEVICE)
    models["Snow"].load_state_dict(torch.load("checkpoints/snow_epoch_490.pt", map_location=DEVICE))
except: pass

try:
    from enhancer.models.snow_wide import SnowWideEnhancer
    class C: base_channels=64; metadata_channels=19
    models["Snow_Wide"] = SnowWideEnhancer(C()).to(DEVICE)
    models["Snow_Wide"].load_state_dict(torch.load("checkpoints/snow_wide_epoch_460.pt", map_location=DEVICE))
except: pass

for n,m in models.items(): m.eval().to(DEVICE); print(f"{n}: {sum(p.numel()for p in m.parameters()):,}")

results = {}
for qp in [22,27,32,37,42]:
    print(f"\n=== QP={qp} ===")
    recs = sorted(Path(f"output_qp{qp}/encoded").glob("*_rec.yuv"))[:10]
    psnr_gains = {n:[] for n in models}
    base_psnr = []
    
    for r in recs:
        s = r.stem.replace(f"_QP{qp}_rec","")
        o = Path("data") / f"{s}.yuv"
        if not o.exists(): continue
        info = list(Path("data").glob(f"{s}*.info"))
        if not info: continue
        w = int(re.search(r"Width\s+:\s+(\d+)", info[0].read_text()).group(1))
        h = int(re.search(r"Height\s+:\s+(\d+)", info[0].read_text()).group(1))
        
        try:
            with open(r,"rb") as rf, open(o,"rb") as of:
                rec_frames = read_yuv(rf,w,h,5)
                orig_frames = read_yuv(of,w,h,5)
        except: continue
        
        for i in range(1,len(rec_frames)-1):
            curr = torch.from_numpy(rec_frames[i]).unsqueeze(0).to(DEVICE)
            orig = torch.from_numpy(orig_frames[i]).unsqueeze(0).to(DEVICE)
            
            mse = torch.nn.functional.mse_loss(curr,orig).item()
            b = 10*np.log10(1/(mse+1e-10))
            base_psnr.append(b)
            
            for n,m in models.items():
                with torch.no_grad():
                    if n in ["Snow","Snow_Wide"]:
                        prev = torch.from_numpy(rec_frames[i-1]).unsqueeze(0).to(DEVICE)
                        nxt = torch.from_numpy(rec_frames[i+1]).unsqueeze(0).to(DEVICE)
                        enh = m(curr,prev,nxt,torch.zeros(1,19,curr.shape[2],curr.shape[3],device=DEVICE)).clamp(0,1)
                    else:
                        enh = m(curr).clamp(0,1)
                e = 10*np.log10(1/(torch.nn.functional.mse_loss(enh,orig).item()+1e-10))
                psnr_gains[n].append(e-b)
    
    if base_psnr:
        results[qp] = {"baseline": np.mean(base_psnr), "gains": {n:np.mean(g) for n,g in psnr_gains.items() if g}}
        print(f"Baseline: {results[qp]['baseline']:.2f}dB")
        for n,g in results[qp]["gains"].items(): print(f"  {n}: {g:+.4f}dB")

with open("qp_final.json","w") as f: json.dump(results,f,indent=2)

md = "# Porównanie Modeli dla Różnych QP\n\n| QP | Baseline | "+" | ".join([f"{n}" for n in models])+" |\n|-----|----------|"+"|".join(["---" for _ in models])+"|\n"
for qp,r in results.items():
    md += f"| {qp} | {r['baseline']:.2f} | "+" | ".join([f"{r['gains'].get(n,0):+.4f}" for n in models])+" |\n"

with open("FINAL_QP_COMPARISON.md","w") as f: f.write(md)
print("\n"+md)
