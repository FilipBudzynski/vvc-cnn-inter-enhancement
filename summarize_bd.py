"""
Aggregate per-model bdrate JSONs into a single comparison markdown table.

Usage:
    python summarize_bd.py bdrate_results/martell.json bdrate_results/vvc_ppff.json \\
        [bdrate_results/snow_wide.json] -o BDRATE_RESULTS.md
"""

import argparse
import json
from pathlib import Path


def fmt(x: float, unit: str = "") -> str:
    if x is None or (isinstance(x, float) and (x != x)):  # NaN check
        return "-"
    sign = "+" if x > 0 and unit == "dB" else ""
    return f"{sign}{x:.3f}{unit}" if unit == "dB" else f"{x:+.2f}{unit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("BDRATE_RESULTS.md"))
    args = ap.parse_args()

    runs = []
    for p in args.results:
        with open(p) as f:
            runs.append(json.load(f))

    md = ["# BD-Rate / BD-PSNR Results", ""]
    md.append(f"**Test set ({len(runs[0]['videos'])} unbiased Xiph sequences):** "
              f"{', '.join(runs[0]['videos'])}")
    md.append(f"**QPs:** {runs[0]['qps']}")
    md.append("")

    md.append("## BD metrics (lower BD-rate is better; higher BD-PSNR is better)")
    md.append("")
    md.append("| Model | Y BD-PSNR (dB) | Y BD-Rate (%) | U BD-PSNR (dB) | U BD-Rate (%) | V BD-PSNR (dB) | V BD-Rate (%) |")
    md.append("|---|---|---|---|---|---|---|")
    for r in runs:
        bd = r["summary"]["bd"]
        row = [r["model"]]
        for ch in "YUV":
            row.append(f"{bd[ch]['bd_psnr_db']:+.4f}")
            row.append(f"{bd[ch]['bd_rate_pct']:+.3f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    # Per-QP RD points
    for r in runs:
        md.append(f"## {r['model']}: per-QP RD points (avg over videos)")
        md.append("")
        md.append("| QP | Bitrate (kbps) | Anchor Y/U/V (dB) | Enhanced Y/U/V (dB) | ΔY (dB) | ΔU (dB) | ΔV (dB) |")
        md.append("|---|---|---|---|---|---|---|")
        for qp in r["qps"]:
            pq = r["summary"]["per_qp"][str(qp)]
            br = pq["bitrate_kbps"]
            a = pq["psnr_anchor"]
            e = pq["psnr_enhanced"]
            anchor_str = f"{a['Y']:.2f} / {a['U']:.2f} / {a['V']:.2f}"
            enh_str = f"{e['Y']:.2f} / {e['U']:.2f} / {e['V']:.2f}"
            dy = e["Y"] - a["Y"]
            du = e["U"] - a["U"]
            dv = e["V"] - a["V"]
            md.append(f"| {qp} | {br:.1f} | {anchor_str} | {enh_str} | "
                      f"{dy:+.3f} | {du:+.3f} | {dv:+.3f} |")
        md.append("")

    # Per-video summary (Y BD-rate only, to keep table compact)
    md.append("## Per-video Y BD-Rate (%) by model")
    md.append("")
    headers = ["Video"] + [r["model"] for r in runs]
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    videos = runs[0]["videos"]
    for v in videos:
        row = [v]
        for r in runs:
            pv = r["per_video"].get(v, {})
            # Build per-video RD for Y only and compute BD-rate
            try:
                from bdrate import bd_rate
                # JSON keys are strings; per-QP entries may also have an "error" dict.
                valid = {int(k): v for k, v in pv.items()
                         if isinstance(v, dict) and "bitrate_kbps" in v}
                qps = sorted(valid.keys())
                if len(qps) >= 4:
                    rates = [valid[qp]["bitrate_kbps"] for qp in qps]
                    pa = [valid[qp]["psnr_anchor_interior"]["Y"] for qp in qps]
                    pe = [valid[qp]["psnr_enhanced"]["Y"] for qp in qps]
                    val = bd_rate(rates, pa, rates, pe)
                    row.append(f"{val:+.2f}")
                else:
                    row.append("-")
            except Exception as e:
                row.append(f"err: {e}")
        md.append("| " + " | ".join(row) + " |")

    args.out.write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
