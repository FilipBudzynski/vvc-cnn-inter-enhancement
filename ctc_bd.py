"""Bjontegaard helpers for the CTC-like comparison."""

import numpy as np
from scipy.interpolate import PchipInterpolator

from bdrate import bd_psnr as bd_psnr_cubic_raw, bd_rate as bd_rate_cubic_raw


def _pchip_avg(x, y, lo, hi) -> float:
    order = np.argsort(x)
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    f = PchipInterpolator(x, y)
    return float(f.integrate(lo, hi) / (hi - lo))


def bd_rate_pchip(rates_a, psnrs_a, rates_b, psnrs_b) -> float:
    la, lb = np.log10(rates_a), np.log10(rates_b)
    lo = max(min(psnrs_a), min(psnrs_b))
    hi = min(max(psnrs_a), max(psnrs_b))
    if hi <= lo:
        return float("nan")
    avg_a = _pchip_avg(psnrs_a, la, lo, hi)
    avg_b = _pchip_avg(psnrs_b, lb, lo, hi)
    return float((10 ** (avg_b - avg_a) - 1) * 100)


def bd_psnr_pchip(rates_a, psnrs_a, rates_b, psnrs_b) -> float:
    la, lb = np.log10(rates_a), np.log10(rates_b)
    lo = max(la.min(), lb.min())
    hi = min(la.max(), lb.max())
    if hi <= lo:
        return float("nan")
    return float(_pchip_avg(lb, psnrs_b, lo, hi) - _pchip_avg(la, psnrs_a, lo, hi))


def bd_all(rates_a, psnrs_a, rates_b, psnrs_b) -> dict:
    """Both interpolants, BD-rate (%) and BD-PSNR (dB). Needs >= 4 points."""
    ra, pa, rb, pb = (np.asarray(v, float) for v in (rates_a, psnrs_a, rates_b, psnrs_b))
    return {
        "cubic": {"bd_rate_pct": bd_rate_cubic_raw(ra, pa, rb, pb),
                  "bd_psnr_db": bd_psnr_cubic_raw(ra, pa, rb, pb)},
        "pchip": {"bd_rate_pct": bd_rate_pchip(ra, pa, rb, pb),
                  "bd_psnr_db": bd_psnr_pchip(ra, pa, rb, pb)},
    }


def bd_for_qpsets(rd_by_qp: dict, qpsets: dict) -> dict:
    """
    rd_by_qp: {qp: {"bitrate_kbps": r, "psnr_anchor": {Y,U,V}, "psnr_enhanced": {Y,U,V}}}
    qpsets:   {"qp4": [22,27,32,37], "qp5": [22,27,32,37,42]}
    Returns {qpset: {ch: bd_all(...)}} (qpsets with missing QPs are skipped).
    """
    out = {}
    for name, qps in qpsets.items():
        if any(q not in rd_by_qp for q in qps):
            continue
        qs = sorted(qps)
        rates = [rd_by_qp[q]["bitrate_kbps"] for q in qs]
        out[name] = {}
        for ch in "YUV":
            pa = [rd_by_qp[q]["psnr_anchor"][ch] for q in qs]
            pe = [rd_by_qp[q]["psnr_enhanced"][ch] for q in qs]
            out[name][ch] = bd_all(rates, pa, rates, pe)
    return out


if __name__ == "__main__":
    r = [100, 200, 400, 800, 1600]
    p = [30.0, 33.0, 36.0, 39.0, 42.0]
    p2 = [v + 1.0 for v in p]
    res = bd_all(r, p, r, p2)
    print(res)
    assert abs(res["pchip"]["bd_psnr_db"] - 1.0) < 1e-6
    assert abs(res["pchip"]["bd_rate_pct"] - res["cubic"]["bd_rate_pct"]) < 0.5
    print("smoke ok")
