"""Bjontegaard BD-PSNR and BD-Rate (G. Bjontegaard, VCEG-M33, 2001)."""

import numpy as np


def _cubic_avg(x: np.ndarray, y: np.ndarray) -> float:
    """Average y over [x.min(), x.max()] via cubic poly fit + analytic integral."""
    coeffs = np.polyfit(x, y, 3)
    integral = np.polyint(coeffs)
    return (np.polyval(integral, x.max()) - np.polyval(integral, x.min())) / (x.max() - x.min())


def _validate(rates, psnrs):
    rates = np.asarray(rates, dtype=np.float64)
    psnrs = np.asarray(psnrs, dtype=np.float64)
    if rates.shape != psnrs.shape or rates.ndim != 1:
        raise ValueError("rates/psnrs must be 1-D arrays of equal length")
    if len(rates) < 4:
        raise ValueError("need at least 4 RD points for cubic Bjontegaard")
    return rates, psnrs


def bd_psnr(rates_a, psnrs_a, rates_b, psnrs_b) -> float:
    """
    Average ΔPSNR (dB) of curve B vs anchor A over their common log-rate range.
    Positive = B is better.
    """
    ra, pa = _validate(rates_a, psnrs_a)
    rb, pb = _validate(rates_b, psnrs_b)

    log_ra, log_rb = np.log10(ra), np.log10(rb)
    lo = max(log_ra.min(), log_rb.min())
    hi = min(log_ra.max(), log_rb.max())
    if hi <= lo:
        return float("nan")

    # Fit PSNR as a cubic in log_rate, integrate over [lo, hi]
    pa_int = np.polyint(np.polyfit(log_ra, pa, 3))
    pb_int = np.polyint(np.polyfit(log_rb, pb, 3))
    avg_a = (np.polyval(pa_int, hi) - np.polyval(pa_int, lo)) / (hi - lo)
    avg_b = (np.polyval(pb_int, hi) - np.polyval(pb_int, lo)) / (hi - lo)
    return float(avg_b - avg_a)


def bd_rate(rates_a, psnrs_a, rates_b, psnrs_b) -> float:
    """
    Average percentage rate change of curve B vs anchor A over common PSNR range.
    Negative = B uses less rate at the same PSNR (improvement).
    """
    ra, pa = _validate(rates_a, psnrs_a)
    rb, pb = _validate(rates_b, psnrs_b)

    log_ra, log_rb = np.log10(ra), np.log10(rb)
    lo = max(pa.min(), pb.min())
    hi = min(pa.max(), pb.max())
    if hi <= lo:
        return float("nan")

    # Fit log_rate as a cubic in PSNR, integrate over [lo, hi]
    ra_int = np.polyint(np.polyfit(pa, log_ra, 3))
    rb_int = np.polyint(np.polyfit(pb, log_rb, 3))
    avg_a = (np.polyval(ra_int, hi) - np.polyval(ra_int, lo)) / (hi - lo)
    avg_b = (np.polyval(rb_int, hi) - np.polyval(rb_int, lo)) / (hi - lo)
    return float((10 ** (avg_b - avg_a) - 1) * 100)


if __name__ == "__main__":
    # Sanity check: same curve -> 0 dB BD-PSNR, 0% BD-rate.
    r = [100, 200, 400, 800, 1600]
    p = [30.0, 33.0, 36.0, 39.0, 42.0]
    assert abs(bd_psnr(r, p, r, p)) < 1e-9
    assert abs(bd_rate(r, p, r, p)) < 1e-9

    # Test curve uniformly +1 dB -> BD-PSNR ~ +1 dB
    p2 = [v + 1.0 for v in p]
    assert abs(bd_psnr(r, p, r, p2) - 1.0) < 1e-6, bd_psnr(r, p, r, p2)
    print(f"smoke ok: BD-PSNR(+1dB)={bd_psnr(r,p,r,p2):.4f}, "
          f"BD-Rate(+1dB)={bd_rate(r,p,r,p2):.2f}%")
