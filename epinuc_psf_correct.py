#!/usr/bin/env python3
"""Per-image PSF and drift measurement (and optional PSF homogenization) for EpiVision TIFs.

READ THIS FIRST -- the EpiVision TIF images are NOT defocused.

Measured on cross-channel-confirmed fiducial beads, each channel refined to its own centre, the
half-light radius r50 of the TIF channels matches or beats the ND2 reference:

    ND2  TIRF Cy3   r50 = 4.20 - 4.64      <- the broadest PSF in either dataset
    ND2  TIRF Cy5   r50 = 3.39 - 3.49
    ND2  TIRF 488   r50 = 2.84 - 2.98
    TIF  G / R / B  r50 = 2.75-4.08 / 3.09-4.12 / 2.24-4.30

So there is no blur to remove, and the correction path below should be left OFF for this data.
An earlier analysis concluded the R/B channels were defocused (sigma ~5.4, flat-topped, donut).
That was an artifact of measuring R/B at coordinates picked from the *green* channel: green
carries ~6500 nucleosome molecules per field which are dim or absent in R/B, and R/B are laterally
offset from G by 2-6 px within a cycle and up to ~20 px across cycles. Sampling a mostly-absent,
mis-centred source yields exactly a broad flat "profile". Refining each channel to its own bead
centre makes the donut vanish and the raw pixels show a compact peak (8740 counts at centre,
880 four px out, over a background of 263).

What IS real in this data:

* **Saturation.** ~14% of fiducial beads clip at the 16-bit ceiling (65535), and the brightest
  fields are worst (e.g. NUC388/ch1/pos051 clips 8 of 26 green beads; a single green frame can
  have ~380 clipped px). Green clips most, then B, then R. Clipped charge blooms *vertically*
  (CCD column bleed: of the heavily-saturated beads, 41% elongate near 90 deg vs 8% near 0 deg) --
  that vertical bleed is what reads as a "streak" on a bright bead. The clipped intensity is
  irrecoverably lost; nothing computational restores it. The fix is at acquisition (lower the
  reference-channel exposure / laser, or an HDR short exposure for the beads). Downstream, the
  right move is to *flag and mask* saturated wells so a clipped, bloomed bead does not corrupt
  background estimation, photometry, or the fiducial centroid. This module reports saturation per
  image and per bead so those frames can be handled explicitly.
* Genuinely streaky, high-aspect (12x) bright objects also exist, but they sit at a FIXED
  location across positions/cycles (e.g. ~(1205, 860)) -- detector/flowcell debris, not beads --
  and near the frame edges. The pipeline's ``ARTIFACT_MASKING`` is aimed at exactly this.
* **Between-cycle XY stage drift of up to ~20 px** (NUC388/ch1/pos051: R drifts 5.8 -> 15.6 ->
  19.7 px over cycles 1->3->5). Recovered here by offset voting on the beads. The pipeline handles
  this correctly *provided* bead-based registration succeeds.
* ``epinuc_colocalization.estimate_channel_transform`` marks every ``phase_correlation`` fallback
  ``success=True`` with no plausibility check on the shift magnitude, so a nonsense shift (up to
  1204 px on a 2048 px frame, seen in stale outputs) propagates silently into the registered
  coordinates and destroys colocalization at a 3 px radius.

This module is therefore a **measurement / QC tool**: run it with ``--dry-run`` for a per-image
report of saturation (px and beads), r50, bead count, recovered drift, and the instrument's own
focus flags (``PsdInPosition``, ``AttemptNumber``) parsed out of the XMP metadata. The
homogenization path is kept because it is correct and tested, not because this data needs it; it
would matter on a run that genuinely is defocused.

Correction method (when used): PSF *matching*, not deconvolution to a delta.

    K = conj(F[P]) * F[T] / (|F[P]|^2 + lam)      corrected = F^-1( F[img] * K )

with ``P`` the measured PSF and ``T`` the target. This is chosen deliberately:

* It is **linear**, so it preserves flux linearity. Richardson-Lucy is nonlinear and would
  invalidate the intensity calibration the downstream ``SPOT_DETECTION_SNR`` constants rely on.
  Because both P and T are normalized to unit sum, K(DC) = 1/(1+lam) ~ 1, so background and
  total flux survive.
* It is **far better conditioned**. A defocus OTF has genuine zeros; those frequencies are gone.
  Matching to a *finite* target only asks for frequencies the target also suppresses, instead of
  demanding the impossible.
* It **equalizes** the PSF across channels and cycles, so the pipeline's fixed sigma=1.2 spot
  filter and sigma=2.0 bead filter become appropriate for R/B again.

Honest limits:

* Homogenization creates no information. It cannot recover what the OTF zeros destroyed.
* It does not correct XY stage drift.
* Noise is amplified. The per-image ``noise_gain`` is measured and reported.

Green is passed through byte-identical by default (``--channels R B``): it is the pipeline's
registration reference.

There is also a **flat-field** path (``--flat-field``), independent of the PSF machinery. Unlike the
defocus story above, the per-mode fixed pattern IS real: vignetting up to ~47%, plus fixed optical
debris and hot pixels, different for R/G/B. It is removed by ``(frame - bias) / gain_norm`` where the
flat is the per-pixel median across FOVs and the bias is measured by photon transfer (transferred
from R/B for green, which has no background to measure it from). This corrects EVERY channel, green
included, and writes corrected TIFs; the pipeline is then pointed at the output tree, unchanged.

Usage
-----
    # what you almost certainly want for PSF QC: measure, write nothing
    python epinuc_psf_correct.py --in /Volumes/scBC/EpiVision/Images/NUC388 --dry-run

    # flat-field correction -- preview the per-mode bias/flat, write nothing
    python epinuc_psf_correct.py --in .../NUC388 --flat-field --dry-run

    # flat-field correction -- write corrected TIFs (every channel) to a mirrored tree
    python epinuc_psf_correct.py --in .../NUC388 --flat-field --out ./corrected/NUC388

    # reuse one run's calibration on another run (calibration lives in the sibling <out>_flatfield dir)
    python epinuc_psf_correct.py --in .../NUC389 --flat-field \
        --flat-load ./corrected/NUC388_flatfield --out ./corrected/NUC389

    # PSF homogenization path (not needed for this data)
    python epinuc_psf_correct.py --in ... --out ./corrected/NUC388

``--dry-run`` writes only the report CSV.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
    import tifffile
except ImportError:  # pragma: no cover
    sys.exit("The 'tifffile' package is required.  pip install tifffile")


# --------------------------------------------------------------------------------------
# Filename parsing -- mirrors epinuc_tiff_loader._TIF_RE so this file stays standalone.
# ch1_pos051_img0_Z1_R_cycle001_C001_000150_000.tif
# --------------------------------------------------------------------------------------
_TIF_RE = re.compile(
    r"^ch(?P<ch>\d+)_pos(?P<pos>\d+)_img(?P<img>\d+)_"
    r"(?P<base>N|Z\d+)_(?P<laser>[A-Za-z0-9]+)_cycle(?P<cycle>\d+)_"
    r"(?P<sample>C\d+)_(?P<frame>\d+)_(?P<dataindex>\d+)\.tif$",
    re.IGNORECASE,
)

_XMP_TAG = 700

#: 16-bit sensor ceiling.  Pixels here (or within a hair of it) are clipped and their true
#: intensity is unrecoverable; the ~99.9% figure catches near-ceiling wells that already bloomed.
_SATURATION_LEVEL = 65400


def parse_tif_name(fname: str) -> Optional[dict]:
    """Parse an EpiVision TIF basename into its fields, or None if it does not match."""
    m = _TIF_RE.match(os.path.basename(fname))
    if not m:
        return None
    g = m.groupdict()
    for k in ("ch", "pos", "img", "cycle", "frame", "dataindex"):
        g[k] = int(g[k])
    g["lane"] = f"ch{g['ch']}"
    g["laser"] = g["laser"].upper()
    return g


# --------------------------------------------------------------------------------------
# Robust IO -- the /Volumes/scBC SMB share drops reads of files that exist.
# --------------------------------------------------------------------------------------
_READ_RETRIES = 5
_READ_BACKOFF_S = 0.5


def read_tif(path: str) -> Tuple[np.ndarray, Optional[bytes]]:
    """Read a TIF as float32 plus its raw XMP bytes (None if absent), retrying on mount dropout."""
    import time

    last = None
    for attempt in range(_READ_RETRIES):
        try:
            with tifffile.TiffFile(path) as tf:
                page = tf.pages[0]
                arr = np.squeeze(np.asarray(page.asarray(), dtype=np.float32))
                xmp = None
                for tag in page.tags:
                    if tag.code == _XMP_TAG:
                        xmp = tag.value if isinstance(tag.value, bytes) else bytes(tag.value)
                        break
                return arr, xmp
        except (FileNotFoundError, OSError) as e:
            last = e
            time.sleep(_READ_BACKOFF_S * (attempt + 1))
    raise last


def xmp_field(xmp: Optional[bytes], key: str) -> Optional[str]:
    if not xmp:
        return None
    m = re.search(rf"<{key}>(.*?)</{key}>", xmp.decode("utf8", errors="replace"))
    return m.group(1) if m else None


class _Progress:
    """Minimal dependency-free progress bar.

    Renders an in-place bar on a TTY; when stdout is redirected (e.g. piped to a log) it falls back
    to a printed line every ~5% so the file stays readable.
    """

    def __init__(self, total: int, prefix: str = "", width: int = 34):
        self.total = int(total)
        self.prefix = prefix
        self.width = width
        self.tty = sys.stdout.isatty()
        self._last_step = -1

    def update(self, i: int) -> None:
        if self.total <= 0:
            return
        i = min(i, self.total)
        pct = int(100 * i / self.total)
        if self.tty:
            filled = int(self.width * i / self.total)
            bar = "#" * filled + "-" * (self.width - filled)
            end = "\n" if i >= self.total else ""
            print(f"\r{self.prefix}[{bar}] {i}/{self.total} {pct:3d}%", end=end, flush=True)
        else:
            step = pct // 5
            if step != self._last_step or i >= self.total:
                print(f"{self.prefix}{i}/{self.total} ({pct}%)", flush=True)
                self._last_step = step


# --------------------------------------------------------------------------------------
# Bead detection -- each image must find its own beads.
#
# Green exists only at cycle 1, so it cannot locate beads for later cycles, and the beads move
# ~20 px across cycles.  Note the background kernel: a median_filter(15) *eats* a sigma~6 bead
# (FWHM ~14 px), which silently biases both flux and sigma low.  A wide uniform_filter does not.
# --------------------------------------------------------------------------------------
def find_beads(img: np.ndarray, smooth: float = 3.0, top_n: int = 60,
               min_sep: int = 30, snr: float = 10.0) -> np.ndarray:
    """Return up to ``top_n`` bright, isolated bead candidates as an (N, 2) array of (y, x)."""
    bg = ndi.uniform_filter(img, 61)
    f = ndi.gaussian_filter(img - bg, smooth)
    med = float(np.median(f))
    noise = 1.4826 * float(np.median(np.abs(f - med))) or 1.0
    peaks = (ndi.maximum_filter(f, min_sep) == f) & (f > med + snr * noise)
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return np.empty((0, 2), float)
    order = np.argsort(f[ys, xs])[::-1][:top_n]
    return np.stack([ys[order], xs[order]], axis=1).astype(float)


def vote_offset(ref: np.ndarray, obs: np.ndarray, max_off: float = 400.0,
                bin_px: float = 2.0) -> Tuple[np.ndarray, int]:
    """Recover the rigid (dy, dx) mapping ``obs`` onto ``ref`` by histogramming pairwise offsets.

    Robust to large drift and to most of ``obs`` being unmatched, which plain nearest-neighbour
    matching is not.  Same coarse-vote trick the pipeline's TIME_COARSE_ALIGN uses.
    """
    if len(ref) == 0 or len(obs) == 0:
        return np.zeros(2), 0
    d = (ref[:, None, :] - obs[None, :, :]).reshape(-1, 2)
    d = d[(np.abs(d[:, 0]) <= max_off) & (np.abs(d[:, 1]) <= max_off)]
    if not len(d):
        return np.zeros(2), 0
    q = np.round(d / bin_px).astype(int)
    uq, cnt = np.unique(q, axis=0, return_counts=True)
    win = uq[cnt.argmax()] * bin_px
    near = np.abs(d - win).max(axis=1) <= 3.0
    return d[near].mean(axis=0), int(cnt.max())


def _zmap(img: np.ndarray, smooth: float = 3.0) -> np.ndarray:
    """Background-subtract, broad-smooth, express in robust (MAD) noise sigmas."""
    f = ndi.gaussian_filter(img - ndi.uniform_filter(img, 61), smooth)
    med = float(np.median(f))
    noise = 1.4826 * float(np.median(np.abs(f - med))) or 1.0
    return (f - med) / noise


def find_fiducials(images: Dict[str, np.ndarray], snr: float = 30.0, min_sep: int = 25,
                   top_n: int = 60) -> np.ndarray:
    """Locate true fiducial beads: peaks bright in EVERY channel.

    A fiducial is bright in G, R and B; a single molecule is bright in exactly one.  Green carries
    ~6500 nucleosome molecules per field, so a per-channel brightness cut on green returns
    molecules, not beads -- and their blended neighbours wreck the PSF estimate.  Taking the
    per-pixel MINIMUM of the per-channel z-maps keeps only what is bright everywhere.  (This is the
    same argument epinuc_colocalization.detect_beads_multichannel makes; we redo it at a broad
    smoothing scale so a defocused R/B bead still registers.)
    """
    usable = [im for im in images.values() if im is not None]
    if len(usable) < 2:
        return np.empty((0, 2), float)
    composite = np.minimum.reduce([_zmap(im) for im in usable])
    peaks = (ndi.maximum_filter(composite, min_sep) == composite) & (composite > snr)
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return np.empty((0, 2), float)
    order = np.argsort(composite[ys, xs])[::-1][:top_n]
    return np.stack([ys[order], xs[order]], axis=1).astype(float)


def locate_in_frame(img: np.ndarray, ref: np.ndarray, search: int = 8) -> Tuple[np.ndarray, float]:
    """Map the fiducial set ``ref`` into ``img`` through a rigid drift, refining each locally.

    Beads are stuck to the flowcell, so one rigid offset carries the whole set; the beads move
    ~20 px between cycles, far beyond any nearest-neighbour tolerance, which is why the offset is
    recovered by voting rather than matching.
    """
    if len(ref) == 0:
        return np.empty((0, 2), float), float("nan")
    cand = find_beads(img)
    off, votes = vote_offset(ref, cand) if len(cand) else (np.zeros(2), 0)
    if votes == 0:
        off = np.zeros(2)
    moved = ref - off  # reference set, expressed in this frame

    z = _zmap(img)
    H, W = img.shape
    out = []
    for y, x in moved:
        y, x = int(round(y)), int(round(x))
        if y < search or x < search or y >= H - search or x >= W - search:
            continue
        win = z[y - search:y + search + 1, x - search:x + search + 1]
        dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
        out.append((y - search + dy, x - search + dx))
    return np.asarray(out, float), float(np.hypot(*off))


# --------------------------------------------------------------------------------------
# Empirical PSF measurement
# --------------------------------------------------------------------------------------
@dataclass
class PSFResult:
    psf: Optional[np.ndarray] = None       # normalized, unit sum
    sigma: float = float("nan")            # sqrt(sigma_maj * sigma_min), px
    n_beads: int = 0
    note: str = ""


def _stamp_geometry(half: int, ap_r: float, ann_in: float, ann_out: float):
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    rr = np.hypot(yy, xx)
    return yy, xx, rr, rr < ap_r, (rr >= ann_in) & (rr <= ann_out)


#: half-light radius of a 2-D Gaussian, in units of sigma
_R50_PER_SIGMA = 1.17741


def psf_r50(stamp: np.ndarray, half: int, ap_r: float) -> float:
    """Half-light radius: the radius enclosing 50% of the flux inside ``ap_r``.

    The second moment is the natural size statistic and it is the wrong one here.  A fixed wide
    aperture weights faint 2% wings by r^2 and reads a sharp sigma~2.5 green PSF as ~4.7; adapting
    the aperture to ~3 sigma instead chops the plateau off a flat-topped defocus disk and reads a
    sigma~5.4 disk as ~3.7.  r50 is bounded, wing-insensitive, and well defined for both a Gaussian
    core and a defocus disk (r50 = R/sqrt(2) for a uniform disk of radius R).
    """
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    r = np.hypot(yy, xx)
    m = r < ap_r
    v = np.clip(stamp, 0, None)[m]
    # Accumulate on the DISTINCT radii of the pixel lattice, then interpolate between them.
    # Interpolating over the raw per-pixel list makes r50 snap to lattice values (3, sqrt(8),
    # sqrt(17), ...) because every duplicate radius is a jump discontinuity in the cumulative.
    uniq, inv = np.unique(r[m], return_inverse=True)
    c = np.cumsum(np.bincount(inv, weights=v))
    if c[-1] <= 0:
        return float("nan")
    return float(np.interp(0.5, c / c[-1], uniq))


def _sigma_of(stamp: np.ndarray, half: int, ap_r: float) -> float:
    """Gaussian-equivalent sigma, derived from the half-light radius."""
    r50 = psf_r50(stamp, half, ap_r)
    return r50 / _R50_PER_SIGMA if np.isfinite(r50) else float("nan")


def measure_psf(img: np.ndarray, coords: np.ndarray, half: int = 32, ap_r: float = 18.0,
                ann_in: float = 22.0, ann_out: float = 30.0, min_beads: int = 4,
                max_centroid_shift: float = 6.0) -> PSFResult:
    """Median-combine centroid-aligned bead stamps into an empirical PSF, then measure its sigma.

    Two details that matter:

    * The background comes from an **annulus**, not a local median filter.  A ``median_filter(15)``
      has a footprint comparable to a sigma~6 bead (FWHM ~14 px) and subtracts the bead's own wings.
    * Stamps are stacked **unclipped**, so the wing noise stays centred on zero.  Clipping each
      stamp at zero before combining biases the wings positive and inflates the sigma.

    Sigma is then read off the high-SNR stack, not averaged over noisy per-bead estimates.
    """
    yy, xx, rr, ap, ann = _stamp_geometry(half, ap_r, ann_in, ann_out)
    # Centroid on a SMALL aperture.  Green carries ~6500 nucleosome molecules per field, so a
    # centroid taken over the full r<ap_r aperture is dragged off by neighbours and the stack
    # smears -- which reads a sharp sigma~2.5 green PSF as ~4.9.  The bead core dominates r<6
    # for both a sharp PSF and a defocus disk, and both are symmetric about it.
    cen = rr < 6.0
    H, W = img.shape
    stamps: List[np.ndarray] = []

    for y, x in coords:
        y, x = int(round(y)), int(round(x))
        if y < half or x < half or y >= H - half or x >= W - half:
            continue
        st = img[y - half:y + half + 1, x - half:x + half + 1]
        bg = float(np.median(st[ann]))
        p = st - bg

        w = np.clip(p, 0, None) * cen
        s = w.sum()
        if s <= 0:
            continue
        cy = float((w * yy).sum() / s)
        cx = float((w * xx).sum() / s)
        if abs(cy) > max_centroid_shift or abs(cx) > max_centroid_shift:
            continue  # blended / off-centre object

        sh = ndi.shift(p, (-cy, -cx), order=1, mode="nearest")   # subpixel recentre, unclipped
        flux = float((np.clip(sh, 0, None) * ap).sum())
        if flux <= 0:
            continue
        stamps.append(sh / flux)

    if len(stamps) < min_beads:
        return PSFResult(None, float("nan"), len(stamps),
                         f"only {len(stamps)} usable beads (<{min_beads})")

    psf = np.median(np.stack(stamps, axis=0), axis=0)
    psf = psf - float(np.median(psf[ann]))   # kill any residual pedestal before the moment
    sigma = _sigma_of(psf, half, ap_r)
    psf = np.clip(psf, 0, None)
    total = psf.sum()
    if total <= 0:
        return PSFResult(None, sigma, len(stamps), "degenerate stack")
    psf /= total
    return PSFResult(psf, sigma, len(stamps), "empirical")


def gaussian_psf(half: int, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    g = np.exp(-(yy ** 2 + xx ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


# --------------------------------------------------------------------------------------
# PSF homogenization
# --------------------------------------------------------------------------------------
def _embed(kernel: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Place an odd-sized kernel at the centre of a zero array of ``shape``."""
    out = np.zeros(shape, np.float32)
    kh, kw = kernel.shape
    H, W = shape
    y0, x0 = (H - kh) // 2, (W - kw) // 2
    out[y0:y0 + kh, x0:x0 + kw] = kernel
    return out


def _matching_kernel(Pf: np.ndarray, Tf: np.ndarray, lam: float) -> np.ndarray:
    """Regularized matching kernel, renormalized to *exact* unit gain at DC.

    Without the renormalization the DC gain is 1/(1+lam), which quietly scales every intensity
    down by a few percent and breaks the photometry the downstream calibration depends on.
    """
    K = np.conj(Pf) * Tf / (np.abs(Pf) ** 2 + lam)
    return K / K[0, 0].real


def _white_noise_gain(K: np.ndarray, shape: Tuple[int, int]) -> float:
    """RMS amplification K applies to white noise: sqrt(mean |K|^2) over the FULL spectrum.

    ``K`` lives on an rfft2 half-grid, so the interior columns stand for two conjugate
    frequencies each and must be counted twice.
    """
    H, W = shape
    w = np.ones(K.shape, float)
    if W % 2 == 0:
        w[:, 1:-1] = 2.0
    else:
        w[:, 1:] = 2.0
    return float(np.sqrt((np.abs(K) ** 2 * w).sum() / (H * W)))


def choose_lambda(Pf: np.ndarray, Tf: np.ndarray, shape: Tuple[int, int],
                  max_noise_gain: float) -> Tuple[float, float]:
    """Bisect lam so the filter's white-noise amplification meets ``max_noise_gain``.

    This is the honest knob.  A defocus OTF has genuine zeros; asking for the target's response
    there demands division by ~0, so *some* regularization is mandatory and it necessarily costs
    sharpness.  Rather than guess a noise-to-signal ratio, we bound the noise we are willing to
    pay and take whatever sharpening that buys.  Noise gain falls monotonically with lam.
    """
    lo, hi = 1e-9, 1.0
    if _white_noise_gain(_matching_kernel(Pf, Tf, hi), shape) > max_noise_gain:
        return hi, _white_noise_gain(_matching_kernel(Pf, Tf, hi), shape)
    for _ in range(60):
        mid = float(np.sqrt(lo * hi))
        if _white_noise_gain(_matching_kernel(Pf, Tf, mid), shape) > max_noise_gain:
            lo = mid
        else:
            hi = mid
    return hi, _white_noise_gain(_matching_kernel(Pf, Tf, hi), shape)


def homogenize(img: np.ndarray, psf: np.ndarray, target: np.ndarray,
               lam: Optional[float] = None,
               max_noise_gain: float = 2.0) -> Tuple[np.ndarray, float, float]:
    """Linearly map ``img`` from PSF ``psf`` onto PSF ``target``.

    Returns ``(corrected, lam, predicted_noise_gain)``.  Both kernels are unit-sum and the matching
    kernel is renormalized to unit DC gain, so background level and total flux are preserved.
    Reflect-padding keeps the FFT's circular wrap-around off the real field.
    """
    pad = psf.shape[0] // 2 + 1
    padded = np.pad(img, pad, mode="reflect")
    shape = padded.shape

    Pf = np.fft.rfft2(np.fft.ifftshift(_embed(psf, shape)))
    Tf = np.fft.rfft2(np.fft.ifftshift(_embed(target, shape)))

    if lam is None:
        lam, gain = choose_lambda(Pf, Tf, shape, max_noise_gain)
    else:
        gain = _white_noise_gain(_matching_kernel(Pf, Tf, lam), shape)

    K = _matching_kernel(Pf, Tf, lam)
    out = np.fft.irfft2(np.fft.rfft2(padded) * K, s=shape)
    return out[pad:pad + img.shape[0], pad:pad + img.shape[1]].astype(np.float32), lam, gain


def richardson_lucy_to_target(img: np.ndarray, psf: np.ndarray, target: np.ndarray,
                              iters: int = 20) -> np.ndarray:
    """Non-negative Poisson alternative: RL-deconvolve by ``psf``, then reconvolve with ``target``.

    Nonlinear -- it will perturb the intensity calibration.  Offered, not recommended as default.
    """
    from scipy.signal import fftconvolve

    floor = float(np.percentile(img, 1))
    obs = np.clip(img - floor, 1e-6, None)
    est = np.full_like(obs, obs.mean())
    psf_flip = psf[::-1, ::-1]
    for _ in range(iters):
        conv = fftconvolve(est, psf, mode="same")
        conv[conv < 1e-9] = 1e-9
        est *= fftconvolve(obs / conv, psf_flip, mode="same")
        est = np.clip(est, 0, None)
    out = fftconvolve(est, target, mode="same") + floor
    return out.astype(np.float32)


# --------------------------------------------------------------------------------------
# Per-file correction
# --------------------------------------------------------------------------------------
def _count_saturated_beads(img: np.ndarray, beads: np.ndarray, sat: np.ndarray,
                           radius: int = 6) -> int:
    """How many located beads have a saturated (clipped) core."""
    H, W = img.shape
    n = 0
    for y, x in beads:
        y, x = int(round(y)), int(round(x))
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        if sat[y0:y1, x0:x1].any():
            n += 1
    return n


@dataclass
class FileResult:
    row: dict = field(default_factory=dict)
    corrected: Optional[np.ndarray] = None
    xmp: Optional[bytes] = None


def correct_file(path: str, ref_beads: np.ndarray, target: np.ndarray, target_sigma: float,
                 method: str, reg: Optional[float], half: int, max_noise_gain: float = 2.0,
                 measure_after: bool = True, measure_only: bool = False) -> FileResult:
    img, xmp = read_tif(path)
    meta = parse_tif_name(path) or {}
    row = {
        "file": os.path.basename(path),
        "lane": meta.get("lane"), "pos": meta.get("pos"),
        "dye": meta.get("laser"), "cycle": meta.get("cycle"),
        "focus_locked": xmp_field(xmp, "PsdInPosition"),
        "attempt_number": xmp_field(xmp, "AttemptNumber"),
        "z_position": xmp_field(xmp, "ZPosition"),
    }

    # Saturation is the real defect in this data: ~14% of beads clip at the 16-bit ceiling, the
    # brightest fields worst.  Clipped wells bloom vertically (CCD column bleed), which is what
    # reads as a "streak".  The clipped intensity is lost -- these columns flag it so those frames
    # (and the pixels under a saturated bead) can be excluded from photometry downstream.
    sat = img >= _SATURATION_LEVEL
    row["n_saturated_px"] = int(sat.sum())
    row["saturated"] = int(row["n_saturated_px"] > 0)

    beads, drift = locate_in_frame(img, ref_beads)
    row["n_candidates"] = len(ref_beads)
    row["drift_px"] = round(drift, 2) if np.isfinite(drift) else ""
    row["n_beads_saturated"] = _count_saturated_beads(img, beads, sat)

    res = measure_psf(img, beads, half=half)
    row["n_beads"] = res.n_beads
    row["sigma_before"] = round(res.sigma, 3) if np.isfinite(res.sigma) else ""
    row["note"] = res.note

    if measure_only:
        # Report-only: skip the FFT homogenization entirely.  The correction is unnecessary for
        # this data and the per-file FFT dominates wall-clock over ~2000 files on the SMB mount.
        row["action"] = "measured"
        return FileResult(row, None, xmp)

    psf = res.psf
    if psf is None:
        if not np.isfinite(res.sigma):
            row["action"] = "passthrough (no PSF)"
            return FileResult(row, None, xmp)
        psf = gaussian_psf(half, res.sigma)
        row["note"] = f"{res.note}; gaussian fallback"

    if np.isfinite(res.sigma) and res.sigma <= target_sigma * 1.05:
        row["action"] = "passthrough (already at target)"
        return FileResult(row, None, xmp)

    if method == "rl":
        out = richardson_lucy_to_target(img, psf, target)
        row["lambda"] = ""
    else:
        out, lam, predicted = homogenize(img, psf, target, reg, max_noise_gain)
        row["lambda"] = float(f"{lam:.3g}")
        row["noise_gain_predicted"] = round(predicted, 3)

    # noise gain, measured on the flat background (below the 60th percentile)
    def _bgnoise(a):
        v = a[a < np.percentile(a, 60)]
        return 1.4826 * float(np.median(np.abs(v - np.median(v)))) or 1.0

    row["noise_gain"] = round(_bgnoise(out) / _bgnoise(img), 3)

    clipped = float(((out < 0) | (out > 65535)).mean())
    row["clipped_frac"] = round(clipped, 6)
    out = np.clip(out, 0, 65535)

    if measure_after and len(beads):
        after = measure_psf(out, beads, half=half)
        row["sigma_after"] = round(after.sigma, 3) if np.isfinite(after.sigma) else ""

    row["action"] = f"corrected ({method})"
    return FileResult(row, out.astype(np.uint16), xmp)


def write_tif(path: str, arr: np.ndarray, xmp: Optional[bytes]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    extratags = [(_XMP_TAG, 1, len(xmp), xmp, True)] if xmp else []
    tifffile.imwrite(path, arr, extratags=extratags)


# --------------------------------------------------------------------------------------
# Flat-field correction  (independent of the PSF path; applies to EVERY channel)
#
# The EpiVision optics impose a per-mode fixed pattern -- vignetting up to ~47%, plus fixed
# optical debris and hot pixels -- that differs for R, G and B (each laser has its own path).
# Across many fields of view the beads/molecules move but that pattern is stationary, so the
# per-pixel MEDIAN across FOVs recovers it (bead-robust).  The correction is the textbook
#
#     corrected = (frame - bias) / gain_norm ,   gain_norm = (flat - bias) / mean(flat - bias)
#
# with ``bias`` the additive sensor pedestal (SUBTRACTED) and ``gain_norm`` the multiplicative
# vignette (DIVIDED).  gain_norm has unit mean, so a flat background is preserved at its own mean
# DN rather than zeroed, and relative photometry within a frame is unchanged.
#
# ``bias`` is measured by photon transfer: for shot-noise-limited background, tile variance is
# linear in tile mean and its x-intercept is the bias.  R and B have real background so this works
# directly.  Green is dense nucleosome signal with no background, so its fit never reaches the
# sensor floor and is invalid; green then borrows the sensor bias (a detector constant) from R/B.
# A measured dark frame would replace this inference -- especially for green.
# --------------------------------------------------------------------------------------
_FLAT_GCLIP = (0.3, 3.0)   # clamp the normalized gain so bright debris cannot nuke real signal
_FLAT_TILE = 48


@dataclass
class Calib:
    mode: str
    flat: np.ndarray
    bias: float
    gain: float            # e- per DN, from the photon-transfer slope (nan if transferred)
    valid: bool            # was the bias measured (True) or transferred from another mode (False)
    n: int
    gnorm: Optional[np.ndarray] = None   # cached normalized gain

    def gain_map(self) -> np.ndarray:
        if self.gnorm is None:
            g = self.flat - self.bias
            m = float(g.mean()) or 1.0
            self.gnorm = np.clip(g / m, *_FLAT_GCLIP).astype(np.float32)
        return self.gnorm


def _mode_files(files: List[str], mode: str, cap: int) -> List[str]:
    """Even-sampled subset of ``files`` for one illumination mode (spans all cycles)."""
    ps = [f for f in files if (parse_tif_name(f) or {}).get("laser") == mode]
    if len(ps) > cap:
        idx = np.linspace(0, len(ps) - 1, cap).round().astype(int)
        ps = [ps[i] for i in idx]
    return ps


def _load_stack(paths: List[str], prefix: str = "") -> np.ndarray:
    frames = []
    prog = _Progress(len(paths), prefix) if prefix else None
    for k, p in enumerate(paths, 1):
        try:
            img, _ = read_tif(p)
            if img.ndim == 2:
                frames.append(img)
        except OSError as e:
            warnings.warn(f"flat-field: skipping {os.path.basename(p)}: {e}")
        if prog:
            prog.update(k)
    if not frames:
        return np.empty((0, 0, 0), np.float32)
    return np.stack(frames, 0)


def photon_transfer_bias(stack: np.ndarray, tile: int = _FLAT_TILE, nbin: int = 40,
                         lo_pct: float = 8.0) -> Tuple[float, float, bool]:
    """Sensor bias (DN) and gain (e-/DN) from the lower envelope of tile variance vs mean.

    Bead/structure tiles only ADD variance, so the lower envelope of variance in each mean bin is
    clean background; a line through it hits ``var = 0`` at the bias.  Returns ``(bias, gain,
    valid)`` -- ``valid`` is False when the cloud never nears the floor (dense-signal channels like
    green), i.e. the extrapolated bias lands at/above the dimmest tile.

    Per-tile variance comes from adjacent-pixel differences, not the raw spatial variance: a steep
    vignette (up to ~47%) leaves a real illumination gradient across a 48 px tile that would swamp
    the shot noise in a plain ``tile.var``. A 1-px difference cancels the smooth gradient, and its
    MAD ignores the odd bead pixel, leaving the shot noise.
    """
    n, h, w = stack.shape
    H, W = (h // tile) * tile, (w // tile) * tile
    means, vars_ = [], []
    for fi in range(n):
        t = stack[fi, :H, :W].reshape(H // tile, tile, W // tile, tile)
        tm = np.median(t, axis=(1, 3)).ravel()                        # robust tile level
        dx = np.abs(np.diff(t, axis=3))                               # 1-px differences cancel gradient
        sig = 1.4826 * np.median(dx, axis=(1, 3)).ravel() / np.sqrt(2)  # shot noise from diff-MAD
        means.append(tm); vars_.append(sig ** 2)
    means = np.concatenate(means); vars_ = np.concatenate(vars_)
    if len(means) < 50:
        return float(np.percentile(stack, 1)), float("nan"), False
    lo, hi = np.percentile(means, 1), np.percentile(means, 97)
    edges = np.linspace(lo, hi, nbin + 1)
    bm, bv = [], []
    for i in range(nbin):
        sel = (means >= edges[i]) & (means < edges[i + 1])
        if sel.sum() >= 20:
            bm.append(0.5 * (edges[i] + edges[i + 1]))
            bv.append(np.percentile(vars_[sel], lo_pct))
    bm, bv = np.asarray(bm), np.asarray(bv)
    if len(bm) < 4:
        return float(np.percentile(means, 1)), float("nan"), False
    slope, intercept = np.polyfit(bm, bv, 1)
    if slope <= 0:
        return float(np.percentile(means, 1)), float("nan"), False
    bias = -intercept / slope
    gain = 1.0 / slope
    valid = (bias < bm.min() - 5.0) and (0.15 < gain < 3.0)
    return float(bias), float(gain), bool(valid)


def build_flatfield(files: List[str], max_frames: int) -> Dict[str, Calib]:
    """Build a per-mode flat + bias calibration from the run's own frames.

    One calibration per illumination mode across all selected FOVs/cycles.  Modes whose bias could
    not be measured (green) inherit the median sensor bias of the modes that could (R, B).
    """
    modes = sorted({(parse_tif_name(f) or {}).get("laser") for f in files} - {None})
    cal: Dict[str, Calib] = {}
    for mode in modes:
        paths = _mode_files(files, mode, max_frames)
        stack = _load_stack(paths, prefix=f"  reading {mode} flat ")
        if stack.size == 0 or len(stack) < 8:
            warnings.warn(f"flat-field: too few frames for mode {mode} ({len(stack)}); skipping")
            continue
        flat = np.median(stack, axis=0).astype(np.float32)
        bias, gain, valid = photon_transfer_bias(stack)
        cal[mode] = Calib(mode, flat, bias, gain, valid, len(stack))
        print(f"  {mode}: n={len(stack):3d}  flat {flat.min():.0f}-{flat.max():.0f}  "
              f"bias={bias:.1f} DN gain={gain:.3f} e-/DN valid={valid}", flush=True)
        del stack
    valid_biases = [c.bias for c in cal.values() if c.valid]
    sensor_bias = float(np.median(valid_biases)) if valid_biases else None
    for c in cal.values():
        if not c.valid:
            if sensor_bias is None:
                warnings.warn(f"flat-field: no mode has a measurable bias; "
                              f"{c.mode} keeps its (unreliable) fit {c.bias:.0f} DN")
            else:
                print(f"  {c.mode}: bias transferred from R/B -> {sensor_bias:.1f} DN "
                      f"(own fit {c.bias:.0f} unreliable)", flush=True)
                c.bias = sensor_bias
    return cal


def apply_flatfield(img: np.ndarray, cal: Calib) -> np.ndarray:
    """Bias-subtract then divide the normalized gain.  Returns float32 (unclipped)."""
    return ((img - cal.bias) / cal.gain_map()).astype(np.float32)


def save_calibration(cal: Dict[str, Calib], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    import csv
    for c in cal.values():
        np.save(os.path.join(outdir, f"flat_{c.mode}.npy"), c.flat)
    with open(os.path.join(outdir, "calibration.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "n", "bias_dn", "gain_e_per_dn", "valid", "flat_min", "flat_max"])
        for c in cal.values():
            w.writerow([c.mode, c.n, f"{c.bias:.2f}", f"{c.gain:.4f}", int(c.valid),
                        f"{c.flat.min():.0f}", f"{c.flat.max():.0f}"])
    print(f"  calibration saved -> {outdir}", flush=True)


def load_calibration(indir: str) -> Dict[str, Calib]:
    import csv
    cal: Dict[str, Calib] = {}
    with open(os.path.join(indir, "calibration.csv")) as fh:
        for row in csv.DictReader(fh):
            flat = np.load(os.path.join(indir, f"flat_{row['mode']}.npy"))
            cal[row["mode"]] = Calib(row["mode"], flat, float(row["bias_dn"]),
                                     float(row["gain_e_per_dn"]), bool(int(row["valid"])),
                                     int(row["n"]))
    return cal


def _mirror_ancillary(indir: str, outdir: str, processed: set) -> int:
    """Copy any input entry that is NOT a corrected TIF (stray files, metadata, subfolders) verbatim.

    The correction only ever touches parseable TIFs, so everything else -- a run-level metadata file,
    an unparseable TIF, an ancillary subfolder -- is copied through unchanged, keeping the output run a
    faithful, complete mirror of the input. (Both current runs are pure flat TIF dirs, so this is a
    no-op today; it future-proofs a run that ships extra content.)
    """
    n = 0
    for name in sorted(os.listdir(indir)):
        if name in processed:
            continue
        src, dst = os.path.join(indir, name), os.path.join(outdir, name)
        if os.path.exists(dst):
            continue
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            n += 1
        except (OSError, shutil.Error) as e:
            warnings.warn(f"could not copy ancillary entry {name}: {e}")
    return n


def _flatfield_one(path: str, cal: Dict[str, Calib], outdir: Optional[str],
                   dry_run: bool) -> dict:
    """Flat-field a single file -- read, correct, write -- and return its report row.

    All the state this touches is either local or read-only (``cal``), and each call writes to its
    own output path, so several of these can run at once.  See ``run_flatfield`` for why that is
    worth doing.
    """
    m = parse_tif_name(path)
    dst = os.path.join(outdir, os.path.basename(path)) if outdir else None
    row = {"file": os.path.basename(path), "lane": m["lane"], "pos": m["pos"],
           "dye": m["laser"], "cycle": m["cycle"]}
    try:
        img, xmp = read_tif(path)
    except OSError as e:
        row["action"] = f"ERROR: {e}"
        warnings.warn(f"{row['file']}: {e}")
        return row

    sat = img >= _SATURATION_LEVEL           # saturation is a RAW-sensor property
    row["n_saturated_px"] = int(sat.sum())
    row["saturated"] = int(row["n_saturated_px"] > 0)
    row["focus_locked"] = xmp_field(xmp, "PsdInPosition")
    row["attempt_number"] = xmp_field(xmp, "AttemptNumber")
    row["z_position"] = xmp_field(xmp, "ZPosition")

    c = cal.get(m["laser"])
    if c is None:
        row["action"] = "passthrough (no calibration for mode)"
        if not dry_run:
            os.makedirs(outdir, exist_ok=True)
            shutil.copy2(path, dst)
        return row

    out = apply_flatfield(img, c)
    row["bias_dn"] = round(c.bias, 1)
    row["flat_valid"] = int(c.valid)
    row["clipped_frac"] = round(float(((out < 0) | (out > 65535)).mean()), 6)
    out = np.clip(out, 0, 65535).astype(np.uint16)
    row["action"] = "flat-fielded"
    if not dry_run:
        write_tif(dst, out, xmp)
    return row


def run_flatfield(args, files: List[str]) -> int:
    """Flat-field the whole run (every channel) and write corrected TIFs to the mirrored tree."""
    import csv
    # Keep --out a pure mirror of the raw layout (a flat directory of *.tif, no subdirectories), so
    # the loader -- which enumerates run directories -- never sees a stray non-image folder.  The
    # calibration and report therefore go to a SIBLING directory ``<out>_flatfield``, not inside --out.
    sidecar = (os.path.abspath(args.outdir).rstrip(os.sep) + "_flatfield") if args.outdir else None
    if args.flat_load:
        print(f"loading flat-field calibration from {args.flat_load} ...", flush=True)
        cal = load_calibration(args.flat_load)
        for c in cal.values():
            print(f"  {c.mode}: bias={c.bias:.1f} DN gain={c.gain:.3f} "
                  f"flat {c.flat.min():.0f}-{c.flat.max():.0f}", flush=True)
    else:
        print("building per-mode flat-field (median across FOVs) + photon-transfer bias ...",
              flush=True)
        cal = build_flatfield(files, args.flat_max_frames)
        if not cal:
            sys.exit("flat-field: no calibration could be built")
        if sidecar:
            save_calibration(cal, sidecar)

    # Correct the files concurrently.  Each file is an independent read -> arithmetic -> write, and
    # the read is the expensive part: the raw runs live on an SMB share, so a serial loop spends most
    # of its wall clock blocked on the network with the CPU idle.  THREADS, not processes: the blocking
    # calls (socket read, file write) release the GIL, as do the NumPy ops in apply_flatfield, so the
    # work genuinely overlaps -- and threads share the per-mode flats instead of pickling a 16 MB array
    # to every worker.  Force the lazy gain map to materialize first so the workers only ever read it.
    for c in cal.values():
        c.gain_map()

    rows: List[Optional[dict]] = [None] * len(files)
    verb = "measuring" if args.dry_run else "correcting"
    prog = _Progress(len(files), f"  {verb} ")
    lock = threading.Lock()
    n_seen = 0

    def _task(item) -> None:
        nonlocal n_seen
        i, p = item
        try:
            row = _flatfield_one(p, cal, args.outdir, args.dry_run)
        except Exception as e:          # one bad frame must not take the whole run down
            m = parse_tif_name(p) or {}
            row = {"file": os.path.basename(p), "lane": m.get("lane"), "pos": m.get("pos"),
                   "dye": m.get("laser"), "cycle": m.get("cycle"), "action": f"ERROR: {e}"}
            warnings.warn(f"{os.path.basename(p)}: {e}")
        rows[i] = row
        with lock:                      # the bar is the only thing the threads share
            n_seen += 1
            prog.update(n_seen)

    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for item in enumerate(files):
            _task(item)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(_task, enumerate(files)))

    rows = [r for r in rows if r is not None]
    n_done = sum(1 for r in rows if r["action"] == "flat-fielded")
    n_skip = sum(1 for r in rows if r["action"].startswith("passthrough"))

    if not args.dry_run and args.outdir:
        nanc = _mirror_ancillary(args.indir, args.outdir, {os.path.basename(p) for p in files})
        if nanc:
            print(f"  copied {nanc} ancillary (non-TIF) entr{'y' if nanc == 1 else 'ies'} verbatim",
                  flush=True)

    cols = ["file", "lane", "pos", "dye", "cycle", "bias_dn", "flat_valid", "n_saturated_px",
            "saturated", "clipped_frac", "focus_locked", "attempt_number", "z_position", "action"]
    report_path = args.report
    if sidecar and not os.path.isabs(report_path):        # keep the report out of the TIF mirror too
        os.makedirs(sidecar, exist_ok=True)
        report_path = os.path.join(sidecar, report_path)
    with open(report_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nflat-fielded {n_done}, passthrough {n_skip}")
    print(f"report -> {report_path}")
    if args.dry_run:
        print("dry run: no pixels written")
    return 0


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def reference_fiducials(files: List[str], half: int) -> Tuple[Dict[Tuple[str, int], np.ndarray],
                                                              List[np.ndarray], List[float]]:
    """Cross-channel fiducials per (lane, pos), plus the green PSF stamps and sigmas.

    Uses the cycle-1 G/R/B triple at each position: a bead is bright in all three, a molecule in
    one.  Beads are flowcell-fixed, so this one set carries into every later cycle under a rigid
    offset.  Returns ``(refs, green_psfs, green_sigmas)``.
    """
    triples: Dict[Tuple[str, int], Dict[str, str]] = {}
    for p in files:
        m = parse_tif_name(p)
        if not m or m["cycle"] != 1:
            continue
        triples.setdefault((m["lane"], m["pos"]), {})[m["laser"]] = p

    refs: Dict[Tuple[str, int], np.ndarray] = {}
    green_psfs: List[np.ndarray] = []
    green_sigmas: List[float] = []
    for key, byl in sorted(triples.items()):
        if "G" not in byl:
            continue
        try:
            imgs = {l: read_tif(p)[0] for l, p in byl.items()}
        except OSError as e:
            warnings.warn(f"could not read cycle-1 triple for {key}: {e}")
            continue
        fids = find_fiducials(imgs)
        if len(fids) == 0:
            warnings.warn(f"no cross-channel fiducials at {key}")
            continue
        refs[key] = fids
        gres = measure_psf(imgs["G"], fids, half=half)
        if gres.psf is not None:
            green_psfs.append(gres.psf)
        if np.isfinite(gres.sigma):
            green_sigmas.append(gres.sigma)
    return refs, green_psfs, green_sigmas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", required=True, help="run directory of raw TIFs")
    ap.add_argument("--out", dest="outdir", help="mirrored output directory (required unless --dry-run)")
    ap.add_argument("--lanes", nargs="*", default=None, help="e.g. ch1 ch2 (default: all)")
    ap.add_argument("--channels", nargs="*", default=["R", "B"],
                    help="lasers to correct; others are copied byte-identical (default: R B)")
    ap.add_argument("--target", choices=["empirical", "gaussian"], default="empirical",
                    help="target PSF: green's own measured PSF (default) or a Gaussian")
    ap.add_argument("--target-sigma", default="auto",
                    help="'auto' = median sigma of the cycle-1 green images, else a float (px)")
    ap.add_argument("--method", choices=["wiener", "rl"], default="wiener")
    ap.add_argument("--max-noise-gain", type=float, default=2.0,
                    help="bound the white-noise amplification; lambda is bisected to meet it "
                         "(default 2.0). Higher = sharper and noisier.")
    ap.add_argument("--reg", default="auto",
                    help="'auto' = pick lambda from --max-noise-gain, else a fixed float")
    ap.add_argument("--stamp-half", type=int, default=32, help="PSF stamp half-width (px)")
    ap.add_argument("--flat-field", action="store_true",
                    help="run the per-mode flat-field / bias correction instead of the PSF path: "
                         "builds a median flat + photon-transfer bias per mode and writes "
                         "(frame - bias)/gain_norm for EVERY channel (green included) to --out.")
    ap.add_argument("--flat-max-frames", type=int, default=80,
                    help="frames per mode used to build the flat (median across FOVs; default 80)")
    ap.add_argument("--flat-load", default=None,
                    help="reuse a saved calibration dir (from a prior run's <out>/_flatfield) "
                         "instead of rebuilding it -- e.g. apply one run's flat to another")
    ap.add_argument("--jobs", type=int, default=8,
                    help="threads used to flat-field files concurrently (default 8). The work is "
                         "dominated by reads off the SMB share, so overlapping them is most of the "
                         "speedup; --jobs 1 restores the old serial loop.")
    ap.add_argument("--report", default="psf_report.csv")
    ap.add_argument("--dry-run", action="store_true", help="measure and report; write no pixels")
    ap.add_argument("--measure-only", action="store_true",
                    help="skip the PSF correction entirely; report saturation/drift/PSF/focus only "
                         "(much faster over many files). Implies --dry-run.")
    args = ap.parse_args(argv)

    if args.measure_only:
        args.dry_run = True
    if not args.dry_run and not args.outdir:
        ap.error("--out is required unless --dry-run")

    files = sorted(
        os.path.join(args.indir, f) for f in os.listdir(args.indir) if f.lower().endswith(".tif")
    )
    files = [f for f in files if parse_tif_name(f)]
    if args.lanes:
        keep = set(args.lanes)
        files = [f for f in files if parse_tif_name(f)["lane"] in keep]
    if not files:
        sys.exit(f"no parseable TIFs under {args.indir}")
    print(f"{len(files)} TIF(s) to consider", flush=True)

    # Flat-field is a self-contained correction path -- it does not need the PSF machinery, and it
    # corrects EVERY channel (green included), so it dispatches here before the fiducial pass.
    if args.flat_field:
        return run_flatfield(args, files)

    print("locating cross-channel fiducials in the cycle-1 G/R/B triples ...", flush=True)
    refs, green_psfs, green_sigmas = reference_fiducials(files, args.stamp_half)
    if not refs:
        sys.exit("no cross-channel fiducials found; cannot measure a PSF")
    nfid = int(np.median([len(v) for v in refs.values()]))
    print(f"  {len(refs)} (lane,pos) reference sets, median {nfid} fiducials each", flush=True)

    # Target PSF: green's own empirical PSF -- "make R/B look like green".  Falls back to a
    # Gaussian only if green's PSF could not be measured.
    if args.target_sigma == "auto":
        if not green_sigmas:
            sys.exit("could not measure a green PSF; pass --target-sigma explicitly")
        target_sigma = float(np.median(green_sigmas))
    else:
        target_sigma = float(args.target_sigma)

    if args.target == "empirical" and green_psfs:
        target = np.median(np.stack(green_psfs, axis=0), axis=0)
        target = np.clip(target, 0, None)
        target /= target.sum()
        tsrc = f"empirical green PSF (median of {len(green_psfs)})"
    else:
        target = gaussian_psf(args.stamp_half, target_sigma)
        tsrc = "gaussian"
    print(f"target = {tsrc}, sigma = {target_sigma:.3f} px "
          f"(from {len(green_sigmas)} green images)", flush=True)

    reg = None if args.reg == "auto" else float(args.reg)
    channels = {c.upper() for c in args.channels}

    rows: List[dict] = []
    n_corrected = n_copied = 0
    for i, p in enumerate(files, 1):
        m = parse_tif_name(p)
        dst = os.path.join(args.outdir, os.path.basename(p)) if args.outdir else None

        if m["laser"] not in channels:
            # Not corrected, but still measured -- green is the reference channel and the one that
            # clips most (up to ~380 px/frame), so it must not be a blind spot in the QC.  Count
            # its saturated beads too: green fiducials are the brightest objects and saturate most,
            # so leaving this uncounted would hide exactly the beads the eye sees clipped.
            crow = {"file": os.path.basename(p), "lane": m["lane"], "pos": m["pos"],
                    "dye": m["laser"], "cycle": m["cycle"], "action": "copied"}
            try:
                cimg, cxmp = read_tif(p)
                csat = cimg >= _SATURATION_LEVEL
                crow["n_saturated_px"] = int(csat.sum())
                crow["saturated"] = int(crow["n_saturated_px"] > 0)
                cref = refs.get((m["lane"], m["pos"]), np.empty((0, 2)))
                if len(cref):
                    cbeads, _ = locate_in_frame(cimg, cref)
                    crow["n_beads"] = len(cbeads)
                    crow["n_beads_saturated"] = _count_saturated_beads(cimg, cbeads, csat)
                crow["focus_locked"] = xmp_field(cxmp, "PsdInPosition")
                crow["attempt_number"] = xmp_field(cxmp, "AttemptNumber")
            except OSError:
                crow["n_saturated_px"] = -1
            rows.append(crow)
            if not args.dry_run:
                os.makedirs(args.outdir, exist_ok=True)
                shutil.copy2(p, dst)
            n_copied += 1
            continue

        ref = refs.get((m["lane"], m["pos"]), np.empty((0, 2)))
        try:
            fr = correct_file(p, ref, target, target_sigma, args.method, reg, args.stamp_half,
                              args.max_noise_gain, measure_only=args.measure_only)
        except Exception as e:  # a bad frame must not kill the run
            rows.append({"file": os.path.basename(p), "lane": m["lane"], "pos": m["pos"],
                         "dye": m["laser"], "cycle": m["cycle"], "action": f"ERROR: {e}"})
            warnings.warn(f"{os.path.basename(p)}: {e}")
            continue
        rows.append(fr.row)

        if not args.dry_run:
            if fr.corrected is None:  # passthrough
                os.makedirs(args.outdir, exist_ok=True)
                shutil.copy2(p, dst)
                n_copied += 1
            else:
                write_tif(dst, fr.corrected, fr.xmp)
                n_corrected += 1
        elif fr.corrected is not None:
            n_corrected += 1

        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)

    import csv
    cols = ["file", "lane", "pos", "dye", "cycle", "n_candidates", "n_beads", "n_beads_saturated",
            "n_saturated_px", "saturated", "drift_px", "sigma_before", "sigma_after", "lambda",
            "noise_gain_predicted", "noise_gain", "clipped_frac", "focus_locked",
            "attempt_number", "z_position", "action", "note"]
    report_path = args.report
    if args.outdir and not os.path.isabs(report_path):
        os.makedirs(args.outdir, exist_ok=True)
        report_path = os.path.join(args.outdir, report_path)
    with open(report_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    sb = [r["sigma_before"] for r in rows if isinstance(r.get("sigma_before"), float)]
    sa = [r["sigma_after"] for r in rows if isinstance(r.get("sigma_after"), float)]
    print(f"\ncorrected {n_corrected}, copied/passthrough {n_copied}")
    if sb:
        print(f"sigma before: median {np.median(sb):.2f} px  (n={len(sb)})")
    if sa:
        print(f"sigma after : median {np.median(sa):.2f} px  (target {target_sigma:.2f})")
    print(f"report -> {report_path}")
    if args.dry_run:
        print("dry run: no pixels written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
