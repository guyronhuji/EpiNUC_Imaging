# %% [markdown]
# # ND2 Multichannel Nucleosome / PTM Colocalization & Time-Point Cumulative Counting
#
# This notebook analyzes ND2 microscopy files containing multiple fields of view (scenes /
# positions) and RGB fluorescence channels. For each field of view it:
#
# 1. Loads ND2 files **lazily** (memory-safe, one plane at a time).
# 2. Extracts fluorescence channels and (optionally) z-projects.
# 3. Detects **marker beads** and uses them to **register/drift-correct** channels.
# 4. Detects diffraction-limited spots:
#    - **nucleosomes** in the Green channel
#    - **R PTMs** in the Red channel
#    - **B PTMs** in the Blue channel
# 5. Counts colocalizations (R↔nucleosome, B↔nucleosome, and R+B+nucleosome triples).
# 6. **Aligns matching fields of view across time points / ND2 files.**
# 7. For later time points, counts only **newly observed** nucleosomes / colocalization
#    events (accounting for imaging efficiency, photobleaching and repeated acquisition).
# 8. Exports every result table to CSV.
# 9. Produces QC plots at each stage.
#
# ---
#
# ### Scientific notes (read before trusting the numbers)
#
# - **Spot-detection thresholds must be tuned on representative images.** The automatic
#   thresholds here are starting points — inspect the QC overlays and adjust
#   `SPOT_DETECTION_THRESHOLD` / `BEAD_DETECTION_THRESHOLD`.
# - **Colocalization radius** should reflect your localization precision and optical
#   resolution, not an arbitrary pixel count. Convert with `PIXEL_SIZE_UM` if known.
# - **Bead-based registration must be validated visually** (see the before/after overlays).
# - **Cumulative "new-only" counting assumes stable field-of-view alignment across time.**
#   If the stage drifts a lot between rounds, verify the across-time overlays first.
# - **Validate against manual annotations** on a subset of images before scaling up.
# - **Memory-safe iterative processing** is essential — raw arrays are released after
#   coordinates are extracted; only compact tables are accumulated.
# - **Bright beads leave a "halo" in the background estimate** — a large-sigma Gaussian blur
#   assumes a smoothly-varying background, which a compact bright bead violates. Confirmed
#   beads are masked out of the background-blur input only (`MASK_BEADS_BEFORE_BACKGROUND`,
#   `mask_beads_for_background`) so this doesn't suppress real spots sitting near a bead.

# %% [markdown]
# ## 1. Installation and imports
#
# Run the install cell once (uncomment it). Core dependencies:
# `nd2, numpy, pandas, matplotlib, scipy, scikit-image, scikit-learn, tifffile`.
#
# Optional alternatives you may prefer: `aicsimageio`, `nd2reader`, `trackpy`, `napari`.

# %%
# Uncomment to install (run once). Restart the kernel afterwards if packages were new.
# !pip install nd2 numpy pandas matplotlib scipy scikit-image scikit-learn tifffile tqdm joblib
# Optional speed accelerator (only for BACKGROUND_BACKEND="cv2"/"auto"): pip install opencv-python-headless

# %%
import os
import gc
import glob
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Iterator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

from skimage.feature import blob_log, peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import AffineTransform, warp, estimate_transform
from skimage.registration import phase_cross_correlation

# tqdm for nested progress bars; degrade to a no-op passthrough if it is not installed.
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from scipy.ndimage import zoom as _ndzoom  # fast background down/up-sampling fallback

# OpenCV is an OPTIONAL accelerator for the (dominant) background-subtraction blur. If it is
# not installed the pipeline transparently falls back to a down-sampled scipy blur.
try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    _HAVE_CV2 = False

# nd2 is the primary reader. We import lazily so the rest of the notebook can still be
# read/inspected even if it is not yet installed.
try:
    import nd2
    _HAVE_ND2 = True
except Exception as _e:  # pragma: no cover - environment dependent
    nd2 = None
    _HAVE_ND2 = False
    warnings.warn(f"'nd2' package not importable ({_e}). Install it before running.")

warnings.filterwarnings("ignore", category=UserWarning, module="skimage")
np.random.seed(0)
print("Imports OK. nd2 available:", _HAVE_ND2)

# %% [markdown]
# ## 2. Configuration
#
# Every knob lives here. Defaults are executable; `None` means "estimate automatically —
# tune after looking at the QC figures". Channel entries accept **either** a channel name
# (string, matched case-insensitively) **or** a 0-based integer index.

# %%
DATA_DIR = "./T50_20260225"       # TODO: folder containing your ND2 files
OUTPUT_DIR = "./analysis_output"  # TODO: change if desired

# Channel roles -> actual ND2 channel names (or 0-based indices). For this dataset the
# acquisition colours map as: Cy3=Green, Cy5=Red, 488=Blue.
CHANNEL_MAP = {
    "nucleosome": "TIRF Cy3",  # Green  -> nucleosomes
    "R_PTM": "TIRF Cy5",       # Red    -> R PTMs
    "B_PTM": "TIRF 488",       # Blue   -> B PTMs
    "beads": None,             # no dedicated bead channel; beads detected per-channel
}

PIXEL_SIZE_UM = 0.0722  # measured voxel size for this dataset (µm/px)

Z_PROJECTION_METHOD = "max"   # "max", "mean", or "single_plane"
Z_PLANE_INDEX = 0

BEAD_DETECTION_SIGMA = 2.0
BEAD_DETECTION_THRESHOLD = None  # None -> estimate automatically; tune after QC
# "fast" = background-subtract + single-scale local maxima (default; ~4x faster and, on
# this data, finds the same beads as LoG). "log" = multi-scale Laplacian-of-Gaussian.
# NOTE: a manual BEAD_DETECTION_THRESHOLD is an absolute smoothed-intensity value for
# "fast", but a relative 0-1 value for "log".
BEAD_DETECTION_METHOD = "fast"

# Fiducial beads are bright in every channel, so they get picked up as "spots" too and
# create false colocalizations (an automatic triple at each bead). Remove any spot within
# this many pixels of a detected bead before counting. Beads are large blobs, so use a
# radius a bit larger than a single-molecule PSF. Set EXCLUDE_BEADS_FROM_SPOTS=False to
# disable (e.g. if beads live in a dedicated channel that is not a signal channel).
EXCLUDE_BEADS_FROM_SPOTS = True
BEAD_EXCLUSION_RADIUS_PX = 5.0
# Real fiducial beads are few (~10-50) and appear in ALL channels at the same spot, whereas
# nucleosome spots appear only in green. When a channel has no clear bright beads, the bead
# detector can flood (thousands of "beads") and then bead-exclusion would delete real spots.
# Guard against this: (1) a channel with more than MAX_TRUSTED_BEADS detections is treated as
# untrustworthy/flooded, and (2) the beads used for exclusion/alignment are those CONFIRMED
# across >=2 channels (see confirmed_bead_coords). Raise if you genuinely have many beads.
MAX_TRUSTED_BEADS = 200
CONFIRM_BEADS_CROSS_CHANNEL = True

# Fiducial beads are bright, compact, and violate the "smoothly varying background"
# assumption behind the large-sigma Gaussian blur used for background estimation in
# detect_spots(): the blur spreads a bead's brightness into its own background estimate,
# producing a visible negative-value "halo" ring around the bead in the background-
# subtracted image. Real spots sitting in that ring get their local background over-
# estimated and their apparent intensity suppressed -- before the separate, distance-based
# bead-exclusion step (below) ever gets a chance to remove the bead's own spurious spot.
# To prevent this, each CONFIRMED bead's footprint is replaced with the median of its
# surrounding annulus (computed from the untouched raw image) before the background blur
# is computed -- ONLY for that purpose. The residual, threshold, and peak-finding still use
# the true, unmasked image everywhere else, so a bead's own bright spot is untouched (it is
# still removed afterwards by EXCLUDE_BEADS_FROM_SPOTS as before).
MASK_BEADS_BEFORE_BACKGROUND = True
BEAD_BACKGROUND_MASK_RADIUS_PX = 10.0

SPOT_DETECTION_SIGMA = {
    "nucleosome": 1.2,
    "R_PTM": 1.2,
    "B_PTM": 1.2,
}

# Spot-detection threshold. The default is a NOISE-RELATIVE (SNR) threshold: each image's
# robust noise level is estimated (MAD of the background-subtracted, smoothed image) and the
# threshold is set at k * noise. This adapts to each channel's and each sample's noise, so a
# noisy channel (e.g. Cy5) is NOT flooded with false positives the way a fixed percentile is.
# k is per channel (SPOT_DETECTION_SNR), calibrated so per-FOV/timepoint spot counts land in
# the reference (CellProfiler) regime: Cy3/nucleosome ~5000, Cy5/R ~300-700, 488/B ~500-1200.
# k is an EMPIRICAL sensitivity knob (local maxima sit above the pixel noise, so useful
# values are larger than a nominal 5-sigma): raise k to detect fewer spots, lower for more.
SPOT_DETECTION_SNR = {
    "nucleosome": 8.0,
    "R_PTM": 12.0,
    "B_PTM": 12.0,
}

# Optional hard override: set a channel to an absolute threshold (compared against the
# background-subtracted, smoothed image) to bypass the SNR estimate. None -> use SNR above.
SPOT_DETECTION_THRESHOLD = {
    "nucleosome": None,
    "R_PTM": None,
    "B_PTM": None,
}

COLOCALIZATION_RADIUS_PX = 3.0
TIME_ALIGNMENT_RADIUS_PX = 3.0  # "same spot across rounds" tolerance AFTER drift correction

# Across-time stage drift on this instrument is a small rigid shift (~2-12 px) that is
# larger than TIME_ALIGNMENT_RADIUS_PX and defeats plain nearest-neighbour matching /
# phase-correlation. We first recover that shift by voting over all pairwise fiducial
# offsets (robust to large drift, no image-content dependence), then fine-match.
TIME_COARSE_ALIGN = True
VOTING_BIN_PX = 1.0        # offset histogram bin for the coarse vote
VOTING_MAX_POINTS = 400    # cap points fed to the vote (brightest kept) for speed

MIN_BEADS_FOR_REGISTRATION = 3
SAVE_QC_FIGURES = True
MAX_QC_IMAGES_PER_FILE = 5

REGISTRATION_MODE = "translation"  # "translation" or "affine"
REFERENCE_CHANNEL = "nucleosome"

MEMORY_SAFE_MODE = True

# Parallelism for process_sample_parallel: each worker handles a disjoint set of FOVs
# (a whole FOV time-series stays in one worker, so cumulative counting is unaffected).
# -1 = all cores, -2 = all but one (keeps the machine responsive), 1 = serial.
N_JOBS = -2

# -------------------- Optional performance accelerators (OFF by default) --------------------
# All default to the exact current behaviour; opt in per knob, or call enable_fast_mode().
#
# Background-subtraction blur backend (the dominant cost). "scipy" = current skimage Gaussian
# (bit-identical). "cv2" = OpenCV GaussianBlur (5-10x faster, tiny numeric difference).
# "downsample" = blur a down-scaled image then upsample (~10x faster, tiny difference).
# "auto" = cv2 if available else downsample. Non-scipy backends change spot counts slightly
# (validate before trusting), so the default stays "scipy".
BACKGROUND_BACKEND = "scipy"
BACKGROUND_DOWNSCALE = 4        # only used by "downsample"/"auto"

# Cap BLAS/threadpool threads inside each parallel worker to avoid oversubscription. None =
# leave as-is (current behaviour); 1 = one thread/worker (bit-identical, often 10-30% faster
# when many workers run). Used by process_sample_parallel.
PARALLEL_INNER_THREADS = None

# Keep each ND2 file open per worker process instead of reopening it for every FOV (fewer
# opens on big parallel runs). Bit-identical results. False = current behaviour.
USE_ND2_HANDLE_CACHE = False

# Bead matching tolerance between channels (px). Beads are near-identical across channels,
# so an initial nearest-neighbour search within this radius pairs them up.
BEAD_MATCH_MAX_DIST_PX = 8.0

# How to order ND2 files along time. "filename" sorts alphabetically; supply an explicit
# ordered list of paths to control it precisely.
TIME_ORDER = "filename"  # or a list like ["round1.nd2", "round2.nd2", ...]

os.makedirs(OUTPUT_DIR, exist_ok=True)
QC_DIR = os.path.join(OUTPUT_DIR, "qc_figures")
os.makedirs(QC_DIR, exist_ok=True)
print("Config loaded. Output ->", os.path.abspath(OUTPUT_DIR))

# %% [markdown]
# ## 3. ND2 loading utilities
#
# ND2 dimension order varies between files, so we never hard-code axis positions. We open
# each file with `nd2.ND2File`, read its **named** sizes (`P`=position/scene, `T`=time,
# `C`=channel, `Z`, `Y`, `X`) and pull one 2D plane at a time via a lazy xarray/dask view.
# Raw pixel data for a whole file is never held in memory at once.

# %%
def list_nd2_files(data_dir: str) -> List[str]:
    """Return a sorted list of .nd2 file paths in ``data_dir`` (validates the path)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR does not exist: {os.path.abspath(data_dir)}")
    files = sorted(glob.glob(os.path.join(data_dir, "*.nd2")))
    if not files:
        warnings.warn(f"No .nd2 files found in {os.path.abspath(data_dir)}")
    return files


def order_files_by_time(files: List[str], time_order) -> List[str]:
    """Order files along the time axis according to ``TIME_ORDER``.

    If ``time_order`` is the string 'filename', sort alphabetically. If it is an explicit
    list, match by basename (falling back to any provided full paths). Unlisted files are
    appended in sorted order so nothing is silently dropped.
    """
    if isinstance(time_order, (list, tuple)):
        by_base = {os.path.basename(f): f for f in files}
        ordered = []
        for entry in time_order:
            base = os.path.basename(entry)
            if base in by_base:
                ordered.append(by_base.pop(base))
            elif entry in files:
                ordered.append(entry)
                by_base.pop(os.path.basename(entry), None)
            else:
                warnings.warn(f"TIME_ORDER entry not found among files: {entry}")
        ordered.extend(sorted(by_base.values()))
        return ordered
    return sorted(files)


class ND2Accessor:
    """Lazy, memory-safe accessor for a single ND2 file.

    Opens the file, exposes its named sizes and channel names, and returns individual 2D
    planes on demand. Use as a context manager so the underlying file handle is closed::

        with ND2Accessor(path) as acc:
            plane = acc.get_plane(scene=0, time=0, channel_index=1)

    Only the requested slice is materialised into a NumPy array; the rest stays lazy.
    """

    def __init__(self, path: str):
        if not _HAVE_ND2:
            raise ImportError("The 'nd2' package is required. pip install nd2")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self.path = path
        self._f = nd2.ND2File(path)
        # sizes is an ordered dict of {axis_name: length}, e.g. {'P':3,'T':2,'C':3,'Y':512,'X':512}
        self.sizes: Dict[str, int] = dict(self._f.sizes)
        self._xarr = self._f.to_xarray(delayed=True, squeeze=False)
        self.channel_names = self._read_channel_names()

    # -- context manager plumbing --------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    # -- metadata helpers ----------------------------------------------------------
    def _read_channel_names(self) -> List[str]:
        """Best-effort extraction of channel display names; falls back to C0, C1, ..."""
        n_c = self.sizes.get("C", 1)
        names = []
        try:
            for ch in self._f.metadata.channels:
                names.append(str(ch.channel.name))
        except Exception:
            names = []
        if len(names) != n_c:
            names = [f"C{i}" for i in range(n_c)]
        return names

    @property
    def n_scenes(self) -> int:
        return self.sizes.get("P", 1)

    @property
    def n_times(self) -> int:
        return self.sizes.get("T", 1)

    @property
    def n_z(self) -> int:
        return self.sizes.get("Z", 1)

    def resolve_channel(self, key) -> Optional[int]:
        """Resolve a CHANNEL_MAP entry (name/index/None) to a 0-based channel index.

        Returns ``None`` if the key is ``None`` (channel not used) or cannot be matched.
        """
        if key is None:
            return None
        if isinstance(key, (int, np.integer)):
            idx = int(key)
            return idx if 0 <= idx < self.sizes.get("C", 1) else None
        key_l = str(key).lower()
        for i, name in enumerate(self.channel_names):
            if str(name).lower() == key_l:
                return i
        # partial / contains match (e.g. "488" inside "EGFP 488")
        for i, name in enumerate(self.channel_names):
            if key_l in str(name).lower():
                return i
        return None

    def get_plane(self, scene: int = 0, time: int = 0, channel_index: int = 0,
                  z_index: Optional[int] = None) -> np.ndarray:
        """Return one 2D (Y, X) float32 plane, materialising only that slice."""
        sel = {}
        if "P" in self.sizes:
            sel["P"] = scene
        if "T" in self.sizes:
            sel["T"] = time
        if "C" in self.sizes:
            sel["C"] = channel_index
        if "Z" in self.sizes and z_index is not None:
            sel["Z"] = z_index
        sub = self._xarr.isel(**sel)
        arr = np.asarray(sub.values, dtype=np.float32)
        return np.squeeze(arr)

    def get_zstack(self, scene: int = 0, time: int = 0, channel_index: int = 0) -> np.ndarray:
        """Return a (Z, Y, X) stack for one channel (or (Y, X) if there is no Z axis)."""
        sel = {}
        if "P" in self.sizes:
            sel["P"] = scene
        if "T" in self.sizes:
            sel["T"] = time
        if "C" in self.sizes:
            sel["C"] = channel_index
        sub = self._xarr.isel(**sel)
        return np.asarray(sub.values, dtype=np.float32).squeeze()

# %% [markdown]
# ## 4. Metadata inspection utilities
#
# Before analysing anything, inspect one file to confirm which axes exist and which channel
# name maps to nucleosome / R PTM / B PTM / beads. Adapt `CHANNEL_MAP` if the printout does
# not match your expectation.

# %%
def describe_nd2(path: str) -> dict:
    """Print and return a compact description of an ND2 file's axes and channels."""
    with ND2Accessor(path) as acc:
        info = {
            "path": path,
            "sizes": acc.sizes,
            "channel_names": acc.channel_names,
            "n_scenes": acc.n_scenes,
            "n_times": acc.n_times,
            "n_z": acc.n_z,
        }
    print(f"File: {os.path.basename(path)}")
    print(f"  axes/sizes : {info['sizes']}")
    print(f"  channels   : {info['channel_names']}")
    print(f"  scenes (P) : {info['n_scenes']}   times (T): {info['n_times']}   z: {info['n_z']}")
    return info


def check_channel_map(path: str, channel_map: dict) -> pd.DataFrame:
    """Report how each CHANNEL_MAP entry resolves against a real file."""
    rows = []
    with ND2Accessor(path) as acc:
        for role, key in channel_map.items():
            idx = acc.resolve_channel(key)
            resolved_name = acc.channel_names[idx] if idx is not None else None
            rows.append({
                "role": role, "requested": key,
                "resolved_index": idx, "resolved_name": resolved_name,
                "status": "OK" if (idx is not None or key is None) else "NOT FOUND",
            })
    df = pd.DataFrame(rows)
    return df

# %% [markdown]
# ## 5. Channel extraction & z-projection utilities
#
# `extract_channel_image` returns a single 2D image for a role (`nucleosome`, `R_PTM`,
# `B_PTM`, `beads`), applying the configured z-projection. It returns `None` if the role's
# channel is not present, so downstream code can skip missing channels gracefully.

# %%
def project_zstack(stack: np.ndarray, method: str, z_index: int) -> np.ndarray:
    """Reduce a (Z, Y, X) stack to a 2D image using ``method``.

    ``method`` is one of {"max", "mean", "single_plane"}. A 2D input is returned as-is.
    """
    if stack.ndim == 2:
        return stack.astype(np.float32)
    if method == "max":
        return stack.max(axis=0).astype(np.float32)
    if method == "mean":
        return stack.mean(axis=0).astype(np.float32)
    if method == "single_plane":
        z = int(np.clip(z_index, 0, stack.shape[0] - 1))
        return stack[z].astype(np.float32)
    raise ValueError(f"Unknown Z_PROJECTION_METHOD: {method}")


def extract_channel_image(acc: "ND2Accessor", role: str, scene: int, time: int,
                          channel_map: dict = None) -> Optional[np.ndarray]:
    """Return a 2D image for ``role`` at (scene, time), or None if the channel is absent.

    Handles z-stacks per Z_PROJECTION_METHOD. Only the needed slice/stack is read.
    """
    channel_map = channel_map if channel_map is not None else CHANNEL_MAP
    key = channel_map.get(role, None)
    ci = acc.resolve_channel(key)
    if ci is None:
        return None
    if acc.n_z > 1 and Z_PROJECTION_METHOD != "single_plane":
        stack = acc.get_zstack(scene=scene, time=time, channel_index=ci)
        return project_zstack(stack, Z_PROJECTION_METHOD, Z_PLANE_INDEX)
    z_index = Z_PLANE_INDEX if acc.n_z > 1 else None
    return acc.get_plane(scene=scene, time=time, channel_index=ci, z_index=z_index)

# %% [markdown]
# ## 6. Marker bead detection
#
# Beads are bright, roughly diffraction-limited fiducials. We detect them with a
# Laplacian-of-Gaussian blob detector after light background subtraction. If no beads are
# found we return an **empty DataFrame with the expected columns** rather than raising.

# %%
BEAD_COLUMNS = ["file", "scene", "time", "channel", "y", "x", "intensity"]


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Large-sigma Gaussian blur via the configured BACKGROUND_BACKEND.

    "scipy" (default) is bit-identical to the original skimage call. "cv2" and "downsample"
    are faster but change the result by a tiny amount (validate before trusting). Only the
    large background blur uses this; small spot-scale smoothing stays exact.
    """
    backend = BACKGROUND_BACKEND
    if backend == "auto":
        backend = "cv2" if _HAVE_CV2 else "downsample"
    if backend == "cv2" and _HAVE_CV2:
        k = int(2 * np.ceil(4 * sigma) + 1)  # match scipy's truncate=4 kernel extent
        return cv2.GaussianBlur(np.asarray(img, np.float32), (k, k),
                                sigmaX=float(sigma), sigmaY=float(sigma),
                                borderType=cv2.BORDER_REFLECT)
    if backend == "downsample":
        f = max(1, int(BACKGROUND_DOWNSCALE))
        if f == 1:
            return gaussian(img, sigma=sigma, preserve_range=True)
        small = _ndzoom(img, 1.0 / f, order=1)                       # coarse background
        smallb = gaussian(small, sigma=sigma / f, preserve_range=True)
        up = _ndzoom(smallb, (img.shape[0] / smallb.shape[0],
                              img.shape[1] / smallb.shape[1]), order=1)
        return up[:img.shape[0], :img.shape[1]].astype(np.float32)
    return gaussian(img, sigma=sigma, preserve_range=True)           # "scipy" (default)


def _subtract_background(img: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Simple large-sigma Gaussian background subtraction; clips negatives to zero."""
    bg = _blur(img, sigma)
    out = img - bg
    out[out < 0] = 0
    return out


def _auto_threshold(img: np.ndarray, fallback_percentile: float = 99.0) -> float:
    """Estimate an absolute-intensity threshold for blob detection.

    Uses Otsu when the image has usable dynamic range, else a high percentile. Values are
    scaled to the normalised [0, 1] image used by the LoG detector below.
    """
    finite = img[np.isfinite(img)]
    if finite.size == 0 or finite.max() <= finite.min():
        return 0.1
    try:
        t = threshold_otsu(finite)
        rel = (t - finite.min()) / (finite.max() - finite.min())
        if not np.isfinite(rel) or rel <= 0:
            raise ValueError
        return float(np.clip(rel, 0.02, 0.5))
    except Exception:
        p = np.percentile(finite, fallback_percentile)
        return float(np.clip((p - finite.min()) / (finite.max() - finite.min() + 1e-9), 0.02, 0.5))


def detect_beads(img: Optional[np.ndarray], file: str, scene: int, time: int, channel: str,
                 sigma: float = None, threshold: float = None,
                 method: str = None) -> pd.DataFrame:
    """Detect bead centroids and return a coordinate DataFrame (BEAD_COLUMNS).

    ``method`` "fast" (default): background subtraction + single-scale Gaussian smoothing +
    local-maxima peak finding (Otsu auto-threshold cleanly isolates bright beads from
    molecules). ``method`` "log": multi-scale Laplacian-of-Gaussian (``blob_log``), slower.
    Returns an empty DataFrame if ``img`` is None or nothing is found.
    """
    if img is None:
        return pd.DataFrame(columns=BEAD_COLUMNS)
    sigma = BEAD_DETECTION_SIGMA if sigma is None else sigma
    method = BEAD_DETECTION_METHOD if method is None else method
    proc = _subtract_background(img)

    if method == "fast":
        smoothed = gaussian(proc, sigma=sigma, preserve_range=True)
        thr = threshold if threshold is not None else BEAD_DETECTION_THRESHOLD
        if thr is None:
            pos = smoothed[smoothed > 0]
            thr = float(threshold_otsu(pos)) if pos.size else 0.0
        coords = peak_local_max(smoothed, min_distance=max(1, int(round(sigma * 3))),
                                threshold_abs=thr, exclude_border=3)
        if coords.shape[0] == 0:
            return pd.DataFrame(columns=BEAD_COLUMNS)
        ys, xs = coords[:, 0], coords[:, 1]
    else:  # "log": multi-scale Laplacian-of-Gaussian
        rng = proc.max() - proc.min()
        norm = (proc - proc.min()) / rng if rng > 0 else np.zeros_like(proc)
        thr = threshold if threshold is not None else BEAD_DETECTION_THRESHOLD
        if thr is None:
            thr = _auto_threshold(norm)
        blobs = blob_log(norm, min_sigma=max(0.5, sigma * 0.6), max_sigma=sigma * 1.6,
                         num_sigma=5, threshold=thr)
        if blobs.size == 0:
            return pd.DataFrame(columns=BEAD_COLUMNS)
        ys, xs = blobs[:, 0].astype(int), blobs[:, 1].astype(int)

    inten = img[np.clip(ys, 0, img.shape[0] - 1), np.clip(xs, 0, img.shape[1] - 1)]
    return pd.DataFrame({
        "file": file, "scene": scene, "time": time, "channel": channel,
        "y": ys.astype(float), "x": xs.astype(float), "intensity": inten,
    })[BEAD_COLUMNS]

# %% [markdown]
# ## 7. Channel registration / drift correction using beads
#
# We align **Red** and **Blue** onto the **Green/nucleosome** coordinate frame.
#
# Preferred path: detect beads in each channel, match them by nearest neighbour, then
# estimate a transform (`translation` = mean offset, or `affine` via
# `skimage.transform.estimate_transform`). If too few beads match, we **fall back** to
# `phase_cross_correlation`, and finally to identity — always logging what happened. Every
# guard checks for empty arrays before building KD-trees or estimating transforms.

# %%
TRANSFORM_COLUMNS = ["file", "scene", "time", "channel_type", "mode", "method",
                     "n_beads_matched", "tx", "ty", "success", "note"]


def match_points(ref_xy: np.ndarray, mov_xy: np.ndarray, max_dist: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mutual-nearest-neighbour matching of two (N,2) point sets within ``max_dist``.

    Returns index arrays (ref_idx, mov_idx). Empty inputs -> empty matches (no KD-tree
    is built on empty arrays).
    """
    if ref_xy is None or mov_xy is None or len(ref_xy) == 0 or len(mov_xy) == 0:
        return np.empty(0, int), np.empty(0, int)
    ref_tree = cKDTree(ref_xy)
    mov_tree = cKDTree(mov_xy)
    d_rm, j_rm = mov_tree.query(ref_xy)   # nearest mov for each ref
    d_mr, i_mr = ref_tree.query(mov_xy)   # nearest ref for each mov
    ref_idx, mov_idx = [], []
    for r, (d, m) in enumerate(zip(d_rm, j_rm)):
        if d <= max_dist and i_mr[m] == r:  # mutual + within tolerance
            ref_idx.append(r)
            mov_idx.append(m)
    return np.asarray(ref_idx, int), np.asarray(mov_idx, int)


def estimate_translation_by_voting(ref_xy: np.ndarray, mov_xy: np.ndarray,
                                   bin_px: float = None, max_points: int = None,
                                   ref_w: np.ndarray = None, mov_w: np.ndarray = None
                                   ) -> Tuple[np.ndarray, int]:
    """Robustly estimate the rigid translation mapping ``mov`` -> ``ref`` (both in (x, y)).

    Votes over every pairwise offset (ref - mov), bins them, and returns the modal offset.
    This tolerates drift far larger than the matching radius and does not depend on image
    content (unlike phase correlation). Points are optionally capped to the brightest
    ``max_points`` (via ``*_w`` weights) to bound the O(N*M) cost. Returns
    ((dx, dy), n_votes); empty inputs -> ((0, 0), 0).
    """
    bin_px = VOTING_BIN_PX if bin_px is None else bin_px
    max_points = VOTING_MAX_POINTS if max_points is None else max_points
    if ref_xy is None or mov_xy is None or len(ref_xy) == 0 or len(mov_xy) == 0:
        return np.zeros(2), 0

    def _cap(xy, w):
        if len(xy) <= max_points:
            return xy
        order = np.argsort(w)[::-1] if w is not None else np.arange(len(xy))
        return xy[order[:max_points]]

    r = _cap(np.asarray(ref_xy, float), ref_w)
    m = _cap(np.asarray(mov_xy, float), mov_w)
    offs = (r[:, None, :] - m[None, :, :]).reshape(-1, 2)   # (Nr*Nm, 2)
    keyed = np.round(offs / bin_px).astype(np.int64)
    uniq, counts = np.unique(keyed, axis=0, return_counts=True)
    j = int(np.argmax(counts))
    return uniq[j] * bin_px, int(counts[j])


@dataclass
class ChannelTransform:
    """Holds a fitted transform and how to apply it to (y, x) coordinates."""
    mode: str = "translation"
    method: str = "identity"
    success: bool = False
    n_matched: int = 0
    note: str = ""
    dy: float = 0.0            # translation in y (row)
    dx: float = 0.0            # translation in x (col)
    affine: Optional[AffineTransform] = None  # maps moving -> reference in (x, y)

    def apply_to_yx(self, yx: np.ndarray) -> np.ndarray:
        """Apply the transform to an (N,2) array of (y, x) coordinates -> reference frame."""
        if yx is None or len(yx) == 0:
            return np.empty((0, 2), float)
        yx = np.asarray(yx, float)
        if self.method == "affine" and self.affine is not None:
            xy = yx[:, ::-1]                       # (x, y)
            out_xy = self.affine(xy)               # apply in (x, y)
            return out_xy[:, ::-1]                 # back to (y, x)
        out = yx.copy()
        out[:, 0] += self.dy
        out[:, 1] += self.dx
        return out


def estimate_channel_transform(ref_beads: pd.DataFrame, mov_beads: pd.DataFrame,
                               ref_img: np.ndarray, mov_img: np.ndarray,
                               mode: str = "translation") -> ChannelTransform:
    """Estimate the transform mapping a moving channel onto the reference channel.

    Strategy: bead matching -> transform; fall back to phase cross-correlation; then
    identity. All empty/too-few cases are handled explicitly and recorded in ``note``.
    """
    ref_xy = ref_beads[["x", "y"]].to_numpy(float) if len(ref_beads) else np.empty((0, 2))
    mov_xy = mov_beads[["x", "y"]].to_numpy(float) if len(mov_beads) else np.empty((0, 2))

    # 1) Bead-based registration (preferred) -----------------------------------------
    if len(ref_xy) >= MIN_BEADS_FOR_REGISTRATION and len(mov_xy) >= MIN_BEADS_FOR_REGISTRATION:
        ri, mi = match_points(ref_xy, mov_xy, BEAD_MATCH_MAX_DIST_PX)
        n = len(ri)
        if n >= MIN_BEADS_FOR_REGISTRATION:
            src = mov_xy[mi]   # moving (x, y)
            dst = ref_xy[ri]   # reference (x, y)
            if mode == "affine" and n >= 3:
                try:
                    tf = estimate_transform("affine", src, dst)
                    if np.all(np.isfinite(tf.params)):
                        return ChannelTransform(mode=mode, method="affine", success=True,
                                                n_matched=n, affine=tf,
                                                note=f"affine from {n} matched beads")
                except Exception as e:
                    warnings.warn(f"Affine estimation failed ({e}); using translation.")
            # translation = mean offset of matched beads (dst - src)
            offset = (dst - src).mean(axis=0)  # (dx, dy)
            return ChannelTransform(mode="translation", method="beads_translation",
                                    success=True, n_matched=n,
                                    dx=float(offset[0]), dy=float(offset[1]),
                                    note=f"translation from {n} matched beads")
        fallback_note = f"only {n} beads matched (<{MIN_BEADS_FOR_REGISTRATION})"
    else:
        fallback_note = "too few beads detected for registration"

    # 2) Phase cross-correlation fallback --------------------------------------------
    if ref_img is not None and mov_img is not None:
        try:
            shift, _, _ = phase_cross_correlation(ref_img, mov_img, upsample_factor=10)
            # shift maps mov onto ref as (row=dy, col=dx)
            return ChannelTransform(mode="translation", method="phase_correlation",
                                    success=True, n_matched=0,
                                    dy=float(shift[0]), dx=float(shift[1]),
                                    note=f"{fallback_note}; fell back to phase correlation")
        except Exception as e:
            warnings.warn(f"phase_cross_correlation failed ({e}); using identity.")

    # 3) Identity (no correction) ----------------------------------------------------
    return ChannelTransform(mode="translation", method="identity", success=False,
                            n_matched=0, note=f"{fallback_note}; no correction applied")


def transform_record(file, scene, time, channel_type, tf: ChannelTransform) -> dict:
    """Flatten a ChannelTransform into a row for registration_transforms.csv."""
    return {
        "file": file, "scene": scene, "time": time, "channel_type": channel_type,
        "mode": tf.mode, "method": tf.method, "n_beads_matched": tf.n_matched,
        "tx": tf.dx, "ty": tf.dy, "success": tf.success, "note": tf.note,
    }


def warp_to_reference(mov_img: np.ndarray, tf: ChannelTransform) -> np.ndarray:
    """Warp a moving image into the reference frame (for QC overlays only)."""
    if tf.method == "affine" and tf.affine is not None:
        # warp expects the inverse map (reference -> moving)
        return warp(mov_img, tf.affine.inverse, preserve_range=True).astype(np.float32)
    at = AffineTransform(translation=(tf.dx, tf.dy))
    return warp(mov_img, at.inverse, preserve_range=True).astype(np.float32)

# %% [markdown]
# ## 8. Spot detection (nucleosomes, R PTMs, B PTMs)
#
# Pipeline per channel: background subtraction → Gaussian smoothing → local-maxima
# detection → thresholding → optional edge removal. Detected coordinates are stored in both
# the **raw** frame and the **registered** frame (after applying the channel transform).
# Empty channels yield an empty DataFrame with the expected columns.
#
# **Bead-halo guard:** the background estimate is a large-sigma Gaussian blur, which assumes
# a smoothly varying background -- a bright, compact fiducial bead violates that assumption
# and the blur spreads the bead's brightness into its own background estimate, leaving a
# negative-value "halo" ring around the bead after subtraction. If confirmed bead
# coordinates are passed in (``bead_yx``) and ``MASK_BEADS_BEFORE_BACKGROUND`` is True, each
# bead's footprint is masked out (filled with its local surrounding-annulus median) *only*
# for the background-blur input; the actual residual/threshold/peak-finding still run on the
# true, unmasked image, so this only removes the ring artifact, it does not touch real spots.

# %%
SPOT_COLUMNS = ["file", "scene", "time", "channel_type", "y", "x", "intensity",
                "registered_y", "registered_x", "spot_id"]


def mask_beads_for_background(img: np.ndarray, bead_yx: Optional[np.ndarray],
                              mask_radius: float = None,
                              sample_radius: float = None) -> np.ndarray:
    """Return a COPY of ``img`` with each bead's footprint filled by its local annulus median.

    Used only to feed the background-estimation blur in detect_spots()/
    spot_threshold_diagnostics() -- never for the residual/threshold/peak-finding themselves.
    Each bead's fill value is sampled from the surrounding annulus in the ORIGINAL image (not
    from ``out``), so nearby/overlapping beads don't compound errors from each other. Empty-
    safe: no beads or no image -> returns ``img`` unchanged (no copy made).
    """
    if img is None or bead_yx is None or len(bead_yx) == 0:
        return img
    mask_radius = BEAD_BACKGROUND_MASK_RADIUS_PX if mask_radius is None else mask_radius
    sample_radius = mask_radius * 3 if sample_radius is None else sample_radius
    out = img.copy()
    H, W = img.shape
    global_fill = float(np.median(img))
    for (y, x) in bead_yx:
        y0, y1 = max(0, int(y - sample_radius)), min(H, int(y + sample_radius) + 1)
        x0, x1 = max(0, int(x - sample_radius)), min(W, int(x + sample_radius) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        py, px = np.ogrid[y0:y1, x0:x1]
        local_disk = (py - y) ** 2 + (px - x) ** 2 <= mask_radius ** 2
        annulus_vals = img[y0:y1, x0:x1][~local_disk]  # sampled from the ORIGINAL image
        fill_value = float(np.median(annulus_vals)) if annulus_vals.size else global_fill
        out[y0:y1, x0:x1][local_disk] = fill_value
    return out


def detect_spots(img: Optional[np.ndarray], file: str, scene: int, time: int,
                 channel_type: str, transform: Optional["ChannelTransform"] = None,
                 sigma: float = None, threshold: float = None,
                 edge_margin: int = 2,
                 bead_yx: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Detect diffraction-limited spots in one channel; return a SPOT_COLUMNS DataFrame.

    ``transform`` (if given) maps raw (y, x) into the reference frame; otherwise the
    registered coordinates equal the raw ones. ``bead_yx`` (confirmed bead (y, x)
    coordinates, if available) is used to mask beads out of the background-blur input only
    (see mask_beads_for_background) when MASK_BEADS_BEFORE_BACKGROUND is True. Returns an
    empty, correctly-columned DataFrame when ``img`` is None or nothing is detected.
    """
    if img is None:
        return pd.DataFrame(columns=SPOT_COLUMNS)

    sigma = SPOT_DETECTION_SIGMA.get(channel_type, 1.2) if sigma is None else sigma
    # Background-subtract WITHOUT clipping negatives, so the residual noise stays symmetric
    # and its spread can be estimated robustly. Smooth to the spot scale. The large background
    # blur honours BACKGROUND_BACKEND; the small spot smoothing stays exact.
    img_for_bg = img
    if MASK_BEADS_BEFORE_BACKGROUND and bead_yx is not None and len(bead_yx):
        img_for_bg = mask_beads_for_background(img, bead_yx)
    bg = _blur(img_for_bg, max(6.0, sigma * 5))
    smoothed = gaussian(img - bg, sigma=sigma, preserve_range=True)

    # Threshold: explicit override, else a noise-relative (SNR) threshold. The robust noise
    # level (MAD) adapts to each channel/sample, so noisy channels are not flooded and bright
    # or atypical samples do not fail (unlike a fixed percentile).
    thr = threshold if threshold is not None else SPOT_DETECTION_THRESHOLD.get(channel_type)
    if thr is None:
        med = float(np.median(smoothed))
        noise = 1.4826 * float(np.median(np.abs(smoothed - med)))
        k = SPOT_DETECTION_SNR.get(channel_type, 8.0)
        thr = med + k * noise

    coords = peak_local_max(smoothed, min_distance=max(1, int(round(sigma))),
                            threshold_abs=thr, exclude_border=edge_margin)
    if coords.shape[0] == 0:
        return pd.DataFrame(columns=SPOT_COLUMNS)

    ys, xs = coords[:, 0], coords[:, 1]
    inten = img[ys, xs]
    reg = transform.apply_to_yx(coords.astype(float)) if transform is not None else coords.astype(float)
    df = pd.DataFrame({
        "file": file, "scene": scene, "time": time, "channel_type": channel_type,
        "y": ys, "x": xs, "intensity": inten,
        "registered_y": reg[:, 0], "registered_x": reg[:, 1],
    })
    df["spot_id"] = [f"{os.path.basename(str(file))}|s{scene}|t{time}|{channel_type}|{i}"
                     for i in range(len(df))]
    return df[SPOT_COLUMNS]

# %% [markdown]
# ### Tuning the detection threshold with a histogram
#
# By default detection uses a **noise-relative (SNR) threshold** (`SPOT_DETECTION_SNR[channel]`
# × the image's robust noise level), which adapts per channel/sample. If instead you set an
# absolute `SPOT_DETECTION_THRESHOLD[channel]`, it is compared against the intensity of each
# local maximum in the background-subtracted, smoothed image. To choose either objectively,
# collect **all** candidate maxima (threshold ≈ 0) and look at the **histogram of their peak
# intensities**:
# noise maxima pile up at low intensity, real spots form a higher shoulder/mode, and the
# **valley between them is your threshold**. The companion *spot-count-vs-threshold* curve
# shows the same thing as a "knee": counts fall steeply through the noise, then flatten over
# the real-spot plateau — the elbow sits at the noise/signal boundary.
#
# The plot marks `threshold_otsu` and the 95th/99th percentiles as *reference lines only* —
# **do not trust any single automatic value blindly.** When ~10^5 noise maxima dominate and
# a few very bright beads sit far out in the tail, Otsu is pulled up past the real spots
# (catching only beads), while log-scale/percentile picks can collapse into the noise. Read
# the value off the **histogram valley / count-curve elbow**, then **confirm it on the spot
# overlay** (`qc_spots`) before committing. Tune each channel independently — they differ.
#
# Run `spot_threshold_diagnostics(img, "nucleosome")` on a representative FOV, pick a value,
# put it in `SPOT_DETECTION_THRESHOLD`, and re-plot `qc_spots` to verify.

# %%
def spot_threshold_diagnostics(img: Optional[np.ndarray], channel_type: str = "nucleosome",
                               sigma: float = None, title: str = "",
                               fname: str = None, show: bool = True,
                               bead_yx: Optional[np.ndarray] = None):
    """Diagnose a good SPOT_DETECTION_THRESHOLD for one channel via histograms.

    Reproduces detect_spots' preprocessing, finds every local maximum (no threshold), then
    plots (1) the histogram of peak intensities and (2) spot count vs threshold, marking
    Otsu / 95th / 99th-percentile candidates. Returns (count_vs_threshold_df, peak_array).
    ``bead_yx``, if given, masks beads out of the background-blur input only (same as
    detect_spots) so this diagnostic matches production behaviour. Empty-safe: returns
    (None, empty array) if the image is missing or has no maxima.
    """
    if img is None:
        print(f"[{channel_type}] no image"); return None, np.array([])
    sigma = SPOT_DETECTION_SIGMA.get(channel_type, 1.2) if sigma is None else sigma
    img_for_bg = img
    if MASK_BEADS_BEFORE_BACKGROUND and bead_yx is not None and len(bead_yx):
        img_for_bg = mask_beads_for_background(img, bead_yx)
    bg = _blur(img_for_bg, max(6.0, sigma * 5))
    proc = img - bg
    proc[proc < 0] = 0  # matches _subtract_background()'s clipping behaviour
    smoothed = gaussian(proc, sigma=sigma, preserve_range=True)
    min_distance = max(1, int(round(sigma)))
    coords = peak_local_max(smoothed, min_distance=min_distance,
                            threshold_abs=0, exclude_border=2)
    peaks = smoothed[coords[:, 0], coords[:, 1]] if len(coords) else np.array([])
    if peaks.size == 0:
        print(f"[{channel_type}] no candidate peaks"); return None, peaks

    otsu = float(threshold_otsu(peaks))
    pct = {q: float(np.percentile(peaks, q)) for q in (90, 95, 99)}
    grid = np.linspace(peaks.min(), np.percentile(peaks, 99.5), 80)
    cvt = pd.DataFrame({"threshold": grid,
                        "n_spots": [int((peaks > t).sum()) for t in grid]})

    marks = [("Otsu", otsu, "crimson"), ("p95", pct[95], "green"), ("p99", pct[99], "navy")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(peaks, bins=120, color="0.6")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("smoothed peak intensity"); ax[0].set_ylabel("candidate count (log)")
    ax[0].set_title("Peak-intensity histogram")
    ax[1].plot(cvt["threshold"], cvt["n_spots"].clip(lower=1), "-k")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("threshold"); ax[1].set_ylabel("resulting n spots (log)")
    ax[1].set_title("Spot count vs threshold")
    for name, val, c in marks:
        ax[0].axvline(val, color=c, ls="--", label=f"{name}={val:.0f}")
        ax[1].axvline(val, color=c, ls="--", label=f"{name} -> {int((peaks > val).sum())} spots")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.suptitle(title or f"Threshold tuning - {channel_type}")
    if fname:
        _save(fig, fname)
    elif show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[{channel_type}] candidate peaks={peaks.size}  Otsu={otsu:.1f} "
          f"(-> {int((peaks > otsu).sum())} spots)  "
          f"p90={pct[90]:.1f}  p95={pct[95]:.1f}  p99={pct[99]:.1f}")
    print(f"  Pick the value in the histogram valley (often near Otsu) and set "
          f"SPOT_DETECTION_THRESHOLD['{channel_type}'].")
    return cvt, peaks

# %% [markdown]
# ### Removing beads from the spot lists
#
# Fiducial beads are bright in **all** channels, so the spot detector finds them in green,
# red and blue alike — producing a spurious nucleosome+R+B **triple** at every bead. Before
# counting, we drop any spot lying within `BEAD_EXCLUSION_RADIUS_PX` of a detected bead. We
# match against the **pooled** bead coordinates from every channel (bead positions differ by
# less than a pixel between channels, well inside the exclusion radius), so a bead missed in
# one channel's detection is still removed. Empty-safe: no beads ⇒ spots returned unchanged.

# %%
def pool_bead_coords(*bead_dfs) -> np.ndarray:
    """Stack (y, x) coordinates from several bead DataFrames into one (N, 2) array."""
    parts = [df[["y", "x"]].to_numpy(float) for df in bead_dfs if df is not None and len(df)]
    return np.vstack(parts) if parts else np.empty((0, 2), float)


def confirmed_bead_coords(bead_dfs, tol: float = None, max_trusted: int = None) -> np.ndarray:
    """Return (y, x) of beads CONFIRMED across channels — robust to a flooded channel.

    Real fiducials appear in >=2 channels at the same spot; nucleosome spots appear only in
    green, so when the green bead detector floods (thousands of false "beads") those are not
    confirmed by red/blue and are dropped. Channels with more than ``max_trusted`` detections
    are treated as untrustworthy and ignored. If fewer than 2 trustworthy channels remain,
    returns empty (exclude nothing) — far safer than deleting real spots.
    """
    if not CONFIRM_BEADS_CROSS_CHANNEL:
        return pool_bead_coords(*bead_dfs)
    tol = BEAD_MATCH_MAX_DIST_PX if tol is None else tol
    max_trusted = MAX_TRUSTED_BEADS if max_trusted is None else max_trusted
    arrs = [df[["y", "x"]].to_numpy(float) for df in bead_dfs
            if df is not None and 0 < len(df) <= max_trusted]
    if len(arrs) < 2:
        return np.empty((0, 2), float)
    arrs.sort(key=len)                       # anchor = fewest points (least likely flooded)
    anchor, others = arrs[0], arrs[1:]
    trees = [cKDTree(o) for o in others]
    keep = [p for p in anchor if any(t.query(p)[0] <= tol for t in trees)]
    return np.asarray(keep, float) if keep else np.empty((0, 2), float)


def beads_are_flooded(*bead_dfs) -> bool:
    """True if every non-empty bead channel has an implausible number of detections."""
    counts = [len(df) for df in bead_dfs if df is not None and len(df)]
    return bool(counts) and all(c > MAX_TRUSTED_BEADS for c in counts)


def _beads_df_from_coords(yx: np.ndarray) -> pd.DataFrame:
    """Wrap an (N, 2) (y, x) array as a bead DataFrame (BEAD_COLUMNS) for the align helpers."""
    if yx is None or len(yx) == 0:
        return pd.DataFrame(columns=BEAD_COLUMNS)
    return pd.DataFrame({"file": "", "scene": 0, "time": 0, "channel": "confirmed",
                         "y": yx[:, 0], "x": yx[:, 1], "intensity": 1.0})[BEAD_COLUMNS]


def exclude_spots_near_beads(spots: pd.DataFrame, bead_yx: np.ndarray,
                             radius: float = None) -> Tuple[pd.DataFrame, int]:
    """Drop spots within ``radius`` px of any bead (raw-frame coordinates).

    Returns (filtered_spots, n_removed). No-ops (returns the input) when disabled, when
    ``spots`` is empty, or when there are no beads — never builds a KD-tree on empty data.
    """
    if not EXCLUDE_BEADS_FROM_SPOTS or spots is None or len(spots) == 0 \
            or bead_yx is None or len(bead_yx) == 0:
        return spots, 0
    radius = BEAD_EXCLUSION_RADIUS_PX if radius is None else radius
    tree = cKDTree(bead_yx)
    d, _ = tree.query(spots[["y", "x"]].to_numpy(float))
    keep = d > radius
    n_removed = int((~keep).sum())
    return spots.loc[keep].reset_index(drop=True), n_removed

# %% [markdown]
# ## 9. Colocalization analysis within one image / time point
#
# Using **registered** coordinates in the nucleosome frame, a PTM colocalizes with a
# nucleosome if it lies within `COLOCALIZATION_RADIUS_PX`. A **triple** exists where a
# nucleosome has at least one R **and** at least one B PTM within the radius. We return
# event-level tables (with distances) plus summary counts, and every empty case is handled
# before any KD-tree is built.

# %%
COLOC_COLUMNS = ["file", "scene", "time", "event_type",
                 "nucleosome_y", "nucleosome_x", "partner_channel",
                 "partner_y", "partner_x",
                 "distance_R_to_nucleosome", "distance_B_to_nucleosome", "distance_R_to_B"]

COUNT_COLUMNS = ["file", "sample", "scene", "time", "n_nucleosomes", "n_R_PTMs", "n_B_PTMs",
                 "n_R_colocalized_with_nucleosome", "n_B_colocalized_with_nucleosome",
                 "n_R_B_colocalized", "n_R_B_nucleosome_triple_colocalized"]


def _reg_xy(df: pd.DataFrame) -> np.ndarray:
    """Registered (y, x) coordinates of a spot DataFrame, or an empty (0,2) array."""
    if df is None or len(df) == 0:
        return np.empty((0, 2), float)
    return df[["registered_y", "registered_x"]].to_numpy(float)


def colocalize(nuc: pd.DataFrame, r: pd.DataFrame, b: pd.DataFrame,
               file: str, scene: int, time: int,
               radius: float = None) -> Tuple[pd.DataFrame, dict]:
    """Compute colocalization events and summary counts for one image.

    Returns (events_df with COLOC_COLUMNS, counts dict with COUNT_COLUMNS). Any missing /
    empty channel yields zero for the affected categories and valid empty event rows.
    """
    radius = COLOCALIZATION_RADIUS_PX if radius is None else radius
    nuc_xy, r_xy, b_xy = _reg_xy(nuc), _reg_xy(r), _reg_xy(b)

    counts = {
        "file": file, "scene": scene, "time": time,
        "n_nucleosomes": len(nuc_xy), "n_R_PTMs": len(r_xy), "n_B_PTMs": len(b_xy),
        "n_R_colocalized_with_nucleosome": 0,
        "n_B_colocalized_with_nucleosome": 0,
        "n_R_B_colocalized": 0,
        "n_R_B_nucleosome_triple_colocalized": 0,
    }
    events: List[dict] = []

    # --- Direct R<->B colocalization, independent of the nucleosome (green) channel. -----
    # Counts every unique (R, B) PTM pair within the radius (an R with two B partners yields
    # two pairs; two R near one B yields two pairs). Computed regardless of whether a
    # nucleosome is present, so it is meaningful even in images with no nucleosomes. Each pair
    # becomes one "R_B" event row: the nucleosome_* columns carry the R-PTM coordinate and
    # partner_* the B-PTM coordinate. Cumulative de-duplication uses the pair midpoint (see
    # rb_pairs_to_ref_xy) so distinct pairs stay spatially distinguishable across time.
    if len(r_xy) and len(b_xy):
        b_tree_rb = cKDTree(b_xy)
        rb_near = b_tree_rb.query_ball_point(r_xy, radius)
        for ri, blist in enumerate(rb_near):
            for bi in blist:
                d = float(np.hypot(*(r_xy[ri] - b_xy[bi])))
                counts["n_R_B_colocalized"] += 1
                events.append({
                    "file": file, "scene": scene, "time": time, "event_type": "R_B",
                    "nucleosome_y": r_xy[ri][0], "nucleosome_x": r_xy[ri][1],
                    "partner_channel": "B_PTM",
                    "partner_y": b_xy[bi][0], "partner_x": b_xy[bi][1],
                    "distance_R_to_nucleosome": np.nan, "distance_B_to_nucleosome": np.nan,
                    "distance_R_to_B": d,
                })

    if len(nuc_xy) == 0:
        # No nucleosomes -> no nucleosome-based colocalization, but R<->B (above) still stands.
        events_df = pd.DataFrame(events, columns=COLOC_COLUMNS) if events else pd.DataFrame(columns=COLOC_COLUMNS)
        return events_df, counts

    nuc_tree = cKDTree(nuc_xy)

    # For each nucleosome, gather nearby R and B partners (empty-safe).
    r_near = nuc_tree.query_ball_point(r_xy, radius) if len(r_xy) else []
    b_near = nuc_tree.query_ball_point(b_xy, radius) if len(b_xy) else []

    # Map nucleosome index -> list of partner (channel, coord, distance)
    nuc_has_r = np.zeros(len(nuc_xy), bool)
    nuc_has_b = np.zeros(len(nuc_xy), bool)
    nuc_partners: Dict[int, dict] = {i: {"R": None, "B": None} for i in range(len(nuc_xy))}

    for ri, nlist in enumerate(r_near):
        for ni in nlist:
            d = float(np.hypot(*(r_xy[ri] - nuc_xy[ni])))
            counts["n_R_colocalized_with_nucleosome"] += 1
            nuc_has_r[ni] = True
            if nuc_partners[ni]["R"] is None or d < nuc_partners[ni]["R"][1]:
                nuc_partners[ni]["R"] = (r_xy[ri], d)
            events.append({
                "file": file, "scene": scene, "time": time, "event_type": "R_nucleosome",
                "nucleosome_y": nuc_xy[ni][0], "nucleosome_x": nuc_xy[ni][1],
                "partner_channel": "R_PTM",
                "partner_y": r_xy[ri][0], "partner_x": r_xy[ri][1],
                "distance_R_to_nucleosome": d, "distance_B_to_nucleosome": np.nan,
                "distance_R_to_B": np.nan,
            })

    for bi, nlist in enumerate(b_near):
        for ni in nlist:
            d = float(np.hypot(*(b_xy[bi] - nuc_xy[ni])))
            counts["n_B_colocalized_with_nucleosome"] += 1
            nuc_has_b[ni] = True
            if nuc_partners[ni]["B"] is None or d < nuc_partners[ni]["B"][1]:
                nuc_partners[ni]["B"] = (b_xy[bi], d)
            events.append({
                "file": file, "scene": scene, "time": time, "event_type": "B_nucleosome",
                "nucleosome_y": nuc_xy[ni][0], "nucleosome_x": nuc_xy[ni][1],
                "partner_channel": "B_PTM",
                "partner_y": b_xy[bi][0], "partner_x": b_xy[bi][1],
                "distance_R_to_nucleosome": np.nan, "distance_B_to_nucleosome": d,
                "distance_R_to_B": np.nan,
            })

    # Triple colocalizations: nucleosomes with both an R and a B partner.
    triple_idx = np.where(nuc_has_r & nuc_has_b)[0]
    counts["n_R_B_nucleosome_triple_colocalized"] = int(len(triple_idx))
    for ni in triple_idx:
        r_coord, dr = nuc_partners[ni]["R"]
        b_coord, db = nuc_partners[ni]["B"]
        events.append({
            "file": file, "scene": scene, "time": time, "event_type": "R_B_nucleosome_triple",
            "nucleosome_y": nuc_xy[ni][0], "nucleosome_x": nuc_xy[ni][1],
            "partner_channel": "R_PTM+B_PTM",
            "partner_y": np.nan, "partner_x": np.nan,
            "distance_R_to_nucleosome": dr, "distance_B_to_nucleosome": db,
            "distance_R_to_B": float(np.hypot(*(r_coord - b_coord))),
        })

    events_df = pd.DataFrame(events, columns=COLOC_COLUMNS) if events else pd.DataFrame(columns=COLOC_COLUMNS)
    return events_df, counts

# %% [markdown]
# ## 10. Across-time image alignment
#
# Different ND2 files/time points may image the same field of view. We map each later time
# point onto the **reference (earliest) time point** for the same scene. Preference order:
# matched **beads** → matched **nucleosome** spots → **phase cross-correlation** on the
# nucleosome images → identity. Empty/too-few cases fall back and are logged.

# %%
def _align_points_coarse_then_fine(ref_xy, mov_xy, ref_w, mov_w, mode, source_label):
    """Coarse rigid vote -> apply -> fine nearest-neighbour match -> refined transform.

    Returns a successful ChannelTransform, or None if there are too few points / inliers.
    ``ref_xy``/``mov_xy`` are (N, 2) in (x, y). This is the piece that makes across-time
    alignment robust to the several-pixel stage drift measured on real data.
    """
    if ref_xy is None or mov_xy is None \
            or len(ref_xy) < MIN_BEADS_FOR_REGISTRATION or len(mov_xy) < MIN_BEADS_FOR_REGISTRATION:
        return None
    # 1) Coarse translation from pairwise-offset voting (tolerates large drift).
    if TIME_COARSE_ALIGN:
        coarse, votes = estimate_translation_by_voting(ref_xy, mov_xy, ref_w=ref_w, mov_w=mov_w)
    else:
        coarse, votes = np.zeros(2), 0
    moved = mov_xy + coarse
    # 2) Fine nearest-neighbour match near the corrected positions.
    ri, mi = match_points(ref_xy, moved, TIME_ALIGNMENT_RADIUS_PX)
    if len(ri) < MIN_BEADS_FOR_REGISTRATION:
        return None
    src, dst = mov_xy[mi], ref_xy[ri]   # original moving vs reference (full transform)
    if mode == "affine" and len(ri) >= 3:
        try:
            aff = estimate_transform("affine", src, dst)
            if np.all(np.isfinite(aff.params)):
                return ChannelTransform(mode="affine", method="affine", success=True,
                                        n_matched=len(ri), affine=aff,
                                        note=f"time-align via {source_label} (vote+affine, "
                                             f"coarse=({coarse[0]:+.1f},{coarse[1]:+.1f}), "
                                             f"{len(ri)} inliers)")
        except Exception:
            pass
    offset = (dst - src).mean(axis=0)   # total translation = coarse + residual
    return ChannelTransform(mode="translation", method=f"{source_label}_vote_translation",
                            success=True, n_matched=len(ri),
                            dx=float(offset[0]), dy=float(offset[1]),
                            note=f"time-align via {source_label} (vote+fine, "
                                 f"coarse=({coarse[0]:+.1f},{coarse[1]:+.1f}), {len(ri)} inliers)")


def estimate_time_transform(ref_beads: pd.DataFrame, mov_beads: pd.DataFrame,
                            ref_nuc: pd.DataFrame, mov_nuc: pd.DataFrame,
                            ref_img: Optional[np.ndarray], mov_img: Optional[np.ndarray],
                            mode: str = "translation") -> ChannelTransform:
    """Estimate the transform mapping a later time point onto the reference time point.

    Order: (1) fiducial beads with coarse-vote + fine match, (2) nucleosome constellation
    with the same, (3) phase correlation (last resort; unreliable when photobleaching
    changes image content), (4) identity. Every step is empty-safe.
    """
    def _xyw(df, xcol, ycol):
        if df is None or len(df) == 0:
            return np.empty((0, 2)), None
        return (df[[xcol, ycol]].to_numpy(float),
                df["intensity"].to_numpy(float) if "intensity" in df else None)

    # 1) Beads
    ref_xy, ref_w = _xyw(ref_beads, "x", "y")
    mov_xy, mov_w = _xyw(mov_beads, "x", "y")
    tf = _align_points_coarse_then_fine(ref_xy, mov_xy, ref_w, mov_w, mode, "beads")
    if tf is not None:
        return tf

    # 2) Nucleosome spots (registered coordinates within each time point)
    ref_xy, ref_w = _xyw(ref_nuc, "registered_x", "registered_y")
    mov_xy, mov_w = _xyw(mov_nuc, "registered_x", "registered_y")
    tf = _align_points_coarse_then_fine(ref_xy, mov_xy, ref_w, mov_w, mode, "nucleosomes")
    if tf is not None:
        return tf

    # 3) Phase correlation on nucleosome images (last resort)
    if ref_img is not None and mov_img is not None:
        try:
            shift, _, _ = phase_cross_correlation(ref_img, mov_img, upsample_factor=10)
            return ChannelTransform(mode="translation", method="phase_correlation", success=True,
                                    n_matched=0, dy=float(shift[0]), dx=float(shift[1]),
                                    note="time-align via phase correlation (fallback)")
        except Exception:
            pass

    # 4) Identity
    return ChannelTransform(mode="translation", method="identity", success=False,
                            note="time-align failed; identity (no correction)")


def apply_time_transform_to_spots(spots: pd.DataFrame, tf: ChannelTransform) -> np.ndarray:
    """Map a spot table's registered (y, x) into the reference-time frame -> (N,2) array."""
    yx = spots[["registered_y", "registered_x"]].to_numpy(float) if len(spots) else np.empty((0, 2))
    return tf.apply_to_yx(yx)

# %% [markdown]
# ## 11. Cumulative "new-only" counting across time
#
# For each scene we keep a compact cumulative set of already-counted nucleosome / event
# coordinates (in the reference-time frame — **no raw images retained**). At each new time
# point we map current detections into that frame and count only those **farther than**
# `TIME_ALIGNMENT_RADIUS_PX` from anything already seen. Empty cumulative sets ⇒ everything
# is new; empty current sets ⇒ zero new, cumulative unchanged.

# %%
CUM_COLUMNS = [
    "file", "sample", "scene", "time",
    "raw_n_nucleosomes", "new_n_nucleosomes", "cumulative_n_nucleosomes",
    "raw_n_R_PTMs", "new_n_R_PTMs", "cumulative_n_R_PTMs",
    "raw_n_B_PTMs", "new_n_B_PTMs", "cumulative_n_B_PTMs",
    "raw_n_R_colocalized_with_nucleosome", "new_n_R_colocalized_with_nucleosome",
    "cumulative_n_R_colocalized_with_nucleosome",
    "raw_n_B_colocalized_with_nucleosome", "new_n_B_colocalized_with_nucleosome",
    "cumulative_n_B_colocalized_with_nucleosome",
    "raw_n_R_B_colocalized", "new_n_R_B_colocalized", "cumulative_n_R_B_colocalized",
    "raw_n_triple_colocalized", "new_n_triple_colocalized", "cumulative_n_triple_colocalized",
]


def _new_mask(current_xy: np.ndarray, seen_xy: np.ndarray, radius: float) -> np.ndarray:
    """Boolean mask of current points with no match within ``radius`` of a seen point.

    Empty-safe: empty ``current`` -> empty mask; empty ``seen`` -> all True.
    """
    if len(current_xy) == 0:
        return np.zeros(0, bool)
    if len(seen_xy) == 0:
        return np.ones(len(current_xy), bool)
    tree = cKDTree(seen_xy)
    d, _ = tree.query(current_xy)
    return d > radius


@dataclass
class SceneCumulativeState:
    """Per-scene memory of already-counted coordinates in the reference-time frame."""
    nucleosomes: np.ndarray = field(default_factory=lambda: np.empty((0, 2), float))
    r_spots: np.ndarray = field(default_factory=lambda: np.empty((0, 2), float))   # all R PTMs
    b_spots: np.ndarray = field(default_factory=lambda: np.empty((0, 2), float))   # all B PTMs
    r_events: np.ndarray = field(default_factory=lambda: np.empty((0, 2), float))  # R-coloc
    b_events: np.ndarray = field(default_factory=lambda: np.empty((0, 2), float))  # B-coloc
    rb_events: np.ndarray = field(default_factory=lambda: np.empty((0, 2), float))  # R<->B coloc
    triples: np.ndarray = field(default_factory=lambda: np.empty((0, 2), float))


def update_cumulative(state: SceneCumulativeState,
                      nuc_ref_xy: np.ndarray,
                      r_all_ref_xy: np.ndarray, b_all_ref_xy: np.ndarray,
                      r_coloc_ref_xy: np.ndarray, b_coloc_ref_xy: np.ndarray,
                      rb_coloc_ref_xy: np.ndarray,
                      triple_ref_xy: np.ndarray,
                      radius: float = None) -> Tuple[dict, SceneCumulativeState]:
    """Fold one time point's reference-frame coordinates into the cumulative state.

    Tracks de-duplicated cumulative counts for every detected type: nucleosomes, all R PTMs,
    all B PTMs, and the R-nucleosome / B-nucleosome / triple colocalizations. All coordinate
    arrays are (N,2) in the reference-time frame. Returns (raw/new/cumulative dict, state).
    """
    radius = TIME_ALIGNMENT_RADIUS_PX if radius is None else radius

    def _fold(current, seen):
        current = np.asarray(current, float).reshape(-1, 2) if len(current) else np.empty((0, 2))
        mask = _new_mask(current, seen, radius)
        new_pts = current[mask] if len(current) else np.empty((0, 2))
        combined = np.vstack([seen, new_pts]) if len(new_pts) else seen
        return int(len(current)), int(mask.sum()), combined

    raw_n, new_n, state.nucleosomes = _fold(nuc_ref_xy, state.nucleosomes)
    raw_rp, new_rp, state.r_spots = _fold(r_all_ref_xy, state.r_spots)
    raw_bp, new_bp, state.b_spots = _fold(b_all_ref_xy, state.b_spots)
    raw_r, new_r, state.r_events = _fold(r_coloc_ref_xy, state.r_events)
    raw_b, new_b, state.b_events = _fold(b_coloc_ref_xy, state.b_events)
    raw_rb, new_rb, state.rb_events = _fold(rb_coloc_ref_xy, state.rb_events)
    raw_t, new_t, state.triples = _fold(triple_ref_xy, state.triples)

    result = {
        "raw_n_nucleosomes": raw_n, "new_n_nucleosomes": new_n,
        "cumulative_n_nucleosomes": len(state.nucleosomes),
        "raw_n_R_PTMs": raw_rp, "new_n_R_PTMs": new_rp,
        "cumulative_n_R_PTMs": len(state.r_spots),
        "raw_n_B_PTMs": raw_bp, "new_n_B_PTMs": new_bp,
        "cumulative_n_B_PTMs": len(state.b_spots),
        "raw_n_R_colocalized_with_nucleosome": raw_r,
        "new_n_R_colocalized_with_nucleosome": new_r,
        "cumulative_n_R_colocalized_with_nucleosome": len(state.r_events),
        "raw_n_B_colocalized_with_nucleosome": raw_b,
        "new_n_B_colocalized_with_nucleosome": new_b,
        "cumulative_n_B_colocalized_with_nucleosome": len(state.b_events),
        "raw_n_R_B_colocalized": raw_rb, "new_n_R_B_colocalized": new_rb,
        "cumulative_n_R_B_colocalized": len(state.rb_events),
        "raw_n_triple_colocalized": raw_t, "new_n_triple_colocalized": new_t,
        "cumulative_n_triple_colocalized": len(state.triples),
    }
    return result, state


def events_to_ref_xy(events_df: pd.DataFrame, event_type: str,
                     tf: ChannelTransform) -> np.ndarray:
    """Extract nucleosome coordinates for an event type and map them into the ref-time frame."""
    if events_df is None or len(events_df) == 0:
        return np.empty((0, 2), float)
    sub = events_df[events_df["event_type"] == event_type]
    if len(sub) == 0:
        return np.empty((0, 2), float)
    yx = sub[["nucleosome_y", "nucleosome_x"]].to_numpy(float)
    return tf.apply_to_yx(yx)


def rb_pairs_to_ref_xy(events_df: pd.DataFrame, tf: ChannelTransform) -> np.ndarray:
    """Midpoint of each R<->B pair ("R_B" events), mapped into the reference-time frame.

    Uses the pair midpoint (average of the R and B coordinates) so that distinct pairs stay
    spatially distinguishable for cumulative de-duplication, even when several pairs share the
    same R or B spot. The transform is affine, so the midpoint of the mapped points equals the
    mapped midpoint. Empty-safe.
    """
    if events_df is None or len(events_df) == 0:
        return np.empty((0, 2), float)
    sub = events_df[events_df["event_type"] == "R_B"]
    if len(sub) == 0:
        return np.empty((0, 2), float)
    r_yx = sub[["nucleosome_y", "nucleosome_x"]].to_numpy(float)
    b_yx = sub[["partner_y", "partner_x"]].to_numpy(float)
    return tf.apply_to_yx(0.5 * (r_yx + b_yx))

# %% [markdown]
# ## 12. QC visualization functions
#
# All plotters tolerate empty coordinate tables (they still show the image). Figures are
# saved under `OUTPUT_DIR/qc_figures/` when `SAVE_QC_FIGURES` is True. Plot volume is capped
# by `MAX_QC_IMAGES_PER_FILE` in the batch loop.

# %%
def _norm01(img):
    """Percentile-stretch an image to [0, 1] for display (robust to outliers)."""
    if img is None:
        return None
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _save(fig, name):
    if SAVE_QC_FIGURES:
        path = os.path.join(QC_DIR, name)
        fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def rgb_overlay(green, red, blue):
    """Stack up to three channels into an RGB image for display (missing -> zeros)."""
    shape = None
    for ch in (green, red, blue):
        if ch is not None:
            shape = ch.shape
            break
    if shape is None:
        return np.zeros((8, 8, 3))
    r = _norm01(red) if red is not None else np.zeros(shape)
    g = _norm01(green) if green is not None else np.zeros(shape)
    b = _norm01(blue) if blue is not None else np.zeros(shape)
    return np.dstack([r, g, b])


def qc_raw_channels(green, red, blue, title, fname):
    """Show the three raw channels side by side plus their RGB overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, name, cmap in zip(
        axes[:3], (green, red, blue), ("Green/nucleosome", "Red/R_PTM", "Blue/B_PTM"),
        ("Greens", "Reds", "Blues")):
        if img is not None:
            ax.imshow(_norm01(img), cmap=cmap)
        ax.set_title(name); ax.axis("off")
    axes[3].imshow(rgb_overlay(green, red, blue)); axes[3].set_title("RGB overlay"); axes[3].axis("off")
    fig.suptitle(title)
    _save(fig, fname)


def qc_beads(img, beads_df, title, fname):
    """Overlay detected bead positions on a channel image."""
    fig, ax = plt.subplots(figsize=(6, 6))
    if img is not None:
        ax.imshow(_norm01(img), cmap="gray")
    if beads_df is not None and len(beads_df):
        ax.scatter(beads_df["x"], beads_df["y"], s=60, facecolors="none",
                   edgecolors="yellow", linewidths=1.2, label=f"{len(beads_df)} beads")
        ax.legend(loc="upper right")
    ax.set_title(title); ax.axis("off")
    _save(fig, fname)


def qc_registration(green, red_raw, red_reg, blue_raw, blue_reg, title, fname):
    """Before/after RGB overlays to visually validate channel registration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb_overlay(green, red_raw, blue_raw)); axes[0].set_title("Before registration"); axes[0].axis("off")
    axes[1].imshow(rgb_overlay(green, red_reg, blue_reg)); axes[1].set_title("After registration"); axes[1].axis("off")
    fig.suptitle(title)
    _save(fig, fname)


def qc_spots(img, spots_df, color, title, fname):
    """Overlay detected spots on their channel image."""
    fig, ax = plt.subplots(figsize=(6, 6))
    if img is not None:
        ax.imshow(_norm01(img), cmap="gray")
    if spots_df is not None and len(spots_df):
        ax.scatter(spots_df["x"], spots_df["y"], s=20, marker="+", c=color,
                   label=f"{len(spots_df)} spots")
        ax.legend(loc="upper right")
    ax.set_title(title); ax.axis("off")
    _save(fig, fname)


def qc_colocalization(green, nuc, r, b, events_df, title, fname):
    """Overlay nucleosome/R/B spots and mark colocalization events."""
    fig, ax = plt.subplots(figsize=(7, 7))
    if green is not None:
        ax.imshow(_norm01(green), cmap="gray")
    for df, c, lbl in ((nuc, "lime", "nucleosome"), (r, "red", "R_PTM"), (b, "deepskyblue", "B_PTM")):
        if df is not None and len(df):
            ax.scatter(df["registered_x"], df["registered_y"], s=14, c=c, label=lbl, alpha=0.7)
    if events_df is not None and len(events_df):
        trip = events_df[events_df["event_type"] == "R_B_nucleosome_triple"]
        if len(trip):
            ax.scatter(trip["nucleosome_x"], trip["nucleosome_y"], s=110, facecolors="none",
                       edgecolors="yellow", linewidths=1.6, label="triple")
    ax.legend(loc="upper right", fontsize=8); ax.set_title(title); ax.axis("off")
    _save(fig, fname)


def qc_time_alignment(ref_xy, cur_new_xy, cur_old_xy, title, fname):
    """Show reference (already-counted) vs current new/old detections after time alignment."""
    fig, ax = plt.subplots(figsize=(7, 7))
    if len(ref_xy):
        ax.scatter(ref_xy[:, 1], ref_xy[:, 0], s=30, c="gray", alpha=0.5, label="previously counted")
    if len(cur_old_xy):
        ax.scatter(cur_old_xy[:, 1], cur_old_xy[:, 0], s=18, c="orange", label="current (already seen)")
    if len(cur_new_xy):
        ax.scatter(cur_new_xy[:, 1], cur_new_xy[:, 0], s=18, c="magenta", label="current (NEW)")
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    _save(fig, fname)

# %% [markdown]
# ## 13. Memory-safe batch processing
#
# The driver iterates **file → scene → time**, processing one field of view at a time. For
# each it: extracts channels, detects beads, registers Red/Blue onto Green, detects spots,
# colocalizes, aligns across time, and updates the cumulative new-only counts. Raw arrays go
# out of scope (and `gc.collect()` runs) after coordinates + QC are extracted — only compact
# tables are accumulated. Results can be flushed to CSV incrementally.
#
# **Time model:** each ND2 *file* is treated as one time point (ordered by `TIME_ORDER`),
# and scenes are matched by index across files. If instead your time points live in the ND2
# `T` axis *within* one file, set the files list to that single file and the loop will use
# its `T` axis. Adjust `scene`/`time` iteration here if your layout differs.

# %%
@dataclass
class ResultAccumulator:
    """Collects compact result tables; can flush incrementally to CSV to bound memory."""
    spots: List[pd.DataFrame] = field(default_factory=list)
    beads: List[pd.DataFrame] = field(default_factory=list)
    transforms: List[dict] = field(default_factory=list)
    counts: List[dict] = field(default_factory=list)
    coloc_events: List[pd.DataFrame] = field(default_factory=list)
    time_aligned: List[pd.DataFrame] = field(default_factory=list)
    cumulative: List[dict] = field(default_factory=list)
    log: List[dict] = field(default_factory=list)

    def logmsg(self, level, file, scene, time, message):
        self.log.append({"level": level, "file": os.path.basename(str(file)),
                         "scene": scene, "time": time, "message": message})


def process_all(files: List[str],
                incremental_csv: bool = False,
                accumulator: Optional[ResultAccumulator] = None) -> ResultAccumulator:
    """Run the full pipeline over ``files`` (each file = one time point).

    Returns a ResultAccumulator holding every result table. When ``incremental_csv`` is
    True, per-image rows are also appended to CSV as they are produced (memory-safe for
    very large datasets).
    """
    acc = accumulator if accumulator is not None else ResultAccumulator()
    files = order_files_by_time(files, TIME_ORDER)
    if not files:
        acc.logmsg("WARN", "-", -1, -1, "No files to process.")
        return acc

    # Reference-time context per scene: bead & nucleosome tables + nucleosome image,
    # used to align later time points back to t0. Compact (coords + one small image).
    ref_context: Dict[int, dict] = {}
    cumulative_state: Dict[int, SceneCumulativeState] = {}
    qc_budget: Dict[str, int] = {}

    for time_idx, path in enumerate(files):
        fname = os.path.basename(path)
        qc_budget[fname] = MAX_QC_IMAGES_PER_FILE
        try:
            acc_file = ND2Accessor(path)
        except Exception as e:
            acc.logmsg("ERROR", path, -1, -1, f"Could not open file: {e}")
            continue

        with acc_file as fa:
            n_scenes = fa.n_scenes
            # If a file carries its own T axis and we only have one file, iterate T too.
            use_internal_time = (len(files) == 1 and fa.n_times > 1)
            n_internal_times = fa.n_times if use_internal_time else 1

            for scene in range(n_scenes):
                for it in range(n_internal_times):
                    t_here = it if use_internal_time else time_idx
                    _process_one_fov(fa, path, fname, scene, it if use_internal_time else 0,
                                     t_here, acc, ref_context, cumulative_state, qc_budget)
                    if MEMORY_SAFE_MODE:
                        gc.collect()

            if incremental_csv:
                _flush_incremental(acc)

    return acc


def _process_one_fov(fa, path, fname, scene, internal_time, t_here,
                     acc, ref_context, cumulative_state, qc_budget,
                     sample=None, store_ref_image=True):
    """Process a single field of view at one time point (helper for process_all).

    ``sample`` tags the output rows; ``store_ref_image=False`` keeps no reference image
    (relies on bead/nucleosome voting for time alignment) to bound memory in big runs.
    """
    read_t = internal_time  # index into the file's T axis (0 if file==timepoint)

    # --- 1. Extract channels (only the slices we need) --------------------------------
    green = extract_channel_image(fa, "nucleosome", scene, read_t)
    red = extract_channel_image(fa, "R_PTM", scene, read_t)
    blue = extract_channel_image(fa, "B_PTM", scene, read_t)
    bead_img = extract_channel_image(fa, "beads", scene, read_t)

    if green is None:
        acc.logmsg("WARN", path, scene, t_here,
                   "Reference/nucleosome channel missing; skipping FOV.")
        return

    # --- 2. Bead detection (dedicated channel, else per-channel) ----------------------
    if bead_img is not None:
        beads_ref = detect_beads(bead_img, fname, scene, t_here, "beads")
        beads_r = beads_b = beads_ref  # same fiducials seen in all channels
    else:
        beads_ref = detect_beads(green, fname, scene, t_here, "nucleosome")
        beads_r = detect_beads(red, fname, scene, t_here, "R_PTM")
        beads_b = detect_beads(blue, fname, scene, t_here, "B_PTM")
    for bdf in (beads_ref, beads_r, beads_b):
        if len(bdf):
            acc.beads.append(bdf)

    # Robust fiducial set: beads confirmed across >=2 channels (a channel whose detector has
    # flooded is ignored). This prevents bead-exclusion / time-alignment from being poisoned
    # when the green (nucleosome) channel's bead finder mistakes nucleosome spots for beads.
    if bead_img is not None:
        conf_beads_yx = beads_ref[["y", "x"]].to_numpy(float) if len(beads_ref) else np.empty((0, 2))
    else:
        conf_beads_yx = confirmed_bead_coords([beads_ref, beads_r, beads_b])
    conf_beads_df = _beads_df_from_coords(conf_beads_yx)
    if beads_are_flooded(beads_ref, beads_r, beads_b) or \
            (len(conf_beads_yx) == 0 and max(len(beads_ref), len(beads_r), len(beads_b)) > MAX_TRUSTED_BEADS):
        acc.logmsg("WARN", path, scene, t_here,
                   f"bead detector flooded (G={len(beads_ref)} R={len(beads_r)} B={len(beads_b)}); "
                   f"using {len(conf_beads_yx)} cross-channel-confirmed beads.")

    # --- 3. Channel registration (Red, Blue -> Green frame) ---------------------------
    tf_green = ChannelTransform(method="reference", success=True, note="reference channel")
    tf_red = estimate_channel_transform(beads_ref, beads_r, green, red, REGISTRATION_MODE) \
        if red is not None else ChannelTransform(method="missing_channel", note="no red channel")
    tf_blue = estimate_channel_transform(beads_ref, beads_b, green, blue, REGISTRATION_MODE) \
        if blue is not None else ChannelTransform(method="missing_channel", note="no blue channel")
    acc.transforms.append(transform_record(fname, scene, t_here, "R_PTM", tf_red))
    acc.transforms.append(transform_record(fname, scene, t_here, "B_PTM", tf_blue))

    # --- 4. Spot detection (registered coordinates in green frame) --------------------
    # bead_yx=conf_beads_yx masks bead footprints out of the background-blur input only
    # (see mask_beads_for_background), preventing the halo-ring artifact around bright beads.
    nuc_spots = detect_spots(green, fname, scene, t_here, "nucleosome", tf_green, bead_yx=conf_beads_yx)
    r_spots = detect_spots(red, fname, scene, t_here, "R_PTM", tf_red, bead_yx=conf_beads_yx)
    b_spots = detect_spots(blue, fname, scene, t_here, "B_PTM", tf_blue, bead_yx=conf_beads_yx)

    # Remove fiducial beads (bright in every channel) so they don't create false spots /
    # triple colocalizations. Use the cross-channel-CONFIRMED beads so a flooded detector
    # can never delete real spots.
    bead_yx = conf_beads_yx
    nuc_spots, rm_n = exclude_spots_near_beads(nuc_spots, bead_yx)
    r_spots, rm_r = exclude_spots_near_beads(r_spots, bead_yx)
    b_spots, rm_b = exclude_spots_near_beads(b_spots, bead_yx)
    if (rm_n + rm_r + rm_b) > 0:
        acc.logmsg("INFO", path, scene, t_here,
                   f"bead exclusion removed spots -> nuc:{rm_n} R:{rm_r} B:{rm_b}")

    for sdf in (nuc_spots, r_spots, b_spots):
        if len(sdf):
            acc.spots.append(sdf)

    # --- 5. Colocalization within this image ------------------------------------------
    events_df, counts = colocalize(nuc_spots, r_spots, b_spots, fname, scene, t_here)
    counts["sample"] = sample
    acc.counts.append(counts)
    if len(events_df):
        acc.coloc_events.append(events_df)

    # --- 6. Across-time alignment + cumulative new-only counting ----------------------
    if scene not in ref_context:
        # First time we see this scene -> it is the reference time point.
        ref_context[scene] = {
            "beads": conf_beads_df.copy(), "nuc": nuc_spots.copy(),
            # small normalised image kept only if requested (phase-corr last-resort)
            "nuc_img": _norm01(green) if store_ref_image else None,
        }
        cumulative_state[scene] = SceneCumulativeState()
        tf_time = ChannelTransform(method="reference", success=True, note="reference time point")
    else:
        ref = ref_context[scene]
        mov_img = _norm01(green) if store_ref_image else None
        tf_time = estimate_time_transform(ref["beads"], conf_beads_df, ref["nuc"], nuc_spots,
                                          ref["nuc_img"], mov_img, REGISTRATION_MODE)
        acc.logmsg("INFO", path, scene, t_here, f"time-align: {tf_time.note}")

    # Map current detections/events into the reference-time frame.
    nuc_ref_xy = apply_time_transform_to_spots(nuc_spots, tf_time)
    r_all_ref_xy = apply_time_transform_to_spots(r_spots, tf_time)   # all R PTM spots
    b_all_ref_xy = apply_time_transform_to_spots(b_spots, tf_time)   # all B PTM spots
    r_ref_xy = events_to_ref_xy(events_df, "R_nucleosome", tf_time)
    b_ref_xy = events_to_ref_xy(events_df, "B_nucleosome", tf_time)
    rb_ref_xy = rb_pairs_to_ref_xy(events_df, tf_time)  # R<->B pairs, anchored on midpoint
    triple_ref_xy = events_to_ref_xy(events_df, "R_B_nucleosome_triple", tf_time)

    # Record time-aligned nucleosome coordinates (before new/old split) for export.
    if len(nuc_ref_xy):
        acc.time_aligned.append(pd.DataFrame({
            "file": fname, "scene": scene, "time": t_here,
            "channel_type": "nucleosome",
            "ref_y": nuc_ref_xy[:, 0], "ref_x": nuc_ref_xy[:, 1],
        }))

    state = cumulative_state[scene]
    seen_before = state.nucleosomes.copy()  # for QC old/new split
    cum, cumulative_state[scene] = update_cumulative(
        state, nuc_ref_xy, r_all_ref_xy, b_all_ref_xy,
        r_ref_xy, b_ref_xy, rb_ref_xy, triple_ref_xy)
    cum_row = {"file": fname, "sample": sample, "scene": scene, "time": t_here, **cum}
    acc.cumulative.append(cum_row)

    # --- 7. QC figures (budget-limited) -----------------------------------------------
    if SAVE_QC_FIGURES and qc_budget.get(fname, 0) > 0:
        tag = f"{os.path.splitext(fname)[0]}_s{scene}_t{t_here}"
        red_reg = warp_to_reference(red, tf_red) if red is not None else None
        blue_reg = warp_to_reference(blue, tf_blue) if blue is not None else None
        qc_raw_channels(green, red, blue, f"Raw channels — {tag}", f"{tag}_raw.png")
        qc_beads(green if bead_img is None else bead_img, beads_ref,
                 f"Beads — {tag}", f"{tag}_beads.png")
        qc_registration(green, red, red_reg, blue, blue_reg,
                        f"Registration — {tag}", f"{tag}_registration.png")
        qc_spots(green, nuc_spots, "lime", f"Nucleosome spots — {tag}", f"{tag}_nuc_spots.png")
        qc_colocalization(green, nuc_spots, r_spots, b_spots, events_df,
                          f"Colocalization — {tag}", f"{tag}_coloc.png")
        if len(nuc_ref_xy):
            new_mask = _new_mask(nuc_ref_xy, seen_before, TIME_ALIGNMENT_RADIUS_PX)
            qc_time_alignment(seen_before, nuc_ref_xy[new_mask], nuc_ref_xy[~new_mask],
                              f"Across-time new vs seen — {tag}", f"{tag}_timealign.png")
        qc_budget[fname] -= 1

    # --- 8. Release raw arrays ---------------------------------------------------------
    del green, red, blue, bead_img
    if MEMORY_SAFE_MODE:
        gc.collect()


def _flush_incremental(acc: ResultAccumulator):
    """Append current per-image tables to CSV, then clear them to bound memory."""
    _append_csv(acc.counts, os.path.join(OUTPUT_DIR, "per_image_counts.csv"), COUNT_COLUMNS)
    _append_csv(acc.cumulative, os.path.join(OUTPUT_DIR, "cumulative_new_counts.csv"), CUM_COLUMNS)
    _append_csv(acc.transforms, os.path.join(OUTPUT_DIR, "registration_transforms.csv"), TRANSFORM_COLUMNS)
    _append_frames(acc.spots, os.path.join(OUTPUT_DIR, "spot_detections.csv"), SPOT_COLUMNS)
    _append_frames(acc.beads, os.path.join(OUTPUT_DIR, "bead_detections.csv"), BEAD_COLUMNS)
    _append_frames(acc.coloc_events, os.path.join(OUTPUT_DIR, "colocalization_events.csv"), COLOC_COLUMNS)
    acc.counts.clear(); acc.cumulative.clear(); acc.transforms.clear()
    acc.spots.clear(); acc.beads.clear(); acc.coloc_events.clear()

# %% [markdown]
# ### Processing one whole sample across all timepoints and FOVs
#
# `process_sample` runs every timepoint × field of view for a single sample, in time order,
# with **cumulative new-only counting so nothing is double-counted between timepoints**. It
# is built for speed and bounded memory:
#
# - each ND2 file is opened **once per timepoint** and its FOVs are read sequentially;
# - the loop is **timepoints (outer) × FOVs (inner)** with a **nested `tqdm` progress bar**;
# - the first timepoint seeds the reference for every FOV; later timepoints are aligned back
#   to it with the robust **coarse-vote + fine-match** step, then only spots/events farther
#   than `TIME_ALIGNMENT_RADIUS_PX` from what was already counted are added;
# - **no raw images are retained across timepoints** — only compact reference bead/nucleosome
#   coordinates and the cumulative event coordinates per FOV;
# - QC figures are **off by default** here (they dominate runtime); pass `save_qc=True` to
#   emit a capped few.
#
# `group_files_by_sample` parses `TS<timepoint>_<sample>.nd2` and returns each sample's files
# already ordered by timepoint.
#
# **`process_sample_parallel`** does the same work but distributes the FOVs across CPU cores
# with joblib (`N_JOBS`). Because a whole FOV time-series stays inside one worker, the
# cumulative new-only counting is identical to the serial result — only faster. It falls back
# to serial automatically if joblib is unavailable or a worker errors.

# %%
import re as _re


def group_files_by_sample(files: List[str],
                          pattern: str = r"TS(\d+)_(\d+)\.nd2$") -> Dict[object, List[str]]:
    """Group files into {sample_id: [paths ordered by timepoint]}.

    ``pattern`` group(1) = timepoint, group(2) = sample id. Non-matching files land under
    key ``None`` in sorted order, so nothing is dropped.
    """
    rx = _re.compile(pattern)
    tmp: Dict[object, List[Tuple[object, str]]] = {}
    for f in sorted(files):
        m = rx.search(os.path.basename(f))
        tp, sample = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        tmp.setdefault(sample, []).append((tp, f))
    out = {}
    for s, lst in tmp.items():
        lst.sort(key=lambda x: (x[0] is None, x[0]))  # order by timepoint, None last
        out[s] = [f for _, f in lst]
    return out


# Per-process ND2 handle cache (only used when USE_ND2_HANDLE_CACHE is True). Each worker
# process keeps its own dict, so files are opened at most once per worker instead of once per
# FOV. Bit-identical results.
_ND2_CACHE: Dict[str, "ND2Accessor"] = {}


def _open_accessor(path: str) -> "ND2Accessor":
    """Return an ND2Accessor for ``path`` — cached (kept open) if USE_ND2_HANDLE_CACHE, else a
    fresh handle the caller is responsible for closing."""
    if not USE_ND2_HANDLE_CACHE:
        return ND2Accessor(path)
    acc = _ND2_CACHE.get(path)
    if acc is None:
        acc = ND2Accessor(path)
        _ND2_CACHE[path] = acc
    return acc


import atexit


@atexit.register
def _close_nd2_cache():
    """Close any cached ND2 handles at process exit (incl. parallel workers) so no file is
    left open — avoids a GC warning when USE_ND2_HANDLE_CACHE is on."""
    for _a in list(_ND2_CACHE.values()):
        try:
            _a.close()
        except Exception:
            pass
    _ND2_CACHE.clear()


def enable_fast_mode(background: str = "auto", inner_threads: int = 1,
                     handle_cache: bool = True) -> None:
    """Turn on the optional accelerators in one call (they are OFF by default).

    Sets BACKGROUND_BACKEND (default "auto" = cv2 if available else downsample — a small,
    validate-first change to spot counts), PARALLEL_INNER_THREADS, and USE_ND2_HANDLE_CACHE.
    Call before process_sample/run_samples.
    """
    global BACKGROUND_BACKEND, PARALLEL_INNER_THREADS, USE_ND2_HANDLE_CACHE
    BACKGROUND_BACKEND = background
    PARALLEL_INNER_THREADS = inner_threads
    USE_ND2_HANDLE_CACHE = handle_cache
    print(f"Fast mode ON: BACKGROUND_BACKEND={background!r} "
          f"(cv2 available: {_HAVE_CV2}), PARALLEL_INNER_THREADS={inner_threads}, "
          f"USE_ND2_HANDLE_CACHE={handle_cache}. NOTE: non-scipy background changes spot "
          f"counts slightly — validate against a scipy run.")


def process_sample(files_ordered: List[str], sample_id=None, scenes=None,
                   save_qc: bool = False, export_dir: Optional[str] = None,
                   accumulator: Optional[ResultAccumulator] = None,
                   show_progress: bool = True) -> ResultAccumulator:
    """Process all timepoints x FOVs for ONE sample with cumulative new-only counting.

    Parameters
    ----------
    files_ordered : ND2 paths for this sample, already ordered by timepoint (see
        ``group_files_by_sample``). Each file = one timepoint.
    sample_id : tag written to the ``sample`` column of the outputs.
    scenes : optional iterable of FOV indices to restrict to (e.g. ``range(3)`` for a quick
        test); ``None`` processes every FOV in the file.
    save_qc : emit a capped number of QC figures per timepoint (slow); default False.
    export_dir : if given, write all CSVs there at the end.

    Returns the ResultAccumulator. Nested tqdm: outer=timepoints, inner=FOVs.
    """
    # When inner threads are capped (parallel runs), also cap OpenCV's own thread pool so it
    # doesn't oversubscribe against the worker processes. In serial/idle-core runs this is
    # None, leaving cv2 free to use all cores (where it actually speeds things up).
    if _HAVE_CV2 and PARALLEL_INNER_THREADS is not None:
        try:
            cv2.setNumThreads(int(PARALLEL_INNER_THREADS))
        except Exception:
            pass
    acc = accumulator if accumulator is not None else ResultAccumulator()
    files_ordered = list(files_ordered)
    if not files_ordered:
        acc.logmsg("WARN", "-", -1, -1, f"process_sample({sample_id}): no files")
        return acc

    # Reference context + cumulative state persist across the outer (timepoint) loop, so
    # each FOV accumulates across time without double counting.
    ref_context: Dict[int, dict] = {}
    cumulative_state: Dict[int, SceneCumulativeState] = {}

    def _bar(iterable, **kw):
        return tqdm(iterable, **kw) if show_progress else iterable

    outer = _bar(list(enumerate(files_ordered)),
                 desc=f"sample {sample_id}: timepoints", position=0)
    for ti, path in outer:
        fname = os.path.basename(path)
        qc_budget = {fname: (MAX_QC_IMAGES_PER_FILE if save_qc else 0)}
        try:
            fa = _open_accessor(path)   # cached (kept open) or fresh, per USE_ND2_HANDLE_CACHE
        except Exception as e:
            acc.logmsg("ERROR", path, -1, ti, f"open failed: {e}")
            continue
        try:
            scene_list = list(range(fa.n_scenes)) if scenes is None else list(scenes)
            inner = _bar(scene_list, desc=f"  t{ti} {fname} FOVs",
                         position=1, leave=False)
            for scene in inner:
                # store_ref_image=False -> no images kept across timepoints (voting aligns)
                _process_one_fov(fa, path, fname, scene, 0, ti, acc,
                                 ref_context, cumulative_state, qc_budget,
                                 sample=sample_id, store_ref_image=False)
        finally:
            if not USE_ND2_HANDLE_CACHE:
                fa.close()

    if export_dir:
        export_results(acc, output_dir=export_dir)
        if sample_id is not None:
            write_sample_summary(acc, sample_id, export_dir)
    return acc


def _merge_accumulators(target: ResultAccumulator,
                        parts: List[ResultAccumulator]) -> ResultAccumulator:
    """Concatenate the row/frame lists of several accumulators into ``target``."""
    for p in parts:
        if p is None:
            continue
        target.spots += p.spots
        target.beads += p.beads
        target.transforms += p.transforms
        target.counts += p.counts
        target.coloc_events += p.coloc_events
        target.time_aligned += p.time_aligned
        target.cumulative += p.cumulative
        target.log += p.log
    return target


def process_sample_parallel(files_ordered: List[str], sample_id=None, n_jobs: int = None,
                            scenes=None, save_qc: bool = False,
                            export_dir: Optional[str] = None) -> ResultAccumulator:
    """Parallel version of :func:`process_sample` — distributes FOVs across processes.

    Each worker runs :func:`process_sample` on a **disjoint subset of FOVs across all
    timepoints**, so every FOV's whole time-series (and its cumulative state) stays inside a
    single worker: the parallelism does not affect the new-only counting. Results are merged
    and exported in the parent. Uses joblib's ``loky`` backend (cloudpickle handles the
    notebook-defined functions). Falls back to serial ``process_sample`` if joblib is
    unavailable, ``n_jobs == 1``, or a worker error occurs.

    ``n_jobs``: default ``N_JOBS`` (-1 all cores, -2 all but one, 1 serial).
    """
    files_ordered = list(files_ordered)
    if not files_ordered:
        acc = ResultAccumulator()
        acc.logmsg("WARN", "-", -1, -1, f"process_sample_parallel({sample_id}): no files")
        return acc

    # Resolve the FOV list once (open a single file for its scene count).
    if scenes is None:
        with ND2Accessor(files_ordered[0]) as fa:
            scenes = list(range(fa.n_scenes))
    else:
        scenes = list(scenes)

    n_jobs = N_JOBS if n_jobs is None else n_jobs
    if n_jobs is not None and n_jobs < 0:
        n_jobs = max(1, (os.cpu_count() or 1) + 1 + n_jobs)  # -1 -> all, -2 -> all but one
    n_jobs = max(1, min(int(n_jobs or 1), len(scenes)))

    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            print(f"Parallel: {len(scenes)} FOVs on {n_jobs} workers ...")
            # One task per FOV (each FOV's whole time-series stays in one worker) so the
            # tqdm bar advances one tick per completed FOV.
            def _tasks():
                for sc in scenes:
                    yield delayed(process_sample)(files_ordered, sample_id, [sc],
                                                  save_qc, None, None, False)
            # Optional inner-thread cap to avoid BLAS/threadpool oversubscription across the
            # worker processes (None -> leave as-is; bit-identical results either way).
            pkw = {"n_jobs": n_jobs, "backend": "loky"}
            if PARALLEL_INNER_THREADS is not None:
                pkw["inner_max_num_threads"] = int(PARALLEL_INNER_THREADS)
            try:
                # joblib >= 1.3: stream results as they finish so tqdm shows live progress.
                gen = Parallel(return_as="generator", **pkw)(_tasks())
                parts = list(tqdm(gen, total=len(scenes),
                                  desc=f"sample {sample_id}: FOVs (x{n_jobs} workers)"))
            except TypeError:
                # Older joblib without return_as: no live bar, fall back to verbose log.
                parts = Parallel(verbose=5, **pkw)(_tasks())
            acc = _merge_accumulators(ResultAccumulator(), parts)
        except Exception as e:  # robust fallback
            warnings.warn(f"Parallel run failed ({e}); falling back to serial.")
            acc = process_sample(files_ordered, sample_id, scenes, save_qc)
    else:
        acc = process_sample(files_ordered, sample_id, scenes, save_qc)

    # Tidy: sort per-image tables by (scene, time) for readable, deterministic output.
    acc.counts.sort(key=lambda r: (r.get("scene", 0), r.get("time", 0)))
    acc.cumulative.sort(key=lambda r: (r.get("scene", 0), r.get("time", 0)))
    if export_dir:
        export_results(acc, output_dir=export_dir)
        if sample_id is not None:
            write_sample_summary(acc, sample_id, export_dir)
    return acc

# %% [markdown]
# ## 14. Results export to CSV
#
# `export_results` writes every table once at the end. `_append_csv` / `_append_frames`
# support incremental writing during the batch loop (header written only once).

# %%
def _append_csv(rows: List[dict], path: str, columns: List[str]):
    """Append a list of dict-rows to ``path`` (write header only if the file is new)."""
    if not rows:
        return
    df = pd.DataFrame(rows).reindex(columns=columns)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def _append_frames(frames: List[pd.DataFrame], path: str, columns: List[str]):
    """Append a list of DataFrames to ``path`` (write header only if the file is new)."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True).reindex(columns=columns)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def _concat(frames, columns):
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def export_results(acc: ResultAccumulator, output_dir: str = None) -> Dict[str, str]:
    """Write all accumulated tables to CSV and return a {name: path} map.

    Safe to call after an incremental run: the in-memory lists will simply be whatever has
    not yet been flushed. Empty tables are written with their expected headers.
    """
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    tables = {
        "spot_detections.csv": _concat(acc.spots, SPOT_COLUMNS),
        "bead_detections.csv": _concat(acc.beads, BEAD_COLUMNS),
        "registration_transforms.csv": pd.DataFrame(acc.transforms, columns=TRANSFORM_COLUMNS),
        "per_image_counts.csv": pd.DataFrame(acc.counts, columns=COUNT_COLUMNS),
        "colocalization_events.csv": _concat(acc.coloc_events, COLOC_COLUMNS),
        "time_aligned_spots.csv": _concat(acc.time_aligned,
                                          ["file", "scene", "time", "channel_type", "ref_y", "ref_x"]),
        "cumulative_new_counts.csv": pd.DataFrame(acc.cumulative, columns=CUM_COLUMNS),
        "processing_log.csv": pd.DataFrame(acc.log, columns=["level", "file", "scene", "time", "message"]),
    }
    paths = {}
    for name, df in tables.items():
        p = os.path.join(output_dir, name)
        # If incremental writing already created the file, append the remainder.
        if os.path.exists(p) and name not in ("time_aligned_spots.csv", "processing_log.csv"):
            if len(df):
                df.to_csv(p, mode="a", header=False, index=False)
        else:
            df.to_csv(p, index=False)
        paths[name] = p
    print("Wrote CSVs to", os.path.abspath(output_dir))
    for name in tables:
        print("  -", name)
    return paths


SAMPLE_SUMMARY_COLUMNS = [
    "sample", "n_FOVs", "n_timepoints",
    "cumulative_n_nucleosomes", "cumulative_n_R_PTMs", "cumulative_n_B_PTMs",
    "cumulative_n_R_colocalized_with_nucleosome",
    "cumulative_n_B_colocalized_with_nucleosome",
    "cumulative_n_R_B_colocalized",
    "cumulative_n_triple_colocalized",
]

_CUM_TYPE_COLS = [
    "cumulative_n_nucleosomes", "cumulative_n_R_PTMs", "cumulative_n_B_PTMs",
    "cumulative_n_R_colocalized_with_nucleosome",
    "cumulative_n_B_colocalized_with_nucleosome",
    "cumulative_n_R_B_colocalized",
    "cumulative_n_triple_colocalized",
]


def sample_summary_frame(acc: ResultAccumulator, sample_id) -> pd.DataFrame:
    """Build the one-row summary for a sample: final cumulative count of each type across
    all FOVs.

    For every detected type — nucleosomes, R PTMs, B PTMs, and the R-nucleosome /
    B-nucleosome / triple colocalizations — reports the de-duplicated total unique
    detections over all timepoints, summed across all FOVs. (Each FOV's cumulative count is
    monotonic in time, so its final value is its per-FOV maximum; those are summed.)
    """
    cum = pd.DataFrame(acc.cumulative, columns=CUM_COLUMNS)
    if len(cum) == 0:
        row = {c: 0 for c in SAMPLE_SUMMARY_COLUMNS}
        row["sample"] = sample_id
    else:
        per_fov_final = cum.groupby("scene")[_CUM_TYPE_COLS].max()  # final per FOV
        row = {"sample": sample_id,
               "n_FOVs": int(cum["scene"].nunique()),
               "n_timepoints": int(cum["time"].nunique())}
        for c in _CUM_TYPE_COLS:
            row[c] = int(per_fov_final[c].sum())                    # sum over FOVs
    return pd.DataFrame([row], columns=SAMPLE_SUMMARY_COLUMNS)


def write_sample_summary(acc: ResultAccumulator, sample_id, output_dir: str) -> str:
    """Write :func:`sample_summary_frame` to ``sample_<id>_summary.csv``; return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"sample_{sample_id}_summary.csv")
    sample_summary_frame(acc, sample_id).to_csv(path, index=False)
    print("Wrote sample summary ->", path)
    return path

# %% [markdown]
# ### Saving & loading tuned parameters
#
# The QA GUI (`epinuc_gui.py`) lets you tune detection thresholds visually, then **Save config**
# to a JSON file. `run_samples(..., config_path="epinuc_config.json")` (or `load_config(...)`)
# re-applies those exact values, so the production run matches what you validated in the GUI.

# %%
# The tunable knobs a user adjusts during QA (dicts are copied so callers can't mutate globals).
CONFIG_KEYS = ["SPOT_DETECTION_SNR", "SPOT_DETECTION_THRESHOLD", "SPOT_DETECTION_SIGMA",
               "COLOCALIZATION_RADIUS_PX", "TIME_ALIGNMENT_RADIUS_PX",
               "BEAD_EXCLUSION_RADIUS_PX", "EXCLUDE_BEADS_FROM_SPOTS", "MAX_TRUSTED_BEADS",
               "CONFIRM_BEADS_CROSS_CHANNEL", "BEAD_DETECTION_SIGMA", "BEAD_DETECTION_THRESHOLD",
               "BEAD_DETECTION_METHOD", "CHANNEL_MAP", "PIXEL_SIZE_UM",
               "REGISTRATION_MODE", "Z_PROJECTION_METHOD", "Z_PLANE_INDEX",
               "MASK_BEADS_BEFORE_BACKGROUND", "BEAD_BACKGROUND_MASK_RADIUS_PX"]


def get_config() -> dict:
    """Return a JSON-serialisable dict of the current tunable parameters (deep-copied)."""
    import copy
    return {k: copy.deepcopy(globals()[k]) for k in CONFIG_KEYS if k in globals()}


def apply_config(cfg: dict) -> None:
    """Set the tunable module globals from a config dict (unknown keys are ignored)."""
    for k, v in cfg.items():
        if k in CONFIG_KEYS:
            globals()[k] = v


def save_config(path: str = "epinuc_config.json") -> str:
    """Write the current tunable parameters to a JSON file; return the path."""
    import json
    with open(path, "w") as f:
        json.dump(get_config(), f, indent=2)
    print("Wrote config ->", os.path.abspath(path))
    return path


def load_config(path: str = "epinuc_config.json") -> dict:
    """Load a JSON config and apply it to the module globals; return the dict."""
    import json
    with open(path) as f:
        cfg = json.load(f)
    apply_config(cfg)
    print("Applied config from", os.path.abspath(path))
    return cfg

# %% [markdown]
# ## 15. One-call runner: `run_samples`
#
# `run_samples` is the top-level entry point (also exported by the companion
# `epinuc_colocalization.py` module). Give it a sample id or a list of ids; it processes each
# sample end to end (parallel across FOVs, cumulative new-only counting) and writes **one
# combined CSV with a single row per sample** — the final cumulative count of each type across
# all FOVs. Returns that table as a DataFrame.

# %%
def run_samples(samples, data_dir=None, output_dir=None, channel_map=None,
                n_jobs=None, scenes=None, save_qc=False, write_details=False,
                summary_path=None, pattern=r"TS(\d+)_(\d+)\.nd2$",
                config_path=None) -> pd.DataFrame:
    """Run one or more samples and write a combined summary CSV (one row per sample).

    Parameters
    ----------
    samples : int or iterable of ints — sample id(s) to process.
    data_dir : folder of ND2 files (default: module ``DATA_DIR``).
    output_dir : output folder (default: module ``OUTPUT_DIR``).
    channel_map : optional dict overriding ``CHANNEL_MAP`` for this run.
    n_jobs : parallel workers (default ``N_JOBS``; 1 = serial).
    scenes : optional subset of FOV indices (e.g. ``range(3)``); ``None`` = all FOVs.
    save_qc : also save QC figures (slow).
    write_details : also write the full per-sample detail CSVs (spots, beads, events, ...).
    summary_path : path for the combined CSV (default ``output_dir/samples_summary.csv``).
    pattern : filename regex, group(1)=timepoint, group(2)=sample.
    config_path : optional path to a JSON config (from the GUI / ``save_config``) whose tuned
        thresholds are applied before the run.

    Returns the combined one-row-per-sample summary DataFrame.
    """
    global CHANNEL_MAP
    if config_path is not None:
        load_config(config_path)
    if isinstance(samples, (int, np.integer)):
        samples = [int(samples)]
    samples = [int(s) for s in samples]
    data_dir = data_dir or DATA_DIR
    output_dir = output_dir or OUTPUT_DIR
    if channel_map is not None:
        CHANNEL_MAP = channel_map
    os.makedirs(output_dir, exist_ok=True)

    files = list_nd2_files(data_dir)
    by_sample = group_files_by_sample(files, pattern)
    available = sorted(k for k in by_sample if k is not None)

    rows = []
    for sid in samples:
        if sid not in by_sample:
            warnings.warn(f"Sample {sid} not found in {data_dir} (available: {available}).")
            continue
        print(f"\n=== Sample {sid}: {len(by_sample[sid])} timepoint(s) ===")
        acc = process_sample_parallel(by_sample[sid], sample_id=sid, n_jobs=n_jobs,
                                      scenes=scenes, save_qc=save_qc, export_dir=None)
        if write_details:
            sdir = os.path.join(output_dir, f"sample_{sid}")
            export_results(acc, output_dir=sdir)
            write_sample_summary(acc, sid, sdir)
        rows.append(sample_summary_frame(acc, sid))

    combined = (pd.concat(rows, ignore_index=True) if rows
                else pd.DataFrame(columns=SAMPLE_SUMMARY_COLUMNS))
    summary_path = summary_path or os.path.join(output_dir, "samples_summary.csv")
    combined.to_csv(summary_path, index=False)
    print(f"\nWrote combined summary ({len(combined)} sample(s)) -> {summary_path}")
    return combined

# %% [markdown]
# ## 16. Example run
#
# **Step A — inspect one file** and confirm the channel map before committing to a full run.

# %%
files = list_nd2_files(DATA_DIR) if os.path.isdir(DATA_DIR) else []
print(f"Found {len(files)} ND2 file(s).")
if files:
    _ = describe_nd2(files[0])
    display(check_channel_map(files[0], CHANNEL_MAP))
else:
    print("No files yet. Set DATA_DIR, or run the synthetic self-test cell below.")

# %% [markdown]
# **Step B — run the full pipeline** and export CSVs. Set `incremental_csv=True` for very
# large datasets so per-image rows are flushed as they are produced.

# %%
if files:
    acc = process_all(files, incremental_csv=False)
    paths = export_results(acc)
    print("\nPer-image counts (head):")
    display(pd.DataFrame(acc.counts, columns=COUNT_COLUMNS).head())
    print("Cumulative new-only counts (head):")
    display(pd.DataFrame(acc.cumulative, columns=CUM_COLUMNS).head())
else:
    print("Skipping full run — no ND2 files found. Try the synthetic self-test below.")

# %% [markdown]
# **Step C — process one whole sample across all timepoints & FOVs (with multiprocessing).**
#
# This groups the files by sample, then runs one sample end to end with cumulative new-only
# counting. It defaults to a **3-FOV subset** so running all cells stays quick; set
# `test_scenes = None` to process every FOV. `process_sample_parallel` spreads FOVs across
# cores (`N_JOBS`); the result is identical to the serial `process_sample`.

# %%
if files:
    by_sample = group_files_by_sample(files)
    print("Samples found (id -> n timepoints):",
          {s: len(v) for s, v in sorted(by_sample.items())})

    test_sample = sorted(by_sample)[0]          # TODO: choose a sample id
    test_scenes = range(3)                       # TODO: set to None to run all FOVs
    sample_files = by_sample[test_sample]
    print(f"Testing sample {test_sample}: "
          f"{[os.path.basename(f) for f in sample_files]}")

    sample_out = os.path.join(OUTPUT_DIR, f"sample_{test_sample}")
    sample_acc = process_sample_parallel(
        sample_files, sample_id=test_sample, n_jobs=N_JOBS,
        scenes=test_scenes, save_qc=False, export_dir=sample_out)

    # process_sample_parallel already wrote sample_<id>_summary.csv: one row with the final
    # cumulative (de-duplicated across timepoints) count of each type, summed over all FOVs.
    summary_path = os.path.join(sample_out, f"sample_{test_sample}_summary.csv")
    print(f"\nSample {test_sample} summary  ({summary_path}):")
    display(pd.read_csv(summary_path))
    print(f"Detailed CSVs for this sample written to: {sample_out}")
else:
    print("No files — set DATA_DIR to run a sample.")

# %% [markdown]
# ### Synthetic self-test (no ND2 files required)
#
# This cell fabricates two "time points" of a single field of view with beads, nucleosomes
# and PTMs — including a small drift between rounds and a few new nucleosomes in round 2 — so
# you can verify the whole pipeline (registration, colocalization, cumulative new-only
# counting) end to end without any data. It exercises the core functions directly rather than
# the ND2 reader.

# %%
def _synthetic_selftest():
    """Exercise detection -> registration -> coloc -> cumulative counting on fake data."""
    rng = np.random.default_rng(1)
    H = W = 256

    def make_image(points, amp=800, sigma=1.3, bg=100, noise=8):
        img = np.full((H, W), float(bg))
        yy, xx = np.mgrid[0:H, 0:W]
        for (y, x) in points:
            img += amp * np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2)))
        img += rng.normal(0, noise, img.shape)
        return np.clip(img, 0, None).astype(np.float32)

    beads = rng.uniform(20, 236, size=(8, 2))
    nuc_pts = rng.uniform(20, 236, size=(40, 2))
    r_pts = nuc_pts[:15] + rng.normal(0, 0.6, (15, 2))   # 15 R colocalize
    b_pts = nuc_pts[:10] + rng.normal(0, 0.6, (10, 2))   # 10 B colocalize (first 10 -> triples)

    acc = ResultAccumulator()
    ref_context, cum_state, qc_budget = {}, {}, {}

    for t in range(2):
        drift = np.array([0.0, 0.0]) if t == 0 else np.array([2.5, -1.5])  # (y, x) stage drift
        chan_shift_r = np.array([0.8, -0.5])   # red channel chromatic offset
        chan_shift_b = np.array([-0.6, 0.9])   # blue channel chromatic offset
        extra = rng.uniform(20, 236, size=(6, 2)) if t == 1 else np.empty((0, 2))  # 6 new nucs
        nuc_all = np.vstack([nuc_pts, extra]) if len(extra) else nuc_pts

        green = make_image(nuc_all + drift)
        red = make_image(r_pts + drift + chan_shift_r)
        blue = make_image(b_pts + drift + chan_shift_b)
        bead_img_g = make_image(beads + drift, amp=1500, sigma=2.0)
        bead_img_r = make_image(beads + drift + chan_shift_r, amp=1500, sigma=2.0)
        bead_img_b = make_image(beads + drift + chan_shift_b, amp=1500, sigma=2.0)

        fname = f"synthetic_round{t}.nd2"
        beads_ref = detect_beads(bead_img_g, fname, 0, t, "nucleosome")
        beads_r = detect_beads(bead_img_r, fname, 0, t, "R_PTM")
        beads_b = detect_beads(bead_img_b, fname, 0, t, "B_PTM")

        tf_g = ChannelTransform(method="reference", success=True)
        tf_r = estimate_channel_transform(beads_ref, beads_r, green, red, REGISTRATION_MODE)
        tf_b = estimate_channel_transform(beads_ref, beads_b, green, blue, REGISTRATION_MODE)

        bead_yx = confirmed_bead_coords([beads_ref, beads_r, beads_b])
        nuc_s = detect_spots(green, fname, 0, t, "nucleosome", tf_g, bead_yx=bead_yx)
        r_s = detect_spots(red, fname, 0, t, "R_PTM", tf_r, bead_yx=bead_yx)
        b_s = detect_spots(blue, fname, 0, t, "B_PTM", tf_b, bead_yx=bead_yx)
        nuc_s, _ = exclude_spots_near_beads(nuc_s, bead_yx)
        r_s, _ = exclude_spots_near_beads(r_s, bead_yx)
        b_s, _ = exclude_spots_near_beads(b_s, bead_yx)
        events, counts = colocalize(nuc_s, r_s, b_s, fname, 0, t)
        acc.counts.append(counts)

        if 0 not in ref_context:
            ref_context[0] = {"beads": beads_ref.copy(), "nuc": nuc_s.copy(), "nuc_img": _norm01(green)}
            cum_state[0] = SceneCumulativeState()
            tf_time = ChannelTransform(method="reference", success=True)
        else:
            tf_time = estimate_time_transform(ref_context[0]["beads"], beads_ref,
                                              ref_context[0]["nuc"], nuc_s,
                                              ref_context[0]["nuc_img"], _norm01(green),
                                              REGISTRATION_MODE)

        nuc_ref = apply_time_transform_to_spots(nuc_s, tf_time)
        r_all_ref = apply_time_transform_to_spots(r_s, tf_time)
        b_all_ref = apply_time_transform_to_spots(b_s, tf_time)
        r_ref = events_to_ref_xy(events, "R_nucleosome", tf_time)
        b_ref = events_to_ref_xy(events, "B_nucleosome", tf_time)
        rb_ref = rb_pairs_to_ref_xy(events, tf_time)
        trip_ref = events_to_ref_xy(events, "R_B_nucleosome_triple", tf_time)
        cum, cum_state[0] = update_cumulative(cum_state[0], nuc_ref, r_all_ref, b_all_ref,
                                              r_ref, b_ref, rb_ref, trip_ref)
        acc.cumulative.append({"file": fname, "scene": 0, "time": t, **cum})
        print(f"t={t}: raw_nuc={counts['n_nucleosomes']:3d}  "
              f"R_coloc={counts['n_R_colocalized_with_nucleosome']:3d}  "
              f"RB_coloc={counts['n_R_B_colocalized']:3d}  "
              f"triple={counts['n_R_B_nucleosome_triple_colocalized']:3d}  "
              f"| new_nuc={cum['new_n_nucleosomes']:3d}  "
              f"cumulative_nuc={cum['cumulative_n_nucleosomes']:3d}  "
              f"time_align={tf_time.note}")
    print("\nSelf-test OK: round 1 should add ~6 new nucleosomes over round 0's ~40.")
    return acc

_ = _synthetic_selftest()

# %% [markdown]
# ## 16. Troubleshooting & parameter tuning notes
#
# **Nothing/too much detected (beads or spots)**
# - Look at the `*_beads.png` / `*_spots.png` QC overlays first.
# - Lower `BEAD_DETECTION_THRESHOLD` / `SPOT_DETECTION_THRESHOLD[...]` to detect more; raise
#   to detect fewer. `None` auto-estimates (Otsu / 99th percentile) — good starting point.
# - Tune `SPOT_DETECTION_SIGMA[...]` to your PSF (≈ spot radius in pixels). Too large merges
#   neighbours; too small fragments single spots.
#
# **Registration looks wrong (channels don't overlap after)**
# - Check the `*_registration.png` before/after overlay. If "after" is worse, you likely
#   matched too few / spurious beads — raise `BEAD_DETECTION_THRESHOLD` for cleaner beads, or
#   lower `BEAD_MATCH_MAX_DIST_PX` if false pairs form.
# - `REGISTRATION_MODE="affine"` needs ≥3 well-spread matched beads; otherwise keep
#   `"translation"`. The `note`/`method` columns in `registration_transforms.csv` tell you
#   which path (beads / phase_correlation / identity) was used per channel.
#
# **Colocalization counts feel off**
# - `COLOCALIZATION_RADIUS_PX` should match localization precision + residual registration
#   error. If `PIXEL_SIZE_UM` is known, choose it from a physical distance (e.g. 250 nm /
#   0.108 µm ≈ 2.3 px). Inspect `distance_*` columns in `colocalization_events.csv` — the
#   bulk of true events should sit well inside the radius.
#
# **Across-time / cumulative counts look wrong**
# - Open `*_timealign.png`: magenta = counted-as-new, orange = matched to a previous round.
#   If genuinely-same spots show as new, the field of view isn't aligning — check
#   `processing_log.csv` for `time-align: ... identity` (alignment failed) and ensure beads
#   or enough nucleosomes exist. Increase `TIME_ALIGNMENT_RADIUS_PX` slightly to absorb
#   residual drift, but not so much that distinct spots merge.
# - This step assumes scenes correspond by index across files. If your acquisition reorders
#   positions, add a scene-matching step (e.g. by stage coordinates from metadata).
#
# **Time model**
# - Default: **one ND2 file per time point**, scenes matched by index. If your time points
#   are the ND2 `T` axis inside a single file, pass just that file — the loop then iterates
#   its internal `T`. Mixed layouts need a small edit in `process_all`.
#
# **Memory**
# - Keep `MEMORY_SAFE_MODE=True`. For very large datasets set `incremental_csv=True` in
#   `process_all` and reduce `MAX_QC_IMAGES_PER_FILE`. Only coordinate tables + one small
#   normalised nucleosome image per reference scene are retained.
#
# **Reader issues**
# - If `nd2` struggles with an odd file, alternatives: `aicsimageio` (`AICSImage`),
#   `nd2reader`. Swap the body of `ND2Accessor` accordingly — the rest of the pipeline only
#   needs `sizes`, `channel_names`, `get_plane`, `get_zstack`.
#
# **Validation**
# - Always validate detection + colocalization against manual annotations on a few images
#   before trusting batch numbers.
