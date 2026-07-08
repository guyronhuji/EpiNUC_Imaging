"""TIF-collection loader for the EPINUC colocalization pipeline.

The production pipeline in ``epinuc_colocalization.py`` was written for ND2 files that carry
all fields of view (P), time points (T) and channels (C) in one container. The EpiVision
imaging instrument (runs NUC044 onwards) instead writes a *flat directory of single-plane
2048x2048 uint16 TIFs*, one file per (flowcell-lane, position, dye, cycle, retry-attempt),
e.g.::

    ch1_pos051_img0_Z1_R_cycle001_C001_000150_000.tif
     |    |     |    |  |    |      |     |      +-- DataIndex   (retry/attempt in a slot)
     |    |     |    |  |    |      |     +--------- FrameNumber (global acquisition seq)
     |    |     |    |  |    |      +--------------- SampleId    C001 / C002
     |    |     |    |  |    +---------------------- ProbeCycle  001..005
     |    |     |    |  +--------------------------- LaserId     R=red  G=green  B=blue
     |    |     |    +------------------------------ Base/dye    N, Z1..Z6
     |    |     +----------------------------------- img order index (redundant)
     |    +----------------------------------------- PositionId (FOV)
     +---------------------------------------------- Channel (flowcell lane 1..6)

Per the EpiVision "PTM Image Analysis Requirements" layout, one flowcell lane ``chN`` is one
EPINUC channel: a nucleosome template (``N``, green, imaged at cycle 1 only) plus a red/blue
antibody pair (``Z1..Z6``, imaged over 5 cycles). That maps onto the pipeline's three roles:

    nucleosome <- laser G (base N)     R_PTM <- laser R (Z1/Z3/Z5)     B_PTM <- laser B (Z2/Z4/Z6)

That laser->role assignment is only the *default* (:data:`DEFAULT_TIF_CHANNEL_MAP`). It is a
property of how the run was acquired, not of the file format, so it is user-selectable -- the
TIF analogue of ``ep.CHANNEL_MAP`` for ND2s. Pass ``tif_channel_map`` to
:func:`configure_pipeline`, pick it in ``epinuc_tiff_gui.py``, or override it on the command
line with ``epinuc_tiff_cli.py --channel-map``. It round-trips through ``epinuc_config.json``
under the ``TIF_CHANNEL_MAP`` key (``CHANNEL_MAP`` there stays reserved for the ND2 path).

This module exposes a :class:`TiffCycleAccessor` that mimics ``ND2Accessor`` for **one lane
at one cycle** (P=positions, T=1, C=3 roles), plus :func:`run_channel`, which drives the
*unchanged* pipeline (`process_sample`) over the 5 cycles of a lane by monkeypatching
``epinuc_colocalization._open_accessor``. Each cycle is presented as one "time point", so the
pipeline's native cumulative new-only counting runs across the 5 antibody cycles (the
requirements doc's "stack the 5 cycles" step, done the pipeline's de-duplicated way).

Nothing in the analysis logic is duplicated or modified here.
"""

from __future__ import annotations

import os
import re
import glob
import time
import shutil
import tempfile
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import tifffile
    _HAVE_TIFFFILE = True
except Exception:  # pragma: no cover - tifffile is a hard dependency for real use
    _HAVE_TIFFFILE = False

import epinuc_colocalization as ep


# --------------------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------------------

# ch1_pos051_img0_Z1_R_cycle001_C001_000150_000.tif
# The laser group accepts any token, not just [RGB]: an unrecognised laser must reach
# assign_roles (where it becomes role=None and is reported) rather than fail the regex and be
# silently dropped from the index as an unparseable file.
_TIF_RE = re.compile(
    r"^ch(?P<ch>\d+)_pos(?P<pos>\d+)_img(?P<img>\d+)_"
    r"(?P<base>N|Z\d+)_(?P<laser>[A-Za-z0-9]+)_cycle(?P<cycle>\d+)_"
    r"(?P<sample>C\d+)_(?P<frame>\d+)_(?P<dataindex>\d+)\.tif$",
    re.IGNORECASE,
)

# Default role -> laser letter. Green (nucleosome template) / Red / Blue antibodies. Only a
# default: see the module docstring and configure_pipeline(tif_channel_map=...).
DEFAULT_TIF_CHANNEL_MAP = {"nucleosome": "G", "R_PTM": "R", "B_PTM": "B"}

# The three pipeline roles, in the channel-index order the accessor exposes them.
ROLE_ORDER = ["nucleosome", "R_PTM", "B_PTM"]


def parse_tif_name(fname: str) -> Optional[dict]:
    """Parse one EpiVision TIF basename into its fields, or ``None`` if it does not match.

    Returns a dict with ints for ``ch``/``pos``/``img``/``cycle``/``frame``/``dataindex``, the raw
    ``base``/``laser``/``sample`` strings, and a ``lane`` key (``"ch{N}"``). Deliberately does
    **not** assign a pipeline ``role``: that depends on the user's channel map, which may change
    after a run has been indexed. See :func:`assign_roles`.
    """
    m = _TIF_RE.match(os.path.basename(fname))
    if not m:
        return None
    g = m.groupdict()
    return {
        "ch": int(g["ch"]),
        "lane": f"ch{int(g['ch'])}",
        "pos": int(g["pos"]),
        "img": int(g["img"]),
        "base": g["base"].upper(),
        "laser": g["laser"].upper(),
        "cycle": int(g["cycle"]),
        "sample": g["sample"].upper(),
        "frame": int(g["frame"]),
        "dataindex": int(g["dataindex"]),
    }


INDEX_COLUMNS = ["path", "filename", "run", "ch", "lane", "pos", "img", "base",
                 "laser", "role", "cycle", "sample", "frame", "dataindex"]


# --------------------------------------------------------------------------------------
# Channel map: which laser carries which pipeline role
# --------------------------------------------------------------------------------------

def current_tif_channel_map() -> Dict[str, str]:
    """The active role -> laser map: ``ep.TIF_CHANNEL_MAP`` if set, else the default.

    Kept on ``ep`` (not here) so it rides along in ``ep.get_config()`` -- which is what
    ``epinuc_config.json`` serialises and what :func:`_run_fov_job` ships to the joblib workers.
    """
    return dict(getattr(ep, "TIF_CHANNEL_MAP", None) or DEFAULT_TIF_CHANNEL_MAP)


def validate_tif_channel_map(tif_channel_map: Dict[str, str]) -> Dict[str, str]:
    """Normalise (upper-case the lasers) and sanity-check a role -> laser map.

    Mirrors ``ep.validate_channel_map``: the nucleosome reference must be mapped, and no two
    roles may share a laser (that would analyse the same image twice under different names).
    """
    m = {r: str(tif_channel_map[r]).upper() for r in ROLE_ORDER if tif_channel_map.get(r)}
    missing = [r for r in ROLE_ORDER if r not in m]
    if "nucleosome" in missing:
        raise ValueError("TIF channel map must assign a laser to 'nucleosome' (the reference "
                         f"channel); got {tif_channel_map!r}")
    dupes = {l for l in m.values() if list(m.values()).count(l) > 1}
    if dupes:
        raise ValueError(f"TIF channel map assigns laser(s) {sorted(dupes)} to more than one role "
                         f"-- each role needs its own laser; got {tif_channel_map!r}")
    return m


def lasers_in_run(run_index: pd.DataFrame) -> List[str]:
    """Sorted list of the distinct laser tokens present in a run index (empty-safe).

    These are the choices a user picks from when assigning roles (the TIF analogue of
    ``ND2Accessor.channel_names``).
    """
    if run_index.empty:
        return []
    return sorted(run_index["laser"].dropna().unique().tolist())


def assign_roles(run_index: pd.DataFrame,
                 tif_channel_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Return a **copy** of ``run_index`` with its ``role`` column set from the channel map.

    Rows whose laser is not claimed by any role get ``role = None``; they stay in the index (so
    they can be reported) but every consumer filters them out. Defaults to the active map.
    """
    m = validate_tif_channel_map(tif_channel_map or current_tif_channel_map())
    laser_role = {laser: role for role, laser in m.items()}
    out = run_index.copy()
    if out.empty:
        out["role"] = pd.Series(dtype=object)
        return out
    out["role"] = out["laser"].map(laser_role).astype(object).where(lambda s: s.notna(), None)
    return out


def check_tif_channel_map(run_index: pd.DataFrame,
                          tif_channel_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Status table for a role -> laser map against a run index (mirrors ``ep.check_channel_map``).

    Columns: ``role | requested_laser | present | n_files``. ``present`` is False when the run
    contains no file acquired on the laser a role was assigned to -- the usual cause of an
    all-blank channel in the GUI.
    """
    m = dict(tif_channel_map or current_tif_channel_map())
    counts = (run_index["laser"].value_counts().to_dict() if not run_index.empty else {})
    rows = []
    for role in ROLE_ORDER:
        laser = m.get(role)
        laser = str(laser).upper() if laser else None
        rows.append({"role": role, "requested_laser": laser,
                     "present": bool(laser and laser in counts),
                     "n_files": int(counts.get(laser, 0)) if laser else 0})
    return pd.DataFrame(rows)


def index_run(run_dir: str) -> pd.DataFrame:
    """Index every TIF in ``run_dir`` into a DataFrame (one row per file).

    Columns: :data:`INDEX_COLUMNS` (path, filename, run + every field from
    :func:`parse_tif_name`, plus ``role`` from the active channel map). Files that don't match the
    naming pattern are warned about and skipped. The directory listing is retried a few times
    because the ``/Volumes/scBC`` SMB mount can transiently return an empty listing for a directory
    that is actually populated. Always returns a DataFrame with :data:`INDEX_COLUMNS` (empty if the
    run has no TIFs).

    To re-role an already-indexed run under a different channel map (e.g. when the user changes
    the GUI's pickers) call :func:`assign_roles` on the result -- no need to re-glob.
    """
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir does not exist: {os.path.abspath(run_dir)}")
    run = os.path.basename(os.path.normpath(run_dir))

    paths = sorted(glob.glob(os.path.join(run_dir, "*.tif")))
    for _ in range(_READ_RETRIES - 1):  # transient empty listing on the flaky mount
        if paths:
            break
        time.sleep(_READ_BACKOFF_S)
        paths = sorted(glob.glob(os.path.join(run_dir, "*.tif")))

    rows, skipped = [], 0
    for path in paths:
        fields = parse_tif_name(path)
        if fields is None:
            skipped += 1
            continue
        rows.append({"path": path, "filename": os.path.basename(path), "run": run, **fields})
    if skipped:
        warnings.warn(f"{run}: {skipped} TIF(s) did not match the expected name pattern.")
    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    if df.empty:
        warnings.warn(f"No parseable TIFs found in {os.path.abspath(run_dir)}")
        return df

    df = assign_roles(df)
    unclaimed = sorted(df.loc[df["role"].isna(), "laser"].unique().tolist())
    if unclaimed:
        warnings.warn(f"{run}: laser(s) {unclaimed} are not assigned to any role by the current "
                      f"channel map {current_tif_channel_map()} -- those files will be ignored.")
    return df


def lanes_in_run(run_index: pd.DataFrame) -> List[str]:
    """Sorted list of lane keys (``ch1``, ``ch2``, ...) present in a run index (empty-safe)."""
    if run_index.empty or "lane" not in run_index:
        return []
    return sorted(run_index["lane"].dropna().unique(), key=lambda s: int(s[2:]))


# --------------------------------------------------------------------------------------
# TIF reading (with retry-attempt stacking)
# --------------------------------------------------------------------------------------

# The /Volumes/scBC SMB mount intermittently drops reads of files that exist. Retry a few
# times with a short backoff before giving up, so a whole run isn't lost to a transient hiccup.
_READ_RETRIES = 5
_READ_BACKOFF_S = 0.5


def _read_tif(path: str) -> np.ndarray:
    if not _HAVE_TIFFFILE:
        raise ImportError("The 'tifffile' package is required. pip install tifffile")
    last_err = None
    for attempt in range(_READ_RETRIES):
        try:
            arr = np.asarray(tifffile.imread(path), dtype=np.float32)
            return np.squeeze(arr)
        except (FileNotFoundError, OSError) as e:  # transient network-mount dropout
            last_err = e
            time.sleep(_READ_BACKOFF_S * (attempt + 1))
    raise last_err


def read_slot(paths: List[str]) -> np.ndarray:
    """Read one acquisition slot into a single 2D float32 image.

    A slot is one (lane, pos, dye, cycle). It may hold >1 file when the instrument re-tried
    focus (distinct DataIndex/FrameNumber); those retry attempts are **max-projected** into
    one image (matching the pipeline's ``Z_PROJECTION_METHOD='max'`` convention).
    """
    if not paths:
        raise ValueError("read_slot called with no paths")
    if len(paths) == 1:
        return _read_tif(paths[0])
    imgs = [_read_tif(p) for p in paths]
    return np.maximum.reduce(imgs).astype(np.float32)


# --------------------------------------------------------------------------------------
# Accessor mimicking ND2Accessor for one lane at one cycle
# --------------------------------------------------------------------------------------

class TiffCycleAccessor:
    """ND2Accessor-compatible view of ONE flowcell lane at ONE cycle.

    Presents the lane as a virtual ND2 with axes P=positions, T=1, C=3 roles. The pipeline
    reads planes through :func:`epinuc_colocalization.extract_channel_image`, which only calls
    ``resolve_channel`` and ``get_plane``/``get_zstack`` on this object.

    The ``nucleosome`` role is imaged at cycle 1 only, so this accessor always serves the
    **cycle-1 N template** for that role regardless of ``self.cycle`` (nucleosomes are static;
    the pipeline requires the reference channel present at every time point).
    """

    def __init__(self, lane_index: pd.DataFrame, cycle: int, positions: List[int]):
        # lane_index: rows of index_run() restricted to one lane (all cycles/roles/positions).
        self.lane_index = lane_index
        self.cycle = int(cycle)
        self.positions = list(positions)
        self.channel_names = list(ROLE_ORDER)
        self.sizes: Dict[str, int] = {
            "P": len(self.positions), "T": 1, "C": len(self.channel_names),
            "Y": 2048, "X": 2048,
        }
        # slot lookup: (pos, role, cycle) -> [paths sorted by dataindex]. Rows whose laser no role
        # claims carry role=None and are dropped here (groupby skips NaN keys; be explicit).
        self._slots: Dict[tuple, List[str]] = {}
        roled = lane_index[lane_index["role"].notna()]
        for (pos, role, cyc), grp in roled.groupby(["pos", "role", "cycle"]):
            key = (int(pos), str(role), int(cyc))
            self._slots[key] = list(grp.sort_values("dataindex")["path"])

    # -- context manager plumbing (mirrors ND2Accessor) --
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        pass  # no open OS handle; reads are lazy per plane

    # -- axis sizes --
    @property
    def n_scenes(self) -> int:
        return self.sizes["P"]

    @property
    def n_times(self) -> int:
        return 1

    @property
    def n_z(self) -> int:
        return 1

    # -- channel resolution (match by role name; identity CHANNEL_MAP) --
    def resolve_channel(self, key) -> Optional[int]:
        """Match by role name; also accepts an index or a list of aliases (see ep.CHANNEL_MAP)."""
        if key is None:
            return None
        if isinstance(key, (list, tuple)):
            for alias in key:
                idx = self.resolve_channel(alias)
                if idx is not None:
                    return idx
            return None
        if isinstance(key, (int, np.integer)):
            idx = int(key)
            return idx if 0 <= idx < len(self.channel_names) else None
        key_l = str(key).lower()
        for i, name in enumerate(self.channel_names):
            if name.lower() == key_l:
                return i
        return None

    # -- plane access --
    def _paths_for(self, scene: int, channel_index: int) -> List[str]:
        role = self.channel_names[channel_index]
        pos = self.positions[scene]
        cyc = 1 if role == "nucleosome" else self.cycle  # N template is cycle-1 only
        return self._slots.get((int(pos), role, int(cyc)), [])

    def get_plane(self, scene: int = 0, time: int = 0, channel_index: int = 0,
                  z_index: Optional[int] = None) -> Optional[np.ndarray]:
        """Return one 2D (Y, X) float32 plane, or None if that role/cycle is absent here."""
        paths = self._paths_for(scene, channel_index)
        if not paths:
            return None
        return read_slot(paths)

    def get_zstack(self, scene: int = 0, time: int = 0, channel_index: int = 0) -> Optional[np.ndarray]:
        # No Z axis in this data; the pipeline only calls this when n_z > 1, but provide it.
        return self.get_plane(scene=scene, time=time, channel_index=channel_index)


# --------------------------------------------------------------------------------------
# Driver: run the unchanged pipeline over one lane's 5 cycles
# --------------------------------------------------------------------------------------

# Identity channel map so extract_channel_image resolves each role to the accessor's channel.
IDENTITY_CHANNEL_MAP = {
    "nucleosome": "nucleosome",
    "R_PTM": "R_PTM",
    "B_PTM": "B_PTM",
    "beads": None,
}


def configure_pipeline(pixel_size_um: Optional[float] = None,
                       artifact_masking: bool = True,
                       tif_channel_map: Optional[Dict[str, str]] = None) -> None:
    """Point the pipeline's channel map at the identity roles this loader exposes.

    Safe to call repeatedly. Does not touch spot-detection / colocalization parameters, so
    the tuned ``epinuc_config.json`` values still apply if loaded via ``ep.apply_config``.

    ``tif_channel_map`` is the *role -> laser* assignment (e.g. ``{"nucleosome": "G", "R_PTM": "R",
    "B_PTM": "B"}``) applied when an index is built or re-roled. It is validated here and stored on
    ``ep.TIF_CHANNEL_MAP``, distinct from ``ep.CHANNEL_MAP`` (which this loader keeps pinned to the
    identity map below, and which the ND2 path owns). Pass ``None`` to leave the active map alone.

    Also disables the across-time COARSE bead-vote (``TIME_COARSE_ALIGN``). Rationale specific
    to this TIF data: the nucleosome (green) template is imaged once and reused every cycle, so
    each cycle's red/blue is registered onto that single fixed template by the per-cycle CHANNEL
    registration. That already places all cycles in one common frame, leaving **no residual
    across-cycle drift** to correct — the time-alignment must be identity. The coarse bead-vote,
    however, recovers the raw R/B stage drift (which channel registration has *already* removed)
    from the drifted per-cycle bead sets and would double-apply it to the static nucleosome,
    mis-registering it and inflating the cumulative "new" counts. With the coarse vote off,
    time-alignment falls through to the (identical) nucleosome constellation and correctly
    resolves to identity. (For ND2 data, where green re-drifts each cycle, leave this on.)
    """
    _ensure_tif_roles()
    if tif_channel_map is not None:
        ep.TIF_CHANNEL_MAP = validate_tif_channel_map(tif_channel_map)
    elif getattr(ep, "TIF_CHANNEL_MAP", None):
        ep.TIF_CHANNEL_MAP = validate_tif_channel_map(ep.TIF_CHANNEL_MAP)  # e.g. straight from JSON
    # Mask saturated blob/streak artifacts (debris that clips the sensor). These are bright in
    # every channel and otherwise get used as false fiducials / seed false spots. See
    # ep.detect_artifact_mask and the ARTIFACT_* config knobs.
    ep.ARTIFACT_MASKING = bool(artifact_masking)
    if pixel_size_um is not None:
        ep.PIXEL_SIZE_UM = float(pixel_size_um)


def _ensure_tif_roles() -> None:
    """Structural setup this loader always needs: identity channel map + no across-cycle coarse
    vote (see configure_pipeline for the TIME_COARSE_ALIGN rationale). Deliberately does NOT touch
    ARTIFACT_MASKING or other tunables, so a caller's chosen preferences/config survive."""
    ep.CHANNEL_MAP = dict(IDENTITY_CHANNEL_MAP)
    ep.TIME_COARSE_ALIGN = False


def save_config(path: str = "epinuc_config.json") -> str:
    """Save the tuned parameters, preserving whatever ``CHANNEL_MAP`` the file already held.

    Both paths share one ``epinuc_config.json``. On the TIF path ``ep.CHANNEL_MAP`` is pinned to
    :data:`IDENTITY_CHANNEL_MAP`, so a plain ``ep.save_config`` would write
    ``{"nucleosome": "nucleosome", ...}`` over the ND2 path's real channel names -- which
    ``ep.validate_channel_map`` then rejects against an actual ND2. Write ``TIF_CHANNEL_MAP`` (our
    key) and leave ``CHANNEL_MAP`` (theirs) exactly as we found it.
    """
    ep.TIF_CHANNEL_MAP = current_tif_channel_map()  # write the default explicitly, never null
    return ep.save_config(path, preserve=("CHANNEL_MAP",))


def run_channel(run_index: pd.DataFrame, lane: str, sample_id=None,
                scenes=None, export_dir: Optional[str] = None,
                save_qc: bool = False, show_progress: bool = True,
                accumulator=None):
    """Process ONE flowcell lane (its nucleosome + red/blue pair across 5 cycles).

    Presents each cycle as one pipeline "time point" and calls the unchanged
    :func:`epinuc_colocalization.process_sample`, temporarily monkeypatching
    ``ep._open_accessor`` so it returns a :class:`TiffCycleAccessor` per cycle. Returns the
    pipeline's ResultAccumulator.
    """
    _ensure_tif_roles()  # structural only; leaves ARTIFACT_MASKING as configured by the caller
    lane_index = run_index[run_index["lane"] == lane]
    if lane_index.empty:
        raise ValueError(f"lane {lane!r} not present in this run index")

    positions = sorted(lane_index["pos"].unique())
    roled = lane_index[lane_index["role"].notna()]  # unclaimed lasers contribute no cycles
    cycles = sorted(roled.loc[roled["role"] != "nucleosome", "cycle"].unique())
    if not cycles:
        cycles = sorted(roled["cycle"].unique())

    # Build one accessor per cycle and register them under synthetic "paths".
    registry: Dict[str, TiffCycleAccessor] = {}
    files_ordered: List[str] = []
    run = str(lane_index["run"].iloc[0])
    for cyc in cycles:
        key = f"{run}::{lane}::cycle{cyc:03d}"
        registry[key] = TiffCycleAccessor(lane_index, cyc, positions)
        files_ordered.append(key)

    original_open = ep._open_accessor
    original_provider = ep.ARTIFACT_MASK_PROVIDER

    def _patched_open(path: str):
        try:
            return registry[path]
        except KeyError:
            return original_open(path)

    ep._open_accessor = _patched_open
    # One fixed artifact mask per FOV, unioned over ALL cycles (see precompute_artifact_masks):
    # the nucleosome template exists only at cycle 1, and a channel-specific artifact can fade in a
    # later cycle, so the mask must be built once and carried through every cycle.
    if ep.ARTIFACT_MASKING:
        scene_list = list(range(len(positions))) if scenes is None else list(scenes)
        masks = precompute_artifact_masks(registry, cycles, scene_list, show_progress=show_progress)
        ep.ARTIFACT_MASK_PROVIDER = lambda scene: masks.get(scene)
    try:
        acc = ep.process_sample(
            files_ordered, sample_id=sample_id, scenes=scenes,
            save_qc=save_qc, export_dir=export_dir,
            accumulator=accumulator, show_progress=show_progress,
        )
    finally:
        ep._open_accessor = original_open
        ep.ARTIFACT_MASK_PROVIDER = original_provider
    return acc


def precompute_artifact_masks(registry: Dict[str, "TiffCycleAccessor"], cycles: List[int],
                              scene_list: List[int], show_progress: bool = False
                              ) -> Dict[int, np.ndarray]:
    """One artifact mask per FOV, unioned over the nucleosome template and EVERY cycle's R/B.

    Debris is physically stuck on the flowcell, so an artifact sits at the same (y, x) in every
    cycle. Two reasons the mask must be built across all cycles rather than per cycle:

    * the nucleosome (green) template is imaged only at cycle 1 -- its blobs must carry through;
    * a channel-specific artifact (e.g. the dim blue-only streak) can fade below the detection
      threshold in a later cycle. A per-cycle mask would miss it there, and the pipeline's
      cumulative de-duplication would then count its false spots as brand-new detections.

    Returns ``{scene_index: bool mask}`` for the requested scenes.
    """
    accs = [registry[k] for k in sorted(registry)]
    if not accs:
        return {}
    green_acc = accs[0]  # nucleosome role always resolves to the cycle-1 template

    def _bar(it):
        if not show_progress:
            return it
        try:
            from tqdm.auto import tqdm
            return tqdm(it, desc="artifact masks (all cycles)", leave=False)
        except Exception:
            return it

    masks: Dict[int, np.ndarray] = {}
    for scene in _bar(scene_list):
        mask = ep.detect_artifact_mask(green_acc.get_plane(scene=scene, channel_index=0))
        for acc in accs:                       # every cycle
            for ci in (1, 2):                  # R_PTM, B_PTM
                m = ep.detect_artifact_mask(acc.get_plane(scene=scene, channel_index=ci))
                if m is not None:
                    mask = m if mask is None else (mask | m)
        if mask is not None:
            masks[scene] = mask
    return masks


# --------------------------------------------------------------------------------------
# Multiprocessing: FOV-level within a lane, so even a single sample uses every core
# --------------------------------------------------------------------------------------

def resolve_n_jobs(n_jobs: Optional[int] = None, cap: Optional[int] = None) -> int:
    """Resolve a joblib-style ``n_jobs`` to a positive worker count.

    ``None`` -> ``ep.N_JOBS`` (default -2). Negative values follow joblib: -1 = all cores,
    -2 = all cores but one (the default, keeps the machine responsive). Capped at ``cap``
    (the number of tasks) so we never spawn idle workers.
    """
    nj = ep.N_JOBS if n_jobs is None else n_jobs
    if nj is not None and nj < 0:
        nj = max(1, (os.cpu_count() or 1) + 1 + nj)   # -1 -> all, -2 -> all but one
    nj = max(1, int(nj or 1))
    if cap is not None:
        nj = max(1, min(nj, int(cap)))
    return nj


def _run_fov_job(lane_index: pd.DataFrame, lane: str, sample_id, scene: int,
                 config: dict, artifact_masking: bool, save_qc: bool):
    """Worker entry: process ONE field of view (all of its cycles) in this child process.

    A FOV's whole cycle time-series stays inside a single worker, so the pipeline's cumulative
    new-only counting is completely unaffected by the parallelism. Child processes start from the
    module defaults, so the parent's tuned config and the TIF role/artifact setup are re-applied
    here. Returns the FOV's ResultAccumulator for the parent to merge.
    """
    if config:
        ep.apply_config(config)                              # tuned SNR/thresholds/artifact knobs
    configure_pipeline(artifact_masking=artifact_masking)    # roles + time-align + artifact flag
    return run_channel(lane_index, lane, sample_id=sample_id, scenes=[int(scene)],
                       export_dir=None, save_qc=save_qc, show_progress=False)


def run_channel_parallel(run_index: pd.DataFrame, lane: str, sample_id=None, scenes=None,
                         export_dir: Optional[str] = None, n_jobs: Optional[int] = None,
                         save_qc: bool = False, artifact_masking: Optional[bool] = None,
                         show_progress: bool = True):
    """Process ONE lane with its **fields of view spread across worker processes**.

    One task per FOV (each FOV's whole 5-cycle series stays in one worker), so a single sample /
    single lane still saturates ``ncores-1`` workers. Results are merged in the parent, sorted, and
    exported once to ``export_dir``. Falls back to serial :func:`run_channel` if joblib is missing,
    ``n_jobs == 1``, or a worker raises. Returns the merged ResultAccumulator.
    """
    lane_index = run_index[run_index["lane"] == lane]
    if lane_index.empty:
        raise ValueError(f"lane {lane!r} not present in this run index")
    if artifact_masking is None:
        artifact_masking = bool(ep.ARTIFACT_MASKING)

    positions = sorted(lane_index["pos"].unique())
    scene_list = list(range(len(positions))) if scenes is None else [int(s) for s in scenes]
    nj = resolve_n_jobs(n_jobs, cap=len(scene_list))
    config = ep.get_config()

    def _bar(it, total, desc):
        if not show_progress:
            return it
        try:
            from tqdm.auto import tqdm
            return tqdm(it, total=total, desc=desc, leave=False)
        except Exception:
            return it

    acc = None
    if nj > 1 and len(scene_list) > 1:
        try:
            from joblib import Parallel, delayed
            tasks = (delayed(_run_fov_job)(lane_index, lane, sample_id, sc,
                                           config, artifact_masking, save_qc)
                     for sc in scene_list)
            pkw = {"n_jobs": nj, "backend": "loky"}
            if ep.PARALLEL_INNER_THREADS is not None:
                pkw["inner_max_num_threads"] = int(ep.PARALLEL_INNER_THREADS)
            desc = f"{sample_id}: FOVs (x{nj} workers)"
            try:  # joblib >= 1.3 streams results, so the bar ticks per finished FOV
                gen = Parallel(return_as="generator", **pkw)(tasks)
                parts = list(_bar(gen, len(scene_list), desc))
            except TypeError:
                parts = Parallel(verbose=5, **pkw)(tasks)
            acc = ep._merge_accumulators(ep.ResultAccumulator(), parts)
        except Exception as e:
            warnings.warn(f"Parallel FOV run failed ({e}); falling back to serial.")
            acc = None
    if acc is None:
        acc = run_channel(lane_index, lane, sample_id=sample_id, scenes=scene_list,
                          export_dir=None, save_qc=save_qc, show_progress=show_progress)

    # Deterministic, readable output regardless of the order workers finished in.
    acc.counts.sort(key=lambda r: (r.get("scene", 0), r.get("time", 0)))
    acc.cumulative.sort(key=lambda r: (r.get("scene", 0), r.get("time", 0)))
    if export_dir:
        ep.export_results(acc, output_dir=export_dir)
        if sample_id is not None:
            ep.write_sample_summary(acc, sample_id, export_dir)
    return acc


def run_channels_parallel(run_index_by_run: Dict[str, pd.DataFrame], jobs: List[tuple],
                          scenes=None, output_root: str = "per_run_output",
                          n_jobs: Optional[int] = None, save_qc: bool = False,
                          artifact_masking: Optional[bool] = None,
                          label_fn=None, show_progress: bool = True) -> pd.DataFrame:
    """Process many ``(run, lane)`` jobs, parallelising the FOVs **within** each lane.

    Lanes are handled one at a time (bounding memory: only one lane's spot tables are held at
    once) while that lane's FOVs fan out over ``ncores-1`` workers. This keeps every core busy even
    when there is only ONE sample / one lane -- unlike distributing whole lanes, which would leave
    a single-lane run serial. Each lane exports its CSVs to ``output_root/<run>/<lane>/``; the
    one-row summaries are concatenated (with ``run``/``lane`` columns) and returned.

    ``n_jobs`` default ``ep.N_JOBS`` (-2 = all cores but one). ``label_fn(run, lane) -> sample_id``
    defaults to :func:`lane_label`. ``artifact_masking`` defaults to the current
    ``ep.ARTIFACT_MASKING``. The parent's tuned ``ep`` config is captured and re-applied per worker.
    """
    label_fn = label_fn or (lambda run, lane: lane_label(run, lane))
    if artifact_masking is None:
        artifact_masking = bool(ep.ARTIFACT_MASKING)

    jobs = list(jobs)
    nj_report = resolve_n_jobs(n_jobs)
    if show_progress:
        print(f"Parallel: {len(jobs)} lane(s), FOVs spread over up to {nj_report} workers "
              f"({os.cpu_count()} cores detected).")

    def _bar(it):
        if not show_progress:
            return it
        try:
            from tqdm.auto import tqdm
            return tqdm(it, desc="lanes")
        except Exception:
            return it

    out = []
    for run, lane in _bar(jobs):
        sid = label_fn(run, lane)
        export_dir = os.path.join(output_root, run, lane)
        acc = run_channel_parallel(run_index_by_run[run], lane, sample_id=sid, scenes=scenes,
                                   export_dir=export_dir, n_jobs=n_jobs, save_qc=save_qc,
                                   artifact_masking=artifact_masking, show_progress=show_progress)
        summ = ep.sample_summary_frame(acc, sid).copy()
        summ.insert(0, "run", run)
        summ.insert(1, "lane", lane)
        out.append(summ)
        del acc  # free this lane's tables before the next one
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


RESULTS_FORMAT_RENAME = {
    "sample": "sample_id",
    "cumulative_n_nucleosomes": "#N",
    "cumulative_n_R_PTMs": "#Ab1(R)",
    "cumulative_n_B_PTMs": "#Ab2(B)",
    "cumulative_n_R_B_colocalized": "#(Ab1&Ab2)",
    "cumulative_n_triple_colocalized": "#(Ab1&Ab2&N)",
    "cumulative_n_R_colocalized_with_nucleosome": "#(Ab1&N)",
    "cumulative_n_B_colocalized_with_nucleosome": "#(Ab2&N)",
}
RESULTS_FORMAT_COLUMNS = ["run", "lane", "sample_id", "n_FOVs", "n_timepoints",
                          "#N", "#Ab1(R)", "#Ab2(B)", "#(Ab1&Ab2)", "#(Ab1&Ab2&N)",
                          "#(Ab1&N)", "#(Ab2&N)"]


def results_format_table(all_summaries: pd.DataFrame) -> pd.DataFrame:
    """Reshape the concatenated lane summaries into the EpiVision "Results Format" table."""
    if all_summaries is None or all_summaries.empty:
        return pd.DataFrame(columns=RESULTS_FORMAT_COLUMNS)
    df = all_summaries.rename(columns=RESULTS_FORMAT_RENAME)
    return df[[c for c in RESULTS_FORMAT_COLUMNS if c in df.columns]]


# --------------------------------------------------------------------------------------
# Run -> antibody/sample labels from "Run information.xlsx" (best effort)
# --------------------------------------------------------------------------------------

def load_run_labels(xlsx_path: str) -> Optional[pd.DataFrame]:
    """Best-effort read of the run-information spreadsheet (flaky network mount tolerant).

    Copies the file to a local temp path first (the ``/Volumes/scBC`` SMB mount intermittently
    fails direct reads) and retries. Returns a DataFrame of the first sheet, or ``None`` if the
    file cannot be read — callers should then fall back to ``NUC###_chN`` labels. The exact
    column layout is not assumed here; the caller inspects it and builds the run+lane -> antibody
    mapping.
    """
    if not xlsx_path:
        return None
    last_err = None
    for _ in range(3):
        try:
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                local = tmp.name
            shutil.copyfile(xlsx_path, local)
            df = pd.read_excel(local, sheet_name=0)
            try:
                os.unlink(local)
            except OSError:
                pass
            return df
        except Exception as e:  # network hiccup, permissions, missing file
            last_err = e
    warnings.warn(f"Could not read run-information xlsx ({xlsx_path}): {last_err}")
    return None


def _find_col(df: pd.DataFrame, *needles: str) -> Optional[str]:
    """First column whose name contains all ``needles`` (case-insensitive)."""
    for c in df.columns:
        name = str(c).lower()
        if all(n.lower() in name for n in needles):
            return c
    return None


# "K36me3 (Efrat's, Red)/K9me3 (Blue 1:15)" -> [("K36me3", "Efrat's, Red"), ("K9me3", "Blue 1:15")]
_ANTIBODY_RE = re.compile(r"([^()/]+?)\s*\(([^)]*)\)")


def parse_antibody_pair(description: str) -> Dict[str, Optional[str]]:
    """Pull the red and blue antibody names out of a 'Layer 2 Description' string.

    The descriptions look like ``"Tailed Plasma with 0.7 ul of each  K36me3 (Efrat's, Red)/K9me3
    (Blue 1:15)"`` -- each antibody is a name followed by a parenthetical that names its dye (plus
    noise like a vendor or dilution). We take every ``name (paren)`` pair, drop the boilerplate
    prefix from the first name, and assign by whichever of "red"/"blue" appears in the paren.
    Returns ``{"red": name|None, "blue": name|None}``.
    """
    out: Dict[str, Optional[str]] = {"red": None, "blue": None}
    if not isinstance(description, str):
        return out
    # Everything after the "... of each" boilerplate (absent -> whole string).
    tail = re.split(r"\bof each\b", description, flags=re.IGNORECASE)[-1]
    for name, paren in _ANTIBODY_RE.findall(tail):
        name = name.strip(" \t,/")
        paren_l = paren.lower()
        if "red" in paren_l and out["red"] is None:
            out["red"] = name
        elif "blue" in paren_l and out["blue"] is None:
            out["blue"] = name
    return out


def parse_run_information(source) -> pd.DataFrame:
    """Parse ``Run information.xlsx`` into one tidy row per (run, lane).

    ``source`` may be a path or an already-loaded DataFrame. The sheet repeats its header row
    between run blocks and only fills ``Run Name`` on each block's first row, so we drop the rows
    whose ``Channel`` isn't a number and then forward-fill the run/date. The description column
    carries the red/blue antibody pair for the whole run (see :func:`parse_antibody_pair`); the
    plasma sample varies per channel.

    Columns: ``run, channel, lane, plasma_sample, Ab1_red, Ab2_blue, date``.
    """
    df = load_run_labels(source) if isinstance(source, str) else source
    if df is None or df.empty:
        return pd.DataFrame(columns=["run", "channel", "lane", "plasma_sample",
                                     "Ab1_red", "Ab2_blue", "date"])
    c_run = _find_col(df, "run") or df.columns[1]
    c_ch = _find_col(df, "channel") or df.columns[2]
    c_sample = _find_col(df, "plasma", "sample") or df.columns[3]
    c_desc = _find_col(df, "description") or df.columns[-1]
    c_date = _find_col(df, "date")

    out = df.copy()
    out["_ch"] = pd.to_numeric(out[c_ch], errors="coerce")
    out = out[out["_ch"].notna()].copy()                    # drops the repeated header rows
    out[c_run] = out[c_run].ffill()                          # run name only on each block's 1st row
    if c_date is not None:
        with warnings.catch_warnings():   # object-dtype ffill downcast warning; harmless here
            warnings.simplefilter("ignore", FutureWarning)
            out[c_date] = out[c_date].ffill()

    rows = []
    for _, r in out.iterrows():
        ab = parse_antibody_pair(r[c_desc])
        rows.append({
            "run": str(r[c_run]).strip(),
            "channel": int(r["_ch"]),
            "lane": f"ch{int(r['_ch'])}",
            "plasma_sample": str(r[c_sample]).strip(),
            "Ab1_red": ab["red"],
            "Ab2_blue": ab["blue"],
            "date": r[c_date] if c_date is not None else None,
        })
    return pd.DataFrame(rows)


# Count columns -> antibody-oriented names for the annotated results table.
ANNOTATED_RENAME = {
    "cumulative_n_nucleosomes": "n_nucleosomes",
    "cumulative_n_R_PTMs": "n_Ab1",
    "cumulative_n_B_PTMs": "n_Ab2",
    "cumulative_n_R_colocalized_with_nucleosome": "n_Ab1_and_N",
    "cumulative_n_B_colocalized_with_nucleosome": "n_Ab2_and_N",
    "cumulative_n_R_B_colocalized": "n_Ab1_and_Ab2",
    "cumulative_n_triple_colocalized": "n_Ab1_and_Ab2_and_N",
}
ANNOTATED_COLUMNS = ["run", "lane", "plasma_sample", "Ab1_red", "Ab2_blue",
                     "n_FOVs", "n_timepoints", "n_nucleosomes",
                     "n_Ab1", "n_Ab2", "n_Ab1_and_Ab2", "n_Ab1_and_Ab2_and_N",
                     "n_Ab1_and_N", "n_Ab2_and_N"]


def annotate_summaries(all_summaries: pd.DataFrame, run_info) -> pd.DataFrame:
    """Join the per-lane results onto the run sheet: plasma sample + red/blue antibody names.

    ``run_info`` is a path to the xlsx or the DataFrame from :func:`parse_run_information` (a raw
    sheet DataFrame is parsed on the fly). Lanes with no sheet entry keep NaN labels rather than
    being dropped, so nothing is silently lost.
    """
    info = run_info
    if isinstance(info, str):
        info = parse_run_information(info)
    elif isinstance(info, pd.DataFrame) and "plasma_sample" not in info.columns:
        info = parse_run_information(info)
    if info is None or info.empty:
        warnings.warn("No run information available; results left unannotated.")
        info = pd.DataFrame(columns=["run", "lane", "plasma_sample", "Ab1_red", "Ab2_blue"])

    merged = all_summaries.merge(
        info[["run", "lane", "plasma_sample", "Ab1_red", "Ab2_blue"]],
        on=["run", "lane"], how="left")
    missing = merged["plasma_sample"].isna().sum()
    if missing:
        warnings.warn(f"{missing} lane(s) had no entry in the run information sheet.")
    merged = merged.rename(columns=ANNOTATED_RENAME)
    return merged[[c for c in ANNOTATED_COLUMNS if c in merged.columns]]


def annotated_long_table(annotated: pd.DataFrame) -> pd.DataFrame:
    """Tidy/long view: one row per (run, lane, antibody) with its own counts.

    Ab1 is the red antibody, Ab2 the blue. ``n_spots`` is that antibody's cumulative unique
    detections; ``n_with_nucleosome`` its colocalizations with the nucleosome template. The
    pair-wise columns (``n_with_other_antibody``, ``n_triple``) are the same for both rows of a
    lane, repeated for convenience.
    """
    rows = []
    for _, r in annotated.iterrows():
        for slot, dye, name, n_spots, n_with_n in (
                ("Ab1", "red", r.get("Ab1_red"), r.get("n_Ab1"), r.get("n_Ab1_and_N")),
                ("Ab2", "blue", r.get("Ab2_blue"), r.get("n_Ab2"), r.get("n_Ab2_and_N"))):
            rows.append({
                "run": r["run"], "lane": r["lane"], "plasma_sample": r.get("plasma_sample"),
                "antibody_slot": slot, "dye": dye, "antibody": name,
                "n_FOVs": r.get("n_FOVs"), "n_nucleosomes": r.get("n_nucleosomes"),
                "n_spots": n_spots, "n_with_nucleosome": n_with_n,
                "n_with_other_antibody": r.get("n_Ab1_and_Ab2"),
                "n_triple": r.get("n_Ab1_and_Ab2_and_N"),
            })
    return pd.DataFrame(rows)


def lane_label(run: str, lane: str, antibody: Optional[str] = None) -> str:
    """Compose a sample_id label: ``NUC###_chN__<antibody>`` (antibody omitted if unknown)."""
    base = f"{run}_{lane}"
    return f"{base}__{antibody}" if antibody else base


# --------------------------------------------------------------------------------------
# Convenience helpers for single-FOV inspection (notebook diagnostics & GUI)
# --------------------------------------------------------------------------------------

def lane_cycles(run_index: pd.DataFrame, lane: str) -> List[int]:
    """Antibody cycles present in a lane (sorted); nucleosome-only cycles excluded.

    Rows whose laser no role claims (``role`` is None) are ignored -- otherwise they would read as
    "not nucleosome" and invent phantom cycles.
    """
    li = run_index[(run_index["lane"] == lane) & run_index["role"].notna()]
    cyc = sorted(li.loc[li["role"] != "nucleosome", "cycle"].unique())
    return cyc or sorted(li["cycle"].unique())


def lane_positions(run_index: pd.DataFrame, lane: str) -> List[int]:
    """Sorted position ids present in a lane."""
    return sorted(run_index[run_index["lane"] == lane]["pos"].unique())


def lane_cycle_accessor(run_index: pd.DataFrame, lane: str, cycle: int) -> TiffCycleAccessor:
    """A :class:`TiffCycleAccessor` for one lane at one cycle."""
    li = run_index[run_index["lane"] == lane]
    if li.empty:
        raise ValueError(f"lane {lane!r} not present in this run index")
    return TiffCycleAccessor(li, int(cycle), sorted(li["pos"].unique()))


def lane_artifact_mask(run_index: pd.DataFrame, lane: str, scene: int,
                       show_progress: bool = False) -> Optional[np.ndarray]:
    """The all-cycles artifact mask for one FOV of a lane -- exactly what the pipeline applies.

    Thin wrapper over :func:`precompute_artifact_masks` for single-FOV inspection (notebook
    diagnostics, GUI), so what you see matches what ``run_channel`` actually masks.
    """
    li = run_index[run_index["lane"] == lane]
    if li.empty:
        raise ValueError(f"lane {lane!r} not present in this run index")
    positions = sorted(li["pos"].unique())
    cycles = lane_cycles(run_index, lane)
    registry = {f"cycle{c:03d}": TiffCycleAccessor(li, c, positions) for c in cycles}
    return precompute_artifact_masks(registry, cycles, [int(scene)], show_progress).get(int(scene))


def fov_planes(run_index: pd.DataFrame, lane: str, cycle: int, scene: int) -> Dict[str, Optional[np.ndarray]]:
    """Return ``{role: 2D image or None}`` for one FOV (position=scene) at one cycle.

    The nucleosome role always resolves to the cycle-1 template (see TiffCycleAccessor).
    """
    acc = lane_cycle_accessor(run_index, lane, cycle)
    return {role: acc.get_plane(scene=scene, time=0, channel_index=i)
            for i, role in enumerate(ROLE_ORDER)}
