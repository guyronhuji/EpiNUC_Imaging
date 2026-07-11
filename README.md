# EPINUC Imaging — Nucleosome / PTM Colocalization

Analysis code for multichannel single-molecule fluorescence microscopy. It detects
diffraction-limited spots in three channels, registers/drift-corrects them against marker beads,
counts colocalizations, aligns the same fields of view across time points, and reports
**cumulative new-only counts** so nothing is double-counted across repeated acquisitions.

Channel roles throughout the code:

| Role | Channel | Meaning |
|------|---------|---------|
| `nucleosome` | Green (Cy3 / TIRF Cy3) | Nucleosome anchor spots |
| `R_PTM` | Red (Cy5 / TIRF Cy5) | "R" post-translational-modification mark |
| `B_PTM` | Blue (488 / TIRF 488) | "B" post-translational-modification mark |

Colocalizations reported per image and cumulatively across time: **R↔nucleosome**,
**B↔nucleosome**, **R+B+nucleosome triples**, and **R↔B pairs** (unique red–blue pairs within
the radius, independent of the nucleosome/green channel).

---

## Install & run

Dependencies are pinned with [uv](https://docs.astral.sh/uv/), so one command reproduces the exact,
tested environment (Python 3.11):

```bash
uv sync                 # build the env from uv.lock
uv sync --extra accel   # also install the optional OpenCV background backend
```

Run everything through `uv run` — it uses the project's environment, no manual activation:

```bash
uv run streamlit run epinuc_tiff_gui.py     # TIF QA / tuning GUI
uv run streamlit run epinuc_gui.py          # ND2 QA / tuning GUI
uv run python epinuc_tiff_cli.py --config epinuc_config.json --output per_run_output
uv run python epinuc_psf_correct.py --in .../NUC388 --flat-field --out ./corrected/NUC388
```

> **In a Dropbox/OneDrive folder?** Keep the virtualenv out of the synced tree, or it will sync
> thousands of files. Either tell the sync client to ignore `.venv/`, or point uv elsewhere:
> `export UV_PROJECT_ENVIRONMENT=~/.venvs/epinuc` before `uv sync`.

No uv? Plain pip works too, just without the pinning:
`pip install numpy pandas scipy scikit-image matplotlib tifffile nd2 xarray "dask[array]" joblib h5py streamlit tqdm`
(add `opencv-python` for the optional accelerator).

---

## Which version do I want?

One analysis core (`epinuc_colocalization.py`) with **two front ends**, chosen by how the images
were acquired. The analysis, the tuning knobs and the config file are shared; only the loader,
the GUI's navigation, and the way you pick channels differ.

| | **ND2 version** | **TIF version** |
|---|---|---|
| Data | Nikon `.nd2` (all FOVs/times/channels in one file) | EpiVision instrument: flat folder of single-plane TIFs |
| Where | e.g. `T50_20260225/TS###_*.nd2` | `/Volumes/scBC/EpiVision/Images/NUC###/` |
| Loader | `epinuc_colocalization.py` (`ND2Accessor`) | `epinuc_tiff_loader.py` (`TiffCycleAccessor`) |
| GUI | `uv run streamlit run epinuc_gui.py` | `uv run streamlit run epinuc_tiff_gui.py` |
| Headless | `uv run python epinuc_colocalization.py …` | `uv run python epinuc_tiff_cli.py …` |
| Navigate by | sample → timepoint (file) → FOV | run → lane → cycle → FOV |
| Time points are | separate ND2 files | antibody **cycles** (green template imaged once, at cycle 1) |
| **A "channel" is** | an ND2 metadata channel name (`"561 VF zyla Azide"`) | the **laser letter** in the filename (`G` / `R` / `B`) |
| Channel map key | `CHANNEL_MAP` | `TIF_CHANNEL_MAP` |

Both GUIs have a **Channel → role** picker in the sidebar and both write the same
`epinuc_config.json`. Each owns its own channel-map key, and **saving from one preserves the
other's** — so a single config file can drive both paths.

---

## Code files

### `epinuc_colocalization.py` — the production pipeline (importable module)
The single source of truth for the analysis. Everything else imports it. Loads ND2 files lazily
(memory-safe), detects beads, registers channels, detects spots, colocalizes, aligns fields of
view across time points, and writes result CSVs.

Key entry points:
- `run_samples(samples, data_dir=..., output_dir=..., config_path=...)` — run one or more
  samples end-to-end; returns a summary DataFrame and writes `samples_summary.csv`.
- `process_sample(...)` / `process_sample_parallel(...)` — process a single sample (serial /
  multi-core).
- `colocalize(nuc, r, b, ...)` — per-image colocalization counts + event table.

```python
import epinuc_colocalization as ep
ep.run_samples([1, 2, 3], data_dir="T50_20260225",
               output_dir="analysis_output", config_path="epinuc_config.json")
```

### `ND2_nucleosome_PTM_colocalization.py` — annotated pipeline twin
The original, heavily-commented version of the same pipeline, in jupytext "percent" (`# %%`)
format. It contains the same analysis logic as `epinuc_colocalization.py` (the two are
near-duplicates) plus a self-test block at the bottom. Good for reading the step-by-step
explanation of the method. (Its paired `.ipynb`, and the per-FOV exploration notebook, are kept
locally but not committed — the repo ships one demo per data path, see below.)

> **Keep in sync:** `epinuc_colocalization.py` and `ND2_nucleosome_PTM_colocalization.py` share
> the same core functions (`colocalize`, `update_cumulative`, the count/column schemas). A change
> to one usually needs the same change in the other.

### `epinuc_gui.py` — Streamlit QA & threshold-tuning GUI (ND2)
A browser front end for `epinuc_colocalization.py`. Assign each role to an ND2 channel, inspect
fields of view, check bead detection / registration, tune per-channel spot-detection thresholds
live, run quick batch QA, and save the tuned parameters to `epinuc_config.json` for the production
run. Reuses the pipeline's functions — no analysis logic is duplicated here.
(For TIF data use `epinuc_tiff_gui.py` instead — see below.)

```
uv run streamlit run epinuc_gui.py
```

### `epinuc_psf_correct.py` — standalone image correction & QC (TIF)
A self-contained tool (needs only numpy/scipy/tifffile) that reads raw EpiVision TIFs and either
reports per-image quality or writes corrected TIFs to a mirrored folder. It never touches the
pipeline — you point the pipeline at the corrected folder afterwards.

It does two separate jobs:

- **QC / measurement** (`--dry-run`, or `--measure-only` for speed): measures saturation, the
  half-light PSF radius, bead count, recovered stage drift, and the instrument's own focus flags
  from each image's fiducial beads, and writes them to a CSV. (This is also where the "the TIF PSF
  is not actually defocused" finding lives — see the module docstring.)
- **Flat-field correction** (`--flat-field`): builds a per-mode (R/G/B) flat field from the run's
  own frames plus an additive bias measured by photon transfer, then writes corrected `uint16`
  TIFs. Corrects every channel, keeps the filenames and XMP metadata, and puts the calibration in a
  sibling `<out>_flatfield/` folder so `--out` stays a clean TIF mirror.

```bash
# preview the per-mode bias/flat, write nothing
uv run python epinuc_psf_correct.py --in /Volumes/scBC/EpiVision/Images/NUC388 --flat-field --dry-run

# write corrected TIFs, then point the pipeline at ./corrected/NUC388
uv run python epinuc_psf_correct.py --in /Volumes/scBC/EpiVision/Images/NUC388 \
    --flat-field --out ./corrected/NUC388
```

```python
import epinuc_psf_correct as pc
cal = pc.build_flatfield(files, max_frames=80)      # {"R": Calib, "G": Calib, "B": Calib}
fixed = pc.apply_flatfield(raw_image, cal["R"])     # bias-subtract, divide the vignette
```

> On the current data the flat field is *safe but marginal*: it leaves the final colocalization and
> the registration method mix essentially unchanged. Its value is uniform-looking images, not better
> numbers — the real limiters (cross-cycle drift, saturation) are acquisition issues it can't fix.

### Demo notebooks — one per data path
Short, runnable starting points. Both take a fast subset first (`scenes=range(3)` /
`scenes=[0, 1]`) and leave the full run commented out.

| Notebook | Data | Shows |
|---|---|---|
| `demo_epinuc_colocalization.ipynb` | ND2 | `ep.run_samples(...)` for one sample or several. |
| `demo_epinuc_tiff.ipynb` | TIF | Index a run, inspect the lasers, assign **Channel → role**, run one lane, then the "Results Format" table. |

---

## Config & dependencies

- **`epinuc_config.json`** — tuned analysis parameters loaded by the pipeline: per-channel
  spot-detection SNR / sigma / thresholds, colocalization & time-alignment radii (px), bead
  handling, pixel size, registration mode, and **both** channel maps — `CHANNEL_MAP` (ND2 channel
  name → role) and `TIF_CHANNEL_MAP` (laser letter → role). Written by either GUI's "Save" button;
  consumed via `run_samples(..., config_path="epinuc_config.json")` or `--config`.
  One file serves both paths: saving from the ND2 GUI preserves `TIF_CHANNEL_MAP`, and saving from
  the TIF GUI preserves `CHANNEL_MAP`.
  It is a local, run-specific file and is **not tracked in the repo** (the code's built-in defaults
  in `epinuc_colocalization.py` are the shipped settings); generate your own from either GUI's
  **Save**, or pass one you already have via `--config`.
- **`requirements-gui.txt`** — the extra dependency for the GUI (`streamlit`) on top of the
  pipeline's own requirements (numpy, pandas, scipy, scikit-image, matplotlib, the `nd2`
  reader; optional `opencv`/`tqdm` accelerators).

> **Notebooks are committed without outputs.** Raw data and result tables stay off GitHub (see
> `.gitignore`). If you intend to *commit* a notebook, run `bash tools/setup-git-filters.sh` once
> per clone — it registers the clean filter that strips outputs on the way into git while leaving
> your working copy's figures intact. Git filter config isn't cloned, so this can't be automatic.

---

## Data & outputs

- **`T50_20260225/`** — a dataset: raw `TS###_*.nd2` acquisitions plus the original
  **CellProfiler** pipelines (`*.cpproj`) that this Python code reimplements, `Batch_data.h5`,
  and `Tricoloc_Cy3_Filtered.ipynb` (a Python reimplementation of the most recent CellProfiler
  pipeline).
- **`analysis_output/`, `analysis_output_sample1/`, `demo_output/`** — generated result tables.

Result CSVs written by a run:

| File | Contents |
|------|----------|
| `per_image_counts.csv` | Per-image spot & colocalization counts (incl. `n_R_B_colocalized`). |
| `cumulative_new_counts.csv` | Raw / new / cumulative counts per time point (de-duplicated across time). |
| `colocalization_events.csv` | One row per colocalization event (`R_nucleosome`, `B_nucleosome`, `R_B_nucleosome_triple`, `R_B`). |
| `spot_detections.csv` / `bead_detections.csv` | Detected spots / beads. |
| `registration_transforms.csv` | Per-channel / per-time registration transforms. |
| `time_aligned_spots.csv` | Nucleosome coordinates mapped into the reference-time frame. |
| `processing_log.csv` | Per-image processing log. |
| `<sample>_summary.csv` / `samples_summary.csv` | One-row-per-sample cumulative totals. |

---

## Typical workflow (ND2)

1. Put ND2 files in a data folder (e.g. `T50_20260225/`).
2. **Check the channels resolve** (see below):
   ```bash
   uv run python epinuc_colocalization.py 1 --data-dir T50_20260225 --check-channels
   ```
3. `uv run streamlit run epinuc_gui.py` → tune thresholds on a few FOVs → **Save** to `epinuc_config.json`.
4. Run the pipeline for the full set — **CLI** (the config carries every tuned parameter
   *including* `CHANNEL_MAP`):
   ```bash
   uv run python epinuc_colocalization.py 1 2 3 \
       --data-dir T50_20260225 --output-dir analysis_output \
       --config epinuc_config.json --n-jobs -2
   ```
   or from Python:
   ```python
   import epinuc_colocalization as ep
   ep.run_samples([1, 2, 3], data_dir="T50_20260225",
                  output_dir="analysis_output", config_path="epinuc_config.json")
   ```
5. Inspect the CSVs in the output folder.

### Channel map — ND2 (read this before a new dataset)

Which ND2 channel carries which role. Pick it in the sidebar of `epinuc_gui.py`, or set the
`CHANNEL_MAP` key. (The TIF path has its own picker and its own key — see
[Channel map — TIF](#channel-map--tif).)

`CHANNEL_MAP` values may be a name, a 0-based index, `None`, or a **list of aliases** tried in
order. The defaults cover both naming conventions seen so far:

```python
CHANNEL_MAP = {"nucleosome": ["TIRF Cy3", "Cy3", "561"],   # green, 561 excitation
               "R_PTM":      ["TIRF Cy5", "Cy5", "640"],
               "B_PTM":      ["TIRF 488", "488"],
               "beads": None}
```

so `TIRF Cy3` *and* `561 VF zyla Azide` both resolve to the nucleosome channel. If a role can't be
resolved, or two roles land on the same channel, the pipeline now **raises** instead of silently
skipping every field of view. `--check-channels` prints the resolution table.

### Sample names

If the dataset folder carries a CellProfiler `Batch_data.h5`, it also carries an
`EPINUC_sum_<date>.csv` in `output/` with the **lane → sample-name** key. The pipeline's sample id
(group 2 of `TS###_<sample>.nd2`) *is* the CellProfiler `Metadata_Lane`, so `run_samples(...)` picks
that file up automatically and adds `sample_name` (and `abs_set`) to `samples_summary.csv`:

```
sample,sample_name,abs_set,n_FOVs,...
2,Hmm 11,A,1,...
5,BCK 29,A,1,...
```

Control it with `--sample-names` (`auto` = default, a path to force a specific CSV, or `none` to
skip), or from Python `run_samples(..., sample_names={1: "L123", ...})`.
Note `Batch_data.h5` itself holds only the lane numbers — no names — so the CSV is the source of
truth; the h5's lane list is used to cross-check it.

### Bead detection

The default is `BEAD_DETECTION_METHOD = "multichannel"`: a fiducial is bright in *every* channel, so
each channel is normalised by its own robust noise and the per-pixel **minimum** is taken; peaks
above `BEAD_DETECTION_SNR` (default 150) are the beads. Per-channel Otsu (`"fast"`/`"log"`, still
selectable) was measured to miss beads (1 of 4) and to flood on channels full of bright molecules
(134 false beads; 7279 in green). The GUI's **Beads & QC** tab plots the sorted composite SNR so you
can see the bead/molecule valley and check the cut lands in it.

---

## EPINUC **TIF** runs (EpiVision instrument) — tune in the GUI, run from the CLI

The same pipeline, unchanged, over the flat TIF collections in
`/Volumes/scBC/EpiVision/Images/NUC###/`. Only the file-loading layer differs
(`epinuc_tiff_loader.py`). One flowcell lane `chN` = one EPINUC channel: a nucleosome template
(green, imaged at cycle 1 only, reused as a static template) plus a red/blue antibody pair over
5 cycles. Positions → FOVs, cycles → time points. Roles come from the filename's laser letter —
see [Channel map — TIF](#channel-map--tif).

| File | Role |
|------|------|
| `epinuc_tiff_loader.py` | TIF → pipeline adapter (parser, `TiffCycleAccessor`, artifact masks, multiprocessing). |
| `epinuc_tiff_gui.py` | Streamlit QA & tuning GUI, navigated **run → lane → cycle → position**, with a *Channel → role* picker. |
| `epinuc_tiff_cli.py` | Headless runner that consumes the GUI-tuned config; `--channel-map` overrides the laser assignment. |
| `demo_epinuc_tiff.ipynb` | Demo notebook: index a run, pick the channels, quick one-lane run, results table. |

### The workflow

```bash
# 1. Check which lasers the run actually contains.
uv run python epinuc_tiff_cli.py --runs NUC388 --list

# 2. Tune interactively — assign each role to a laser (Channel → role), then the sliders for
#    spot k, coloc radius, bead + blob/streak artifact knobs. Watch the overlays update,
#    then press "💾 Save".
uv run streamlit run epinuc_tiff_gui.py            # -> writes epinuc_config.json

# 3. Run headlessly with exactly those parameters.
uv run python epinuc_tiff_cli.py --config epinuc_config.json --output per_run_output
```

The GUI and CLI share one source of truth: **`epinuc_config.json`**, written by `tl.save_config`
(every key in `ep.CONFIG_KEYS`, including `ARTIFACT_*` and `TIF_CHANNEL_MAP`) and read back by
`ep.load_config`. The parent captures that config and re-applies it inside each worker process, so
what you tuned is exactly what runs.

Useful CLI flags:

```bash
uv run python epinuc_tiff_cli.py --list                        # inventory + the lasers in each run
uv run python epinuc_tiff_cli.py --runs NUC388 --lanes ch1 ch2 # subset of runs / lanes
uv run python epinuc_tiff_cli.py --scenes 0-3 --n-jobs 4       # FOV subset, 4 workers
uv run python epinuc_tiff_cli.py --no-artifact-masking         # override the saved masking flag
uv run python epinuc_tiff_cli.py --channel-map nucleosome=G,R_PTM=B,B_PTM=R   # override the saved map
```

Outputs land in `<output>/<run>/<lane>/` (the usual per-image / cumulative / event CSVs) plus
`all_lanes_summary.csv` and `results_format.csv` (the EpiVision "Results Format" table).

### Channel map — TIF

Here a "channel" is the **laser letter** in the filename, not the `chN` prefix (that's the flowcell
lane). In `ch1_pos051_img0_Z1_R_cycle001_C001_000150_000.tif` the laser is `R`:

```
ch1_pos051_img0_Z1_R_cycle001_C001_000150_000.tif
 |                  +-- LaserId: G=green (nucleosome template), R=red, B=blue
 +--------------------- flowcell lane (NOT a fluorescence channel)
```

The default assignment is `{"nucleosome": "G", "R_PTM": "R", "B_PTM": "B"}`. It is a property of
how the run was acquired, not of the format, so it is **selectable** three ways:

1. **GUI** — the *Channel → role* dropdowns in `epinuc_tiff_gui.py`'s sidebar, populated with the
   lasers actually present in the selected run. Press **💾 Save** to persist as `TIF_CHANNEL_MAP`.
2. **CLI** — `--channel-map nucleosome=G,R_PTM=B,B_PTM=R` (overrides the config file).
   Use `--list` first to see which laser tokens a run contains.
3. **Python** — `tl.configure_pipeline(tif_channel_map={"nucleosome": "G", ...})`, or re-role an
   existing index in place with `tl.assign_roles(idx, cmap)`.

Each role needs its own laser and `nucleosome` must be assigned — otherwise the GUI shows an error
and the CLI exits, rather than silently analysing one image under two names. A laser that no role
claims is reported and its files ignored. `tl.check_tif_channel_map(idx, cmap)` prints the
resolution table (the TIF analogue of `--check-channels`).

### Notes specific to this data

- **Artifact masking** (`ARTIFACT_MASKING`, on by default here) removes bright blob/streak debris:
  each channel is thresholded at its own `ARTIFACT_BRIGHT_PCT` percentile, connected regions
  larger than `ARTIFACT_MIN_AREA_PX` are kept and dilated by `ARTIFACT_DILATION_PX`. Beads and
  spots inside are dropped and the region is blanked from the background estimate. The mask is
  unioned over the template **and every cycle's** R/B, so it carries through all cycles.
- **Multiprocessing** is at the **FOV** level (`ncores-1` workers via `ep.N_JOBS = -2`), so a
  single sample still uses the whole machine. Each FOV's full 5-cycle series stays in one worker,
  so cumulative new-only counting is unaffected — results are bit-identical to a serial run.
- `TIME_COARSE_ALIGN` is forced off: the fixed green template already puts every cycle in one
  common frame, so across-cycle alignment must be the identity.

---

## API reference

The functions and classes you'd actually import. Conventional aliases: `import epinuc_colocalization
as ep`, `import epinuc_tiff_loader as tl`, `import epinuc_psf_correct as pc`.

### `epinuc_colocalization` (`ep`) — analysis core

| Call | Does |
|------|------|
| `run_samples(samples, data_dir=…, output_dir=…, config_path=…, channel_map=…, n_jobs=…)` | Run whole samples end-to-end; writes result CSVs, returns a summary DataFrame. |
| `process_sample(files_ordered, sample_id=…, scenes=…)` | One sample, serial. |
| `process_sample_parallel(files_ordered, …, n_jobs=…)` | One sample, FOV-parallel (bit-identical to serial). |
| `detect_beads_multichannel(images, file, scene, time, sigma=…, snr=…)` | `{role: bead DataFrame}` — fiducials bright in every channel (per-pixel min of z-maps). |
| `detect_spots(img, file, scene, time, channel_type, transform=…, bead_yx=…, artifact_mask=…)` | Diffraction-limited spots in one channel → `SPOT_COLUMNS` DataFrame. |
| `confirmed_bead_coords([nuc, r, b], tol=…, max_trusted=…)` | Cross-channel-confirmed bead `(y, x)` array (drops flooded channels). |
| `estimate_channel_transform(ref_beads, mov_beads, ref_img, mov_img, mode=…)` | `ChannelTransform` — bead-based, else phase-correlation, else identity. |
| `colocalize(nuc, r, b, file, scene, time, radius=…)` | `(events_df, counts)` for one image. |
| `get_config()` / `apply_config(cfg)` / `load_config(path)` / `save_config(path)` | Read / set / load-from-JSON / write the tuned parameter block. |

### `epinuc_tiff_loader` (`tl`) — TIF → pipeline adapter

| Call | Does |
|------|------|
| `index_run(run_dir)` | DataFrame indexing one run's TIFs (parses lane / pos / cycle / laser). |
| `assign_roles(run_index, channel_map=…)` | Add the `role` column from a laser→role map. |
| `fov_planes(run_index, lane, cycle, scene)` | `{role: 2D image or None}` for one FOV (nucleosome resolves to the cycle-1 template). |
| `configure_pipeline(pixel_size_um=…, tif_channel_map=…)` | Point `ep` at the TIF conventions (identity channel map, no coarse time vote). |
| `current_tif_channel_map()` / `check_tif_channel_map(idx, cmap)` / `validate_tif_channel_map(cmap)` | Inspect / verify the laser→role assignment. |
| `lanes_in_run(idx)` | Sorted `chN` lane keys present. |
| `save_config(path)` | Write the config (mirrors `ep.save_config`, preserving the other path's channel map). |

### `epinuc_psf_correct` (`pc`) — flat-field & QC

| Call | Does |
|------|------|
| `build_flatfield(files, max_frames)` | `{mode: Calib}` — per-mode median flat + photon-transfer bias (green transfers the R/B bias when its own fit is unreliable). |
| `apply_flatfield(img, cal)` | `(img - bias) / gain_norm` → corrected `float32` (bias subtracted, vignette divided). |
| `Calib(mode, flat, bias, gain, valid, n)` | Calibration for one mode; `.gain_map()` returns the cached normalized gain. |
| `photon_transfer_bias(stack)` | `(bias, gain_e_per_dn, valid)` from tile variance-vs-mean. |
| `save_calibration(cal, dir)` / `load_calibration(dir)` | Persist / reuse a calibration (the `<out>_flatfield/` folder; reuse with `--flat-load`). |
| `measure_psf(img, coords)` / `find_fiducials(images)` | PSF (half-light radius) and cross-channel fiducial detection, for the QC path. |
| `parse_tif_name(name)` / `read_tif(path)` | Filename fields; `(float32 image, XMP bytes)` with SMB-dropout retry. |

Run the tool from the shell with `uv run python epinuc_psf_correct.py --help`.
