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

## Which version do I want?

One analysis core (`epinuc_colocalization.py`) with **two front ends**, chosen by how the images
were acquired. The analysis, the tuning knobs and the config file are shared; only the loader,
the GUI's navigation, and the way you pick channels differ.

| | **ND2 version** | **TIF version** |
|---|---|---|
| Data | Nikon `.nd2` (all FOVs/times/channels in one file) | EpiVision instrument: flat folder of single-plane TIFs |
| Where | e.g. `T50_20260225/TS###_*.nd2` | `/Volumes/scBC/EpiVision/Images/NUC###/` |
| Loader | `epinuc_colocalization.py` (`ND2Accessor`) | `epinuc_tiff_loader.py` (`TiffCycleAccessor`) |
| GUI | `streamlit run epinuc_gui.py` | `streamlit run epinuc_tiff_gui.py` |
| Headless | `python epinuc_colocalization.py …` | `python epinuc_tiff_cli.py …` |
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

### `ND2_nucleosome_PTM_colocalization.py` / `.ipynb` — annotated notebook twin
The original, heavily-commented notebook version of the same pipeline, in jupytext "percent"
(`# %%`) format, with a paired `.ipynb`. It contains the same analysis logic as
`epinuc_colocalization.py` (the two are near-duplicates) plus a self-test block at the bottom.
Good for reading the step-by-step explanation of the method.

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
streamlit run epinuc_gui.py
```

### `demo_epinuc_colocalization.ipynb` — demo driver
A short notebook showing how to call `ep.run_samples(...)` for one sample or several, using
`scenes=range(3)` for a fast test or `scenes=None` for the full run.

### `Untitled.ipynb` — scratch
Exploratory scratch notebook (basic `nd2.imread` + channel normalization). Not part of the
pipeline.

---

## Config & dependencies

- **`epinuc_config.json`** — tuned analysis parameters loaded by the pipeline: per-channel
  spot-detection SNR / sigma / thresholds, colocalization & time-alignment radii (px), bead
  handling, pixel size, registration mode, and **both** channel maps — `CHANNEL_MAP` (ND2 channel
  name → role) and `TIF_CHANNEL_MAP` (laser letter → role). Written by either GUI's "Save" button;
  consumed via `run_samples(..., config_path="epinuc_config.json")` or `--config`.
  One file serves both paths: saving from the ND2 GUI preserves `TIF_CHANNEL_MAP`, and saving from
  the TIF GUI preserves `CHANNEL_MAP`.
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
   python epinuc_colocalization.py 1 --data-dir T50_20260225 --check-channels
   ```
3. `streamlit run epinuc_gui.py` → tune thresholds on a few FOVs → **Save** to `epinuc_config.json`.
4. Run the pipeline for the full set — **CLI** (the config carries every tuned parameter
   *including* `CHANNEL_MAP`):
   ```bash
   python epinuc_colocalization.py 1 2 3 \
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
| `epinuc_tiff_imaging.ipynb` | Driver notebook: inventory, diagnostics/overlays, full run, results table. |

### The workflow

```bash
# 1. Check which lasers the run actually contains.
python epinuc_tiff_cli.py --runs NUC388 --list

# 2. Tune interactively — assign each role to a laser (Channel → role), then the sliders for
#    spot k, coloc radius, bead + blob/streak artifact knobs. Watch the overlays update,
#    then press "💾 Save".
streamlit run epinuc_tiff_gui.py            # -> writes epinuc_config.json

# 3. Run headlessly with exactly those parameters.
python epinuc_tiff_cli.py --config epinuc_config.json --output per_run_output
```

The GUI and CLI share one source of truth: **`epinuc_config.json`**, written by `tl.save_config`
(every key in `ep.CONFIG_KEYS`, including `ARTIFACT_*` and `TIF_CHANNEL_MAP`) and read back by
`ep.load_config`. The parent captures that config and re-applies it inside each worker process, so
what you tuned is exactly what runs.

Useful CLI flags:

```bash
python epinuc_tiff_cli.py --list                        # inventory + the lasers in each run
python epinuc_tiff_cli.py --runs NUC388 --lanes ch1 ch2 # subset of runs / lanes
python epinuc_tiff_cli.py --scenes 0-3 --n-jobs 4       # FOV subset, 4 workers
python epinuc_tiff_cli.py --no-artifact-masking         # override the saved masking flag
python epinuc_tiff_cli.py --channel-map nucleosome=G,R_PTM=B,B_PTM=R   # override the saved map
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
