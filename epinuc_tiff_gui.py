"""EPINUC QA & threshold-tuning GUI for the EpiVision TIF runs (Streamlit).

The TIF twin of ``epinuc_gui.py``. Same analysis and QA — inspect fields of view, check bead
detection / registration, tune per-channel spot-detection thresholds live, run quick batch QA,
save the tuned parameters to ``epinuc_config.json`` — but the data source is the flat TIF
collection under ``/Volumes/scBC/EpiVision/Images/<NUC###>/`` instead of ND2 files.

Navigation is **run → lane (chN) → cycle → position (FOV)** rather than file → timepoint → scene.
All detection / registration / colocalization is delegated to ``epinuc_colocalization`` via the
``epinuc_tiff_loader`` accessor, so no analysis logic is duplicated. (Time-alignment / cumulative
counting is a whole-lane operation done by the notebook / pipeline, not this per-FOV GUI.)

Run:   streamlit run epinuc_tiff_gui.py
Needs: pip install streamlit tifffile   (plus the pipeline's own deps)
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import epinuc_colocalization as ep
import epinuc_tiff_loader as tl

st.set_page_config(page_title="EPINUC TIF QA & threshold tuning", layout="wide")
ROLES = ["nucleosome", "R_PTM", "B_PTM"]
ROLE_COLOR = {"nucleosome": "lime", "R_PTM": "red", "B_PTM": "deepskyblue"}
ROLE_LABEL = {"nucleosome": "Green / nucleosome", "R_PTM": "Red / R PTM", "B_PTM": "Blue / B PTM"}

# Identity channel map + no across-cycle coarse vote (see loader docstring). Per-FOV QA does not
# time-align, but this keeps the module consistent with how the pipeline runs this data.
tl.configure_pipeline()


# --------------------------------------------------------------------------- data helpers
@st.cache_data(show_spinner="Indexing run…")
def index_run_cached(run_dir):
    return tl.index_run(run_dir)


@st.cache_data(show_spinner="Reading FOV…")
def load_planes(run_dir, lane, cycle, scene, cmap_items):
    """{role: 2D image or None} for one FOV (position=scene) at one cycle in a lane.

    ``cmap_items`` is the role -> laser map as a sorted tuple: it selects which image each role
    reads *and* keys the cache, so re-assigning a channel invalidates the planes.
    """
    idx = tl.assign_roles(tl.index_run(run_dir), dict(cmap_items))
    return tl.fov_planes(idx, lane, int(cycle), int(scene))


def fig_image(img, cmap="gray", title="", points=None, pcolor="yellow", psize=18, pmark="+"):
    """A matplotlib figure of one image with optional scatter overlay."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if img is not None:
        ax.imshow(ep._norm01(img), cmap=cmap)
    if points is not None and len(points):
        if pmark == "o":
            ax.scatter(points[:, 1], points[:, 0], s=psize, marker="o",
                       facecolors="none", edgecolors=pcolor, linewidths=1.0)
        else:
            ax.scatter(points[:, 1], points[:, 0], s=psize, marker=pmark, c=pcolor, linewidths=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return fig


def identity_tf():
    return ep.ChannelTransform(method="reference", success=True)


# --------------------------------------------------------------------------- sidebar: data
st.sidebar.title("EPINUC TIF QA")
images_root = st.sidebar.text_input("Images root", value="/Volumes/scBC/EpiVision/Images")

if not os.path.isdir(images_root):
    st.sidebar.error(f"Not reachable: {images_root}\n(the /Volumes/scBC share may be unmounted)")
    st.stop()
runs = sorted(d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d)))
if not runs:
    st.sidebar.error("No run folders found.")
    st.stop()

run = st.sidebar.selectbox("Run", runs, index=0)
run_dir = os.path.join(images_root, run)
try:
    idx = index_run_cached(run_dir)
except Exception as e:
    st.sidebar.error(f"Could not index run: {e}")
    st.stop()
if idx.empty:
    st.sidebar.error("No parseable TIFs in this run.")
    st.stop()

# ------------------------------------------------------------------- sidebar: channel -> role
# The TIF "channel" is the laser letter in the filename (…_Z1_R_cycle001_…), not the `chN` lane.
# Mirrors the ND2 GUI's channel picker; the choice re-roles the index below and is saved under
# the TIF_CHANNEL_MAP config key.
st.sidebar.markdown("**Channel → role**")
opts = tl.lasers_in_run(idx)
default_map = tl.current_tif_channel_map()


def _default_idx(role):
    """Index of this role's mapped laser, else a DISTINCT positional fallback.

    Never fall back to 0 for every role: that would silently point all three roles at the same
    laser (analysing the red image as "green"), producing nonsense beads and spots.
    """
    laser = default_map.get(role)
    if laser in opts:
        return opts.index(laser)
    return ROLES.index(role) % max(1, len(opts))


_absent = [r for r in ROLES if default_map.get(r) not in opts]
if _absent:
    st.sidebar.warning(
        f"The saved channel map does not resolve {_absent} against this run's lasers {opts}. "
        f"Using positional defaults — **check the assignment below** (the nucleosome template is "
        f"normally the green 'G' laser).")
    st.sidebar.dataframe(tl.check_tif_channel_map(idx, default_map), use_container_width=True)

cmap = {}
for role in ROLES:
    cmap[role] = st.sidebar.selectbox(ROLE_LABEL[role], opts, index=_default_idx(role),
                                      key=f"cmap_{role}")
cmap_items = tuple(sorted(cmap.items()))

if len(set(cmap.values())) < len(ROLES):
    st.sidebar.error(
        "Two roles point at the same laser — each role needs its own. "
        "The nucleosome template is normally the green 'G' one.")
    st.stop()

# Re-role the index (a copy) so every helper below reads the user's assignment, and publish the
# map so Save picks it up.
ep.TIF_CHANNEL_MAP = dict(cmap)
idx = tl.assign_roles(idx, cmap)

lanes = tl.lanes_in_run(idx)
lane = st.sidebar.selectbox("Lane (flowcell channel)", lanes, index=0)
cycles = tl.lane_cycles(idx, lane)
cycle = st.sidebar.selectbox("Cycle (antibody round)", cycles, index=0)
positions = tl.lane_positions(idx, lane)
pos_i = st.sidebar.selectbox("Position (FOV)", range(len(positions)),
                             format_func=lambda i: f"FOV {i} — pos{positions[i]:03d}")
scene = int(pos_i)
sample_c = sorted(idx[idx["lane"] == lane]["sample"].unique())
st.sidebar.caption(f"SampleId: {', '.join(sample_c)} · {len(positions)} FOVs · cycles {cycles}")

# --------------------------------------------------------------------------- sidebar: params
st.sidebar.markdown("---")
st.sidebar.subheader("Detection parameters")

if "seeded" not in st.session_state:
    d = ep.get_config()
    st.session_state.update({
        "k_nucleosome": float(d["SPOT_DETECTION_SNR"]["nucleosome"]),
        "k_R_PTM": float(d["SPOT_DETECTION_SNR"]["R_PTM"]),
        "k_B_PTM": float(d["SPOT_DETECTION_SNR"]["B_PTM"]),
        "coloc_r": float(d["COLOCALIZATION_RADIUS_PX"]),
        "bead_excl_r": float(d["BEAD_EXCLUSION_RADIUS_PX"]),
        "max_beads": int(d["MAX_TRUSTED_BEADS"]),
        "excl_beads": bool(d["EXCLUDE_BEADS_FROM_SPOTS"]),
        "mask_beads_bg": bool(d["MASK_BEADS_BEFORE_BACKGROUND"]),
        "bead_method": str(d.get("BEAD_DETECTION_METHOD", "fast")),
        "bead_snr": float(d.get("BEAD_DETECTION_SNR", 150.0)),
        "artifact_masking": bool(d.get("ARTIFACT_MASKING", True)),
        "artifact_pct": float(d.get("ARTIFACT_BRIGHT_PCT", 99.5)),
        "artifact_min_area": int(d.get("ARTIFACT_MIN_AREA_PX", 500)),
        "artifact_dilation": int(d.get("ARTIFACT_DILATION_PX", 15)),
        "seeded": True,
    })

st.sidebar.caption("Spot threshold = k × noise (SNR). Higher k → fewer spots.")
for role in ROLES:
    st.sidebar.slider(f"k — {ROLE_LABEL[role]}", 2.0, 20.0, step=0.5, key=f"k_{role}")
st.sidebar.slider("Colocalization radius (px)", 0.5, 10.0, step=0.5, key="coloc_r")
st.sidebar.slider("Bead-exclusion radius (px)", 0.0, 15.0, step=0.5, key="bead_excl_r")
st.sidebar.slider("Max trusted beads (flood cutoff)", 10, 2000, step=10, key="max_beads")
st.sidebar.checkbox("Exclude beads from spots", key="excl_beads")
st.sidebar.checkbox("Mask beads before background", key="mask_beads_bg",
                    help="Fill bright bead footprints with their local median before the "
                         "background blur, removing the halo-ring artifact that seeds false spots.")

st.sidebar.markdown("**Bead detector**")
st.sidebar.selectbox(
    "Method", ["fast", "log", "multichannel"], key="bead_method",
    help="'fast'/'log' threshold each channel on its own (Otsu). If the per-channel bead counts "
         "come out wildly different, Otsu's bimodality assumption has failed — switch to "
         "'multichannel', which keeps only peaks bright in EVERY channel (a fiducial by "
         "definition) and so rejects single molecules.")
if st.session_state["bead_method"] == "multichannel":
    st.sidebar.slider("Bead SNR (all channels)", 10.0, 1000.0, step=10.0, key="bead_snr",
                      help="A fiducial must reach this many robust-noise sigmas in EVERY channel "
                           "(on the cross-channel minimum). Beads land in the hundreds-thousands, "
                           "single molecules under ~100, so ~150 sits in the valley between them. "
                           "Raise if molecules leak in; lower if dim beads are missed.")

st.sidebar.markdown("**Blob/streak artifacts**")
st.sidebar.checkbox("Mask saturated blob/streak artifacts", key="artifact_masking",
                    help="Saturated aggregates/debris (bright in every channel) get used as false "
                         "fiducials and seed false spots/triples. When on, beads & spots inside "
                         "them are dropped and the region is blanked for the background estimate.")
if st.session_state["artifact_masking"]:
    st.sidebar.slider("Bright percentile", 98.0, 99.95, step=0.05, key="artifact_pct",
                      help="Per-channel brightness percentile for the threshold — adapts to each "
                           "channel (green field is bright, red/blue dim). Lower → catches fainter/larger "
                           "artifacts (and risks the field); higher → only the brightest.")
    st.sidebar.slider("Min region area (px)", 50, 2000, step=50, key="artifact_min_area",
                      help="Min connected bright area to count as an artifact. Set above the largest "
                           "real dense-signal cluster but below a blob/streak. Lower → catches smaller "
                           "debris (and risks dense real clusters); higher → only big blobs/streaks.")
    st.sidebar.slider("Mask dilation (px)", 0, 40, step=1, key="artifact_dilation",
                      help="Grow the mask to cover the surrounding halo / diffraction spikes / tail.")

# apply widget values to module globals (drives all detection below)
ep.SPOT_DETECTION_SNR = {r: float(st.session_state[f"k_{r}"]) for r in ROLES}
ep.COLOCALIZATION_RADIUS_PX = float(st.session_state["coloc_r"])
ep.BEAD_EXCLUSION_RADIUS_PX = float(st.session_state["bead_excl_r"])
ep.MAX_TRUSTED_BEADS = int(st.session_state["max_beads"])
ep.EXCLUDE_BEADS_FROM_SPOTS = bool(st.session_state["excl_beads"])
ep.MASK_BEADS_BEFORE_BACKGROUND = bool(st.session_state["mask_beads_bg"])
ep.BEAD_DETECTION_METHOD = str(st.session_state["bead_method"])
ep.BEAD_DETECTION_SNR = float(st.session_state["bead_snr"])
ep.ARTIFACT_MASKING = bool(st.session_state["artifact_masking"])
ep.ARTIFACT_BRIGHT_PCT = float(st.session_state["artifact_pct"])
ep.ARTIFACT_MIN_AREA_PX = int(st.session_state["artifact_min_area"])
ep.ARTIFACT_DILATION_PX = int(st.session_state["artifact_dilation"])

# save / load / reset
st.sidebar.markdown("---")
cfg_path = st.sidebar.text_input("Config file", value="epinuc_config.json")
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("💾 Save"):
    tl.save_config(cfg_path)  # not ep.save_config: keeps the ND2 path's CHANNEL_MAP intact
    st.sidebar.success(f"Saved {cfg_path}")
if c2.button("📂 Load") and os.path.isfile(cfg_path):
    cfg = ep.load_config(cfg_path)
    snr = cfg.get("SPOT_DETECTION_SNR", {})
    for r in ROLES:
        if r in snr:
            st.session_state[f"k_{r}"] = float(snr[r])
    # Re-seat the channel pickers too, else the rerun immediately overwrites the loaded map with
    # the stale widget values. Skip lasers this run doesn't have — a selectbox whose session_state
    # value is outside its options raises, and _default_idx already handles the absent case.
    for r, laser in (cfg.get("TIF_CHANNEL_MAP") or {}).items():
        if r in ROLES and str(laser).upper() in opts:
            st.session_state[f"cmap_{r}"] = str(laser).upper()
    for key, ck in [("coloc_r", "COLOCALIZATION_RADIUS_PX"), ("bead_excl_r", "BEAD_EXCLUSION_RADIUS_PX"),
                    ("max_beads", "MAX_TRUSTED_BEADS"), ("excl_beads", "EXCLUDE_BEADS_FROM_SPOTS"),
                    ("mask_beads_bg", "MASK_BEADS_BEFORE_BACKGROUND"),
                    ("bead_method", "BEAD_DETECTION_METHOD"), ("bead_snr", "BEAD_DETECTION_SNR"),
                    ("artifact_masking", "ARTIFACT_MASKING"), ("artifact_pct", "ARTIFACT_BRIGHT_PCT"),
                    ("artifact_min_area", "ARTIFACT_MIN_AREA_PX"), ("artifact_dilation", "ARTIFACT_DILATION_PX")]:
        if ck in cfg:
            st.session_state[key] = cfg[ck]
    st.rerun()
if c3.button("↺ Reset"):
    del st.session_state["seeded"]
    ep.TIF_CHANNEL_MAP = None  # back to the loader's default laser assignment
    for r in ROLES:
        st.session_state.pop(f"cmap_{r}", None)
    st.rerun()

# --------------------------------------------------------------------------- load this FOV
planes = load_planes(run_dir, lane, int(cycle), scene, cmap_items)
green, red, blue = planes["nucleosome"], planes["R_PTM"], planes["B_PTM"]
imgs = {"nucleosome": green, "R_PTM": red, "B_PTM": blue}
st.title("EPINUC (TIF) — QA & threshold tuning")
st.caption(f"{run} · {lane} · cycle {cycle} · FOV {scene} (pos{positions[scene]:03d}) · "
           f"lasers {', '.join(f'{r}←{cmap[r]}' for r in ROLES)} · "
           f"nucleosome shown is the cycle-1 template")

tabs = st.tabs(["Channels", "Beads & QC", "Registration", "Spots & tuning",
                "Colocalization", "Batch QA"])


@st.cache_data(show_spinner="Building all-cycles artifact mask…")
def _all_cycles_mask(run_dir, lane, scene, pct, min_area, dil, cmap_items):
    """Cached all-cycles artifact mask. The detection params and the channel map are cache keys so
    moving a slider or re-assigning a laser recomputes; ep.ARTIFACT_* globals are already set from
    those same widgets above."""
    idx = tl.assign_roles(tl.index_run(run_dir), dict(cmap_items))
    return tl.lane_artifact_mask(idx, lane, int(scene))


def artifact_mask():
    """The artifact mask the pipeline applies to this FOV (None if masking is off).

    Unioned over the nucleosome template AND every cycle's R/B — debris is fixed on the flowcell,
    green exists only at cycle 1, and a channel-specific artifact can fade in a later cycle, so one
    mask carries through all cycles. Cycle-independent by construction.
    """
    if not ep.ARTIFACT_MASKING:
        return None
    return _all_cycles_mask(run_dir, lane, int(scene), ep.ARTIFACT_BRIGHT_PCT,
                            ep.ARTIFACT_MIN_AREA_PX, ep.ARTIFACT_DILATION_PX, cmap_items)


def detect_all_beads():
    if ep.BEAD_DETECTION_METHOD == "multichannel":
        b = ep.detect_beads_multichannel(imgs, "", int(scene), 0)
    else:
        b = {r: ep.detect_beads(imgs[r], "", int(scene), 0, r) for r in ROLES}
    art = artifact_mask()
    if art is not None and art.any():   # drop artifact 'beads' before confirmation (as in the pipeline)
        b = {r: ep.exclude_points_in_mask(b[r], art)[0] for r in ROLES}
    conf = ep.confirmed_bead_coords([b["nucleosome"], b["R_PTM"], b["B_PTM"]])
    return b, conf


# ============================================================ Channels
with tabs[0]:
    cols = st.columns(4)
    cmaps = {"nucleosome": "Greens", "R_PTM": "Reds", "B_PTM": "Blues"}
    for ax_col, role in zip(cols[:3], ROLES):
        with ax_col:
            if imgs[role] is None:
                st.warning(f"{ROLE_LABEL[role]}: channel missing")
            else:
                st.pyplot(fig_image(imgs[role], cmap=cmaps[role], title=ROLE_LABEL[role]))
                st.caption(f"median={np.median(imgs[role]):.0f}  max={imgs[role].max():.0f}")
    with cols[3]:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.imshow(ep.rgb_overlay(green, red, blue)); ax.set_title("RGB overlay", fontsize=10); ax.axis("off")
        st.pyplot(fig)

# ============================================================ Beads & QC
with tabs[1]:
    beads, conf = detect_all_beads()
    any_flood = any(len(beads[r]) > ep.MAX_TRUSTED_BEADS for r in ROLES)
    cc = st.columns(3)
    for col, role in zip(cc, ROLES):
        with col:
            n = len(beads[role])
            st.metric(f"{ROLE_LABEL[role]} beads", n, delta="FLOODED" if n > ep.MAX_TRUSTED_BEADS else None,
                      delta_color="inverse")
    st.metric("Cross-channel confirmed beads (used for exclusion/alignment)", len(conf))
    if any_flood:
        st.warning(f"A bead detector flooded (> {ep.MAX_TRUSTED_BEADS}) on this FOV — expected on "
                   f"fields without clear fiducials. The cross-channel guard kept {len(conf)} "
                   f"confirmed beads, so real spots are protected (not deleted).")
    elif len(conf) == 0:
        st.info("No cross-channel-confirmed beads on this FOV — bead exclusion will remove nothing "
                "here (safe). Registration falls back to phase-correlation/identity.")
    ov = st.columns(2)
    with ov[0]:
        st.pyplot(fig_image(green, title="Green + per-channel beads (green ch)",
                            points=beads["nucleosome"][["y", "x"]].to_numpy() if len(beads["nucleosome"]) else None,
                            pcolor="orange", pmark="o", psize=40))
    with ov[1]:
        st.pyplot(fig_image(green, title=f"Green + {len(conf)} CONFIRMED beads",
                            points=conf if len(conf) else None, pcolor="yellow", pmark="o", psize=60))


    if ep.BEAD_DETECTION_METHOD == "multichannel":
        vals, n_pass = ep.bead_snr_diagnostics(imgs)
        if len(vals):
            st.markdown("**Bead SNR diagnostic** — composite = per-pixel MIN of the per-channel "
                        "noise-normalised images, so only peaks bright in *every* channel survive. "
                        "Fiducials sit in the hundreds–thousands, single molecules under ~100; the "
                        "cut should land in the empty valley between the two clouds.")
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            ax.semilogy(np.arange(1, len(vals) + 1), vals, "o", ms=4, color="0.35")
            ax.axhline(ep.BEAD_DETECTION_SNR, color="crimson", ls="--",
                       label=f"cut = {ep.BEAD_DETECTION_SNR:g} -> {n_pass} beads")
            ax.set_xlabel("candidate (sorted)"); ax.set_ylabel("composite SNR (log)")
            ax.legend(fontsize=8)
            st.pyplot(fig)
        else:
            st.info("No bead candidates above the diagnostic floor on this FOV.")

    art = artifact_mask()
    if art is not None:
        if art.any():
            from skimage.measure import label as _label
            st.metric("Saturated blob/streak artifacts",
                      f"{_label(art).max()} region(s) · {100*art.mean():.3f}% of FOV")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(ep._norm01(green), cmap="gray")
            ax.imshow(np.ma.masked_where(~art, np.ones_like(green)), cmap="autumn", alpha=0.45)
            ax.set_title("Green + artifact mask (red) — beads/spots here are dropped", fontsize=10)
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.caption("No saturated blob/streak artifacts detected on this FOV.")

# ============================================================ Registration
with tabs[2]:
    beads, conf = detect_all_beads()
    tf_r = ep.estimate_channel_transform(beads["nucleosome"], beads["R_PTM"], green, red, ep.REGISTRATION_MODE) if red is not None else identity_tf()
    tf_b = ep.estimate_channel_transform(beads["nucleosome"], beads["B_PTM"], green, blue, ep.REGISTRATION_MODE) if blue is not None else identity_tf()
    m = st.columns(2)
    m[0].info(f"Red→Green: {tf_r.method}  shift=({tf_r.dx:+.2f}, {tf_r.dy:+.2f})px  n={tf_r.n_matched}")
    m[1].info(f"Blue→Green: {tf_b.method}  shift=({tf_b.dx:+.2f}, {tf_b.dy:+.2f})px  n={tf_b.n_matched}")
    red_reg = ep.warp_to_reference(red, tf_r) if red is not None else None
    blue_reg = ep.warp_to_reference(blue, tf_b) if blue is not None else None
    cols = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 6)); ax.imshow(ep.rgb_overlay(green, red, blue))
    ax.set_title("Before registration", fontsize=10); ax.axis("off"); cols[0].pyplot(fig)
    fig, ax = plt.subplots(figsize=(6, 6)); ax.imshow(ep.rgb_overlay(green, red_reg, blue_reg))
    ax.set_title("After registration", fontsize=10); ax.axis("off"); cols[1].pyplot(fig)
    st.caption("Red/blue are registered onto the fixed cycle-1 green template; this transform "
               "absorbs the cycle's stage drift, so no separate across-cycle alignment is needed.")


def detect_all_spots():
    beads, conf = detect_all_beads()
    tf = {"nucleosome": identity_tf()}
    tf["R_PTM"] = ep.estimate_channel_transform(beads["nucleosome"], beads["R_PTM"], green, red, ep.REGISTRATION_MODE) if red is not None else identity_tf()
    tf["B_PTM"] = ep.estimate_channel_transform(beads["nucleosome"], beads["B_PTM"], green, blue, ep.REGISTRATION_MODE) if blue is not None else identity_tf()
    art = artifact_mask()
    spots = {r: ep.detect_spots(imgs[r], "", int(scene), 0, r, tf[r], bead_yx=conf, artifact_mask=art) for r in ROLES}
    if ep.EXCLUDE_BEADS_FROM_SPOTS and len(conf):
        for r in ROLES:
            spots[r], _ = ep.exclude_spots_near_beads(spots[r], conf)
    if art is not None and art.any():   # drop spots inside blob/streak artifacts
        for r in ROLES:
            spots[r], _ = ep.exclude_points_in_mask(spots[r], art)
    return spots

# ============================================================ Spots & tuning
with tabs[3]:
    with st.spinner("Detecting spots…"):
        spots = detect_all_spots()
    cc = st.columns(3)
    for col, role in zip(cc, ROLES):
        col.metric(f"{ROLE_LABEL[role]} spots", len(spots[role]))
    sel = st.selectbox("Overlay / diagnostics channel", ROLES, format_func=lambda r: ROLE_LABEL[r])
    left, right = st.columns([1, 1])
    with left:
        df = spots[sel]
        st.pyplot(fig_image(imgs[sel], title=f"{ROLE_LABEL[sel]} — {len(df)} spots (k={ep.SPOT_DETECTION_SNR[sel]})",
                            points=df[["y", "x"]].to_numpy() if len(df) else None,
                            pcolor=ROLE_COLOR[sel], pmark="+", psize=16))
    with right:
        img = imgs[sel]
        if img is not None:
            _, conf = detect_all_beads()
            ks = np.arange(3, 18.1, 1.0)
            saved = ep.SPOT_DETECTION_SNR[sel]
            art = artifact_mask()
            counts = []
            sweep_prog = st.progress(0.0, text=f"k-sweep ({ROLE_LABEL[sel]})…")
            for i, k in enumerate(ks):
                ep.SPOT_DETECTION_SNR[sel] = float(k)
                counts.append(len(ep.detect_spots(img, "", int(scene), 0, sel, identity_tf(),
                                                  bead_yx=conf, artifact_mask=art)))
                sweep_prog.progress((i + 1) / len(ks))
            sweep_prog.empty()
            ep.SPOT_DETECTION_SNR[sel] = saved
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(ks, counts, "-o", ms=3)
            ax.axvline(saved, color="crimson", ls="--", label=f"current k={saved}")
            ax.set_xlabel("k (σ above noise)"); ax.set_ylabel("spot count"); ax.set_yscale("log")
            ax.set_title(f"{ROLE_LABEL[sel]}: count vs k"); ax.legend(fontsize=8)
            st.pyplot(fig)
    st.caption("Move the k slider (sidebar) until counts and the overlay look right for each "
               "channel; the noisy Cy5/red channel usually needs a higher k.")

# ============================================================ Colocalization
with tabs[4]:
    with st.spinner("Detecting & colocalizing…"):
        spots = detect_all_spots()
    events, counts = ep.colocalize(spots["nucleosome"], spots["R_PTM"], spots["B_PTM"], "", int(scene), 0)
    nnuc = max(1, counts["n_nucleosomes"])
    m = st.columns(4)
    m[0].metric("R coloc / nucleosome", f'{100*counts["n_R_colocalized_with_nucleosome"]/nnuc:.2f}%',
                help=f'{counts["n_R_colocalized_with_nucleosome"]} events')
    m[1].metric("B coloc / nucleosome", f'{100*counts["n_B_colocalized_with_nucleosome"]/nnuc:.2f}%',
                help=f'{counts["n_B_colocalized_with_nucleosome"]} events')
    m[2].metric("Triple / nucleosome", f'{100*counts["n_R_B_nucleosome_triple_colocalized"]/nnuc:.2f}%',
                help=f'{counts["n_R_B_nucleosome_triple_colocalized"]} events')
    m[3].metric("R–B pairs (no nucleosome req.)", f'{counts["n_R_B_colocalized"]}',
                help='Unique R–B PTM pairs within the colocalization radius, independent of green')
    fig, ax = plt.subplots(figsize=(8, 8))
    if green is not None:
        ax.imshow(ep._norm01(green), cmap="gray")
    for role in ROLES:
        d = spots[role]
        if len(d):
            ax.scatter(d["registered_x"], d["registered_y"], s=9, c=ROLE_COLOR[role],
                       alpha=0.4, label=ROLE_LABEL[role])
    for etype, label, color, size in [
            ("R_nucleosome", "R–nucleosome (double)", "orange", 70),
            ("B_nucleosome", "B–nucleosome (double)", "magenta", 130),
            ("R_B_nucleosome_triple", "triple (R+B+nuc)", "yellow", 220)]:
        e = events[events["event_type"] == etype] if len(events) else events
        if len(e):
            ax.scatter(e["nucleosome_x"], e["nucleosome_y"], s=size, facecolors="none",
                       edgecolors=color, linewidths=1.6, label=f"{label} — {len(e)}")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.6)
    ax.set_title("Colocalization (registered) — doubles & triples", fontsize=10)
    ax.axis("off")
    st.pyplot(fig)

# ============================================================ Batch QA
with tabs[5]:
    st.write("Detect on a sample of FOVs in this lane/cycle to catch flooded / empty / saturated "
             "fields **before** a full run.")
    n_fov = st.slider("Number of FOVs to sample", 3, min(40, len(positions)), min(12, len(positions)))
    if st.button("Run batch QA"):
        scenes_qa = np.linspace(0, len(positions) - 1, n_fov).astype(int)
        rows = []
        prog = st.progress(0.0)
        for i, sc in enumerate(scenes_qa):
            pl = load_planes(run_dir, lane, int(cycle), int(sc), cmap_items)
            bd = {r: ep.detect_beads(pl[r], "", int(sc), 0, r) for r in ROLES}
            conf = ep.confirmed_bead_coords([bd["nucleosome"], bd["R_PTM"], bd["B_PTM"]])
            row = {"FOV": int(sc), "pos": positions[int(sc)], "confirmed_beads": len(conf),
                   "green_bead_raw": len(bd["nucleosome"])}
            for r in ROLES:
                sp = ep.detect_spots(pl[r], "", int(sc), 0, r, identity_tf(), bead_yx=conf)
                if ep.EXCLUDE_BEADS_FROM_SPOTS and len(conf):
                    sp, _ = ep.exclude_spots_near_beads(sp, conf)
                row[r] = len(sp)
            rows.append(row)
            prog.progress((i + 1) / len(scenes_qa))
        st.session_state["qa_df"] = pd.DataFrame(rows)
    if "qa_df" in st.session_state:
        qa = st.session_state["qa_df"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for ax, role in zip(axes, ROLES):
            ax.boxplot(qa[role], vert=True)
            ax.scatter(np.ones(len(qa)) + np.random.uniform(-0.05, 0.05, len(qa)),
                       qa[role], s=12, alpha=0.6, c=ROLE_COLOR[role])
            ax.set_title(f"{ROLE_LABEL[role]}\nmedian={int(qa[role].median())}"); ax.set_xticks([])
        st.pyplot(fig)
        flooded_fovs = qa[qa["green_bead_raw"] > ep.MAX_TRUSTED_BEADS]
        if len(flooded_fovs):
            st.warning(f"{len(flooded_fovs)} FOV(s) had a flooded green bead detector "
                       f"(handled by the confirmed-bead guard): FOVs {list(flooded_fovs['FOV'])}")
        st.dataframe(qa, use_container_width=True)
