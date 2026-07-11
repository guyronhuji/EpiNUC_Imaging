"""EPINUC QA & threshold-tuning GUI (Streamlit).

A browser-based front end for `epinuc_colocalization.py`: inspect ND2 fields of view, check
bead detection / registration, tune per-channel spot-detection thresholds live, run quick
batch QA across FOVs, and save the tuned parameters to a JSON config that the pipeline
(`run_samples(..., config_path=...)`) can load for the production run.

Run:   streamlit run epinuc_gui.py
Needs: pip install streamlit   (plus the pipeline's own deps, incl. the nd2 reader)

The app imports the pipeline module and reuses its functions — no analysis logic is
duplicated here, so the module stays the single source of truth.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import epinuc_colocalization as ep

st.set_page_config(page_title="EPINUC QA & threshold tuning", layout="wide")
ROLES = ["nucleosome", "R_PTM", "B_PTM"]
ROLE_COLOR = {"nucleosome": "lime", "R_PTM": "red", "B_PTM": "deepskyblue"}
ROLE_LABEL = {"nucleosome": "Green / nucleosome", "R_PTM": "Red / R PTM", "B_PTM": "Blue / B PTM"}


# --------------------------------------------------------------------------- data helpers
@st.cache_data(show_spinner=False)
def file_info(path):
    """(channel_names, n_scenes) for an ND2 file."""
    with ep.ND2Accessor(path) as acc:
        return list(acc.channel_names), int(acc.n_scenes)


@st.cache_data(show_spinner="Reading FOV…")
def load_planes(path, scene, cmap_items):
    """Return {role: 2D image or None} for one FOV, resolving channels via ``cmap_items``."""
    cmap = dict(cmap_items)
    out = {}
    with ep.ND2Accessor(path) as acc:
        for role in ROLES:
            ci = acc.resolve_channel(cmap.get(role))
            out[role] = None if ci is None else acc.get_plane(scene=scene, channel_index=ci)
    return out


def fig_image(img, cmap="gray", title="", points=None, pcolor="yellow", psize=18, pmark="+"):
    """A matplotlib figure of one image with optional scatter overlay."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if img is not None:
        ax.imshow(ep._norm01(img), cmap=cmap)
    if points is not None and len(points):
        if pmark == "o":  # hollow circles for beads
            ax.scatter(points[:, 1], points[:, 0], s=psize, marker="o",
                       facecolors="none", edgecolors=pcolor, linewidths=1.0)
        else:             # filled markers (e.g. "+") for spots
            ax.scatter(points[:, 1], points[:, 0], s=psize, marker=pmark, c=pcolor, linewidths=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return fig


def identity_tf():
    return ep.ChannelTransform(method="reference", success=True)


# --------------------------------------------------------------------------- sidebar: data
st.sidebar.title("EPINUC QA")
data_dir = st.sidebar.text_input("Data folder", value=getattr(ep, "DATA_DIR", "./T50_20260225"))

if not os.path.isdir(data_dir):
    st.sidebar.error(f"Folder not found: {data_dir}")
    st.stop()
files = ep.list_nd2_files(data_dir)
if not files:
    st.sidebar.error("No .nd2 files found.")
    st.stop()
by_sample = ep.group_files_by_sample(files)
samples = sorted(k for k in by_sample if k is not None) or sorted(by_sample)

sample = st.sidebar.selectbox("Sample (lane)", samples, index=0)
sfiles = by_sample[sample]
tp_idx = st.sidebar.selectbox("Timepoint (file)", range(len(sfiles)),
                              format_func=lambda i: f"t{i}: {os.path.basename(sfiles[i])}")
path = sfiles[tp_idx]
chan_names, n_scenes = file_info(path)
scene = st.sidebar.number_input("FOV (scene)", 0, n_scenes - 1, 0, step=1)

# channel map (default from the module, overridable)
st.sidebar.markdown("**Channel → role**")
default_map = getattr(ep, "CHANNEL_MAP", {"nucleosome": None, "R_PTM": None, "B_PTM": None})
opts = chan_names

with ep.ND2Accessor(path) as _acc:
    _resolved = {r: _acc.resolve_channel(default_map.get(r)) for r in ROLES}
_unresolved = [r for r in ROLES if _resolved[r] is None]


def _default_idx(role):
    """Resolved index, else a DISTINCT positional fallback.

    Never fall back to 0 for every role: that silently maps all three roles onto the first channel
    (e.g. analysing the 640/red channel as "green"), which is how a differently-named ND2 produces
    nonsense beads and spots.
    """
    ci = _resolved[role]
    return ci if ci is not None else (ROLES.index(role) % max(1, len(chan_names)))


if _unresolved:
    st.sidebar.warning(
        f"CHANNEL_MAP did not resolve {_unresolved} against this file's channels {chan_names}. "
        f"Using positional defaults — **check the assignment below** (green/nucleosome is normally "
        f"the 561 / Cy3 channel).")
    st.sidebar.dataframe(ep.check_channel_map(path, default_map), use_container_width=True)

cmap = {}
for role in ROLES:
    cmap[role] = st.sidebar.selectbox(ROLE_LABEL[role], opts, index=_default_idx(role), key=f"cmap_{role}")
cmap_items = tuple(sorted(cmap.items()))

if len(set(cmap.values())) < len(ROLES):
    st.sidebar.error(
        "Two roles point at the same channel — each role needs its own. "
        "The nucleosome/green channel is normally the 561 / Cy3 one.")

# --------------------------------------------------------------------------- sidebar: params
st.sidebar.markdown("---")
st.sidebar.subheader("Detection parameters")

# seed slider state from module defaults once
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
        "bead_method": str(d.get("BEAD_DETECTION_METHOD", "multichannel")),
        "bead_snr": float(d.get("BEAD_DETECTION_SNR", 150.0)),
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
                         "background blur, removing the halo-ring artifact that seeds false "
                         "spots. Affects the background estimate only, not the peak-finding.")

st.sidebar.markdown("**Bead detector**")
st.sidebar.selectbox(
    "Method", ["fast", "log", "multichannel"], key="bead_method",
    help="'fast'/'log' threshold each channel on its own (Otsu). If the per-channel bead counts "
         "come out wildly different (e.g. 1 / 134 / 1), Otsu's bimodality assumption has failed — "
         "switch to 'multichannel', which keeps only peaks bright in EVERY channel (a fiducial "
         "by definition) and so rejects single molecules.")
if st.session_state["bead_method"] == "multichannel":
    st.sidebar.slider("Bead SNR (all channels)", 10.0, 1000.0, step=10.0, key="bead_snr",
                      help="A fiducial must reach this many robust-noise sigmas in EVERY channel "
                           "(on the cross-channel minimum). Beads land in the hundreds-thousands, "
                           "single molecules under ~100, so ~150 sits in the valley between them. "
                           "Raise if molecules leak in; lower if dim beads are missed.")

# apply current widget values to the module globals (drives all detection below)
ep.BEAD_DETECTION_METHOD = str(st.session_state["bead_method"])
ep.BEAD_DETECTION_SNR = float(st.session_state["bead_snr"])
ep.SPOT_DETECTION_SNR = {r: float(st.session_state[f"k_{r}"]) for r in ROLES}
ep.COLOCALIZATION_RADIUS_PX = float(st.session_state["coloc_r"])
ep.BEAD_EXCLUSION_RADIUS_PX = float(st.session_state["bead_excl_r"])
ep.MAX_TRUSTED_BEADS = int(st.session_state["max_beads"])
ep.EXCLUDE_BEADS_FROM_SPOTS = bool(st.session_state["excl_beads"])
ep.MASK_BEADS_BEFORE_BACKGROUND = bool(st.session_state["mask_beads_bg"])
# Keep the "beads" entry (the GUI only offers the three signal roles); dropping it would make a
# saved config silently lose a dedicated bead channel.
ep.CHANNEL_MAP = {**cmap, "beads": default_map.get("beads")}

# save / load / reset
st.sidebar.markdown("---")
cfg_path = st.sidebar.text_input("Config file", value="epinuc_config.json")
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("💾 Save"):
    # Keep the TIF path's laser assignment; this GUI has no widget for it and would write null.
    ep.save_config(cfg_path, preserve=("TIF_CHANNEL_MAP",))
    st.sidebar.success(f"Saved {cfg_path}")
if c2.button("📂 Load") and os.path.isfile(cfg_path):
    cfg = ep.load_config(cfg_path)
    snr = cfg.get("SPOT_DETECTION_SNR", {})
    for r in ROLES:
        if r in snr:
            st.session_state[f"k_{r}"] = float(snr[r])
    for key, ck in [("coloc_r", "COLOCALIZATION_RADIUS_PX"), ("bead_excl_r", "BEAD_EXCLUSION_RADIUS_PX"),
                    ("max_beads", "MAX_TRUSTED_BEADS"), ("excl_beads", "EXCLUDE_BEADS_FROM_SPOTS"),
                    ("mask_beads_bg", "MASK_BEADS_BEFORE_BACKGROUND"),
                    ("bead_method", "BEAD_DETECTION_METHOD"), ("bead_snr", "BEAD_DETECTION_SNR")]:
        if ck in cfg:
            st.session_state[key] = cfg[ck]
    # Re-seat the channel pickers too, else the rerun overwrites the loaded CHANNEL_MAP from the
    # stale widget values below. A saved map holds resolved names; skip anything this file lacks
    # (or an alias list) — _default_idx resolves those on the rerun anyway.
    for r, ch in (cfg.get("CHANNEL_MAP") or {}).items():
        if r in ROLES and ch in chan_names:
            st.session_state[f"cmap_{r}"] = ch
    st.rerun()
if c3.button("↺ Reset"):
    del st.session_state["seeded"]
    for r in ROLES:
        st.session_state.pop(f"cmap_{r}", None)
    st.rerun()

# --------------------------------------------------------------------------- load this FOV
planes = load_planes(path, int(scene), cmap_items)
green, red, blue = planes["nucleosome"], planes["R_PTM"], planes["B_PTM"]
imgs = {"nucleosome": green, "R_PTM": red, "B_PTM": blue}
st.title("EPINUC — QA & threshold tuning")
st.caption(f"Sample {sample} · {os.path.basename(path)} (t{tp_idx}) · FOV {scene} / {n_scenes - 1}")

tabs = st.tabs(["Channels", "Beads & QC", "Registration", "Spots & tuning",
                "Colocalization", "Batch QA"])

# ---- helper: beads for this FOV (current params)
def detect_all_beads():
    if ep.BEAD_DETECTION_METHOD == "multichannel":
        b = ep.detect_beads_multichannel(imgs, "", int(scene), 0)
    else:
        b = {r: ep.detect_beads(imgs[r], "", int(scene), 0, r) for r in ROLES}
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

# ---- helper: spots for this FOV (registered, current params), with bead exclusion
def detect_all_spots():
    beads, conf = detect_all_beads()
    tf = {"nucleosome": identity_tf()}
    tf["R_PTM"] = ep.estimate_channel_transform(beads["nucleosome"], beads["R_PTM"], green, red, ep.REGISTRATION_MODE) if red is not None else identity_tf()
    tf["B_PTM"] = ep.estimate_channel_transform(beads["nucleosome"], beads["B_PTM"], green, blue, ep.REGISTRATION_MODE) if blue is not None else identity_tf()
    spots = {r: ep.detect_spots(imgs[r], "", int(scene), 0, r, tf[r], bead_yx=conf) for r in ROLES}
    if ep.EXCLUDE_BEADS_FROM_SPOTS and len(conf):
        for r in ROLES:
            spots[r], _ = ep.exclude_spots_near_beads(spots[r], conf)
    return spots

# ============================================================ Spots & tuning
with tabs[3]:
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
        # count vs k sweep for the selected channel
        img = imgs[sel]
        if img is not None:
            _, conf = detect_all_beads()  # confirmed beads, so the sweep matches production
            ks = np.arange(3, 18.1, 1.0)
            saved = ep.SPOT_DETECTION_SNR[sel]
            counts = []
            for k in ks:
                ep.SPOT_DETECTION_SNR[sel] = float(k)
                counts.append(len(ep.detect_spots(img, "", int(scene), 0, sel, identity_tf(), bead_yx=conf)))
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
    m[3].metric("R–B pairs (no nucleosome req.)",
                f'{counts["n_R_B_colocalized"]}',
                help='Unique R–B PTM pairs within the colocalization radius, independent of '
                     'the nucleosome (green) channel')
    fig, ax = plt.subplots(figsize=(8, 8))
    if green is not None:
        ax.imshow(ep._norm01(green), cmap="gray")
    # underlying spots (faint dots)
    for role in ROLES:
        d = spots[role]
        if len(d):
            ax.scatter(d["registered_x"], d["registered_y"], s=9, c=ROLE_COLOR[role],
                       alpha=0.4, label=ROLE_LABEL[role])
    # colocalization events circled at the nucleosome position — doubles + triples, distinct
    # colors and concentric sizes (a triple nucleosome is also an R- and a B-double, so it
    # gets all three nested rings).
    circle_spec = [
        ("R_nucleosome", "R–nucleosome (double)", "orange", 70),
        ("B_nucleosome", "B–nucleosome (double)", "magenta", 130),
        ("R_B_nucleosome_triple", "triple (R+B+nuc)", "yellow", 220),
    ]
    for etype, label, color, size in circle_spec:
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
    st.write("Detect on a sample of FOVs at this timepoint to catch flooded / empty / saturated "
             "fields **before** a full run.")
    n_fov = st.slider("Number of FOVs to sample", 3, min(40, n_scenes), min(12, n_scenes))
    if st.button("Run batch QA"):
        scenes_qa = np.linspace(0, n_scenes - 1, n_fov).astype(int)
        rows = []
        prog = st.progress(0.0)
        for i, sc in enumerate(scenes_qa):
            pl = load_planes(path, int(sc), cmap_items)
            bd = {r: ep.detect_beads(pl[r], "", int(sc), 0, r) for r in ROLES}
            conf = ep.confirmed_bead_coords([bd["nucleosome"], bd["R_PTM"], bd["B_PTM"]])
            row = {"scene": int(sc), "confirmed_beads": len(conf),
                   "green_bead_raw": len(bd["nucleosome"])}
            for r in ROLES:
                sp = ep.detect_spots(pl[r], "", int(sc), 0, r, identity_tf(), bead_yx=conf)
                if ep.EXCLUDE_BEADS_FROM_SPOTS and len(conf):
                    sp, _ = ep.exclude_spots_near_beads(sp, conf)
                row[r] = len(sp)
            rows.append(row)
            prog.progress((i + 1) / len(scenes_qa))
        qa = pd.DataFrame(rows)
        st.session_state["qa_df"] = qa
    if "qa_df" in st.session_state:
        qa = st.session_state["qa_df"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for ax, role in zip(axes, ROLES):
            ax.boxplot(qa[role], vert=True); ax.scatter(np.ones(len(qa)) + np.random.uniform(-0.05, 0.05, len(qa)),
                                                        qa[role], s=12, alpha=0.6, c=ROLE_COLOR[role])
            ax.set_title(f"{ROLE_LABEL[role]}\nmedian={int(qa[role].median())}"); ax.set_xticks([])
        st.pyplot(fig)
        flooded_fovs = qa[qa["green_bead_raw"] > ep.MAX_TRUSTED_BEADS]
        if len(flooded_fovs):
            st.warning(f"{len(flooded_fovs)} FOV(s) had a flooded green bead detector "
                       f"(handled by the confirmed-bead guard): scenes {list(flooded_fovs['scene'])}")
        st.dataframe(qa, use_container_width=True)
