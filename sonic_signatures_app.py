"""
=============================================================
Q3A & Q3B – Sonic Signatures / Signals to Softwares
EE200: Audio Fingerprinting
Works on Python 3.12 / 3.13 / 3.14
NO pydub, NO librosa, NO audioop, NO numba
Uses subprocess + ffmpeg to decode audio directly
=============================================================
"""

import os
import csv
import pickle
import io
import tempfile
import subprocess
import struct
import base64

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import spectrogram as scipy_spectrogram

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import streamlit as st

# ─────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────

SAMPLE_RATE    = 22050
N_FFT          = 2048
HOP_LENGTH     = 512
N_PEAKS        = 10
FAN_VALUE      = 5
MIN_TIME_DELTA = 1
MAX_TIME_DELTA = 100
FREQ_RANGE     = 200
DB_FILE        = "fingerprint_db.pkl"
META_FILE      = "fingerprint_meta.pkl"   # stores per-song metadata & thumbnail
SONGS_FOLDER   = "songs"

# ─────────────────────────────────────────────────────────
#  AUDIO LOADING via ffmpeg subprocess
# ─────────────────────────────────────────────────────────

def _ffmpeg_to_pcm(input_path):
    """
    Decode any audio file to raw 16-bit mono PCM at SAMPLE_RATE Hz
    using ffmpeg as a subprocess. No Python audio library needed.
    """
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", input_path,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "pipe:1"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {result.stderr.decode('utf-8', errors='ignore')}"
        )
    return result.stdout


def load_audio(source):
    """
    Load audio from a file path (str) or raw bytes / BytesIO.
    Returns (y, sr) where y is float32 in [-1, 1].
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    if hasattr(source, "read"):
        source.seek(0)
        raw_data = source.read()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(raw_data)
            tmp_path = tmp.name
        try:
            pcm_bytes = _ffmpeg_to_pcm(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        pcm_bytes = _ffmpeg_to_pcm(str(source))

    n_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
    y = np.array(samples, dtype=np.float32) / 32768.0
    return y, SAMPLE_RATE


# ─────────────────────────────────────────────────────────
#  STEP 1 – SPECTROGRAM
# ─────────────────────────────────────────────────────────

def compute_spectrogram(y, sr):
    """STFT: slides N_FFT window along signal, returns dB spectrogram."""
    freqs, times, Sxx = scipy_spectrogram(
        y, fs=sr,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        scaling="spectrum",
    )
    S_db = 10 * np.log10(Sxx + 1e-10)
    return S_db, freqs, times


def plot_spectrogram(S_db, freqs, times, title="Spectrogram"):
    """Return a matplotlib figure of the spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma")
    fig.colorbar(img, ax=ax, label="Power (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
#  STEP 2 – CONSTELLATION  (local peak picking)
# ─────────────────────────────────────────────────────────

def extract_peaks(S_db, n_peaks=N_PEAKS):
    """Find local maxima in spectrogram. Returns list of (freq_bin, time_frame)."""
    local_max = maximum_filter(S_db, size=20) == S_db
    strong    = S_db > S_db.max() - 60
    peak_mask = local_max & strong
    freq_indices, time_indices = np.where(peak_mask)
    peaks = []
    for t in np.unique(time_indices):
        mask_t    = time_indices == t
        f_at_t    = freq_indices[mask_t]
        strengths = S_db[f_at_t, t]
        top_idx   = np.argsort(strengths)[::-1][:n_peaks]
        for idx in top_idx:
            peaks.append((int(f_at_t[idx]), int(t)))
    return peaks


def plot_constellation(S_db, freqs, times, peaks, title="Constellation"):
    """Overlay cyan peak dots on spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma", alpha=0.7)
    pt = [times[t] if t < len(times) else times[-1] for (f, t) in peaks]
    pf = [freqs[f] if f < len(freqs) else freqs[-1] for (f, t) in peaks]
    ax.scatter(pt, pf, s=15, c="cyan", marker="o", linewidths=0.5,
               label=f"{len(peaks)} peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    return fig


def make_constellation_thumbnail(peaks, n_time_frames, n_freq_bins, size=(180, 120)):
    """
    Generate a small dark constellation image (scatter plot of peaks only,
    no axes) as a PNG bytes object — used for the Library grid cards.
    Each peak is coloured by its frequency (low=teal, high=yellow/pink),
    matching the style shown in the demo video.
    """
    fig, ax = plt.subplots(figsize=(size[0]/72, size[1]/72), dpi=72)
    fig.patch.set_facecolor("#0d1117")   # dark background
    ax.set_facecolor("#0d1117")

    if peaks:
        ts = np.array([t for (f, t) in peaks], dtype=float)
        fs = np.array([f for (f, t) in peaks], dtype=float)
        # Normalise for colouring
        fs_norm = fs / (n_freq_bins + 1e-6)
        # Use a colormap that goes teal → yellow → magenta (like the demo)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "demo", ["#00ffcc", "#ffe066", "#ff66cc"])
        colors = cmap(fs_norm)
        ax.scatter(ts, fs, s=1.5, c=colors, alpha=0.85, linewidths=0)

    ax.set_xlim(0, max(n_time_frames, 1))
    ax.set_ylim(0, max(n_freq_bins, 1))
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72,
                facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────
#  STEP 3 – FINGERPRINTING  (hash generation)
# ─────────────────────────────────────────────────────────

def generate_hashes(peaks, fan_value=FAN_VALUE):
    """
    Pair anchor peaks with nearby targets in a target zone.
    Hash = f1 * 10^10 + f2 * 10^5 + delta_t
    """
    hashes = []
    peaks_sorted = sorted(peaks, key=lambda x: x[1])
    for i, (f1, t1) in enumerate(peaks_sorted):
        count = 0
        for j in range(i + 1, len(peaks_sorted)):
            f2, t2  = peaks_sorted[j]
            delta_t = t2 - t1
            if delta_t < MIN_TIME_DELTA:
                continue
            if delta_t > MAX_TIME_DELTA:
                break
            if abs(f2 - f1) > FREQ_RANGE:
                continue
            h = int(f1) * 10**10 + int(f2) * 10**5 + int(delta_t)
            hashes.append((h, t1))
            count += 1
            if count >= fan_value:
                break
    return hashes


# ─────────────────────────────────────────────────────────
#  STEP 4 – DATABASE  (index & persist)
# ─────────────────────────────────────────────────────────

def build_database(songs_folder=SONGS_FOLDER, db_file=DB_FILE, meta_file=META_FILE):
    """
    Index every .mp3/.wav file. Saves two files:
      - fingerprint_db.pkl   : hash → [(song_name, anchor_time), ...]
      - fingerprint_meta.pkl : song_name → {hash_count, thumbnail_png_bytes,
                                            n_peaks, duration_s}
    """
    if os.path.exists(db_file) and os.path.exists(meta_file):
        with open(db_file, "rb") as f:
            db = pickle.load(f)
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
        return db, meta

    db   = {}
    meta = {}

    os.makedirs(songs_folder, exist_ok=True)
    audio_files = sorted([
        fn for fn in os.listdir(songs_folder)
        if fn.lower().endswith((".mp3", ".wav", ".flac", ".ogg"))
    ])

    if not audio_files:
        st.error(f"No audio files found in '{songs_folder}' folder.")
        return db, meta

    progress = st.progress(0, text="Indexing songs…")

    for idx, filename in enumerate(audio_files):
        song_name = os.path.splitext(filename)[0]
        path      = os.path.join(songs_folder, filename)
        try:
            y, sr  = load_audio(path)
            S_db, freqs, times = compute_spectrogram(y, sr)
            peaks  = extract_peaks(S_db)
            hashes = generate_hashes(peaks)

            # Store hashes in main DB
            for (h, t) in hashes:
                db.setdefault(h, []).append((song_name, t))

            # Generate thumbnail constellation image
            thumb_bytes = make_constellation_thumbnail(
                peaks,
                n_time_frames=S_db.shape[1],
                n_freq_bins=S_db.shape[0]
            )

            meta[song_name] = {
                "hash_count" : len(hashes),
                "n_peaks"    : len(peaks),
                "duration_s" : round(len(y) / sr, 1),
                "thumbnail"  : thumb_bytes,   # PNG bytes
            }

        except Exception as e:
            st.warning(f"Skipped {filename}: {e}")

        progress.progress((idx + 1) / len(audio_files),
                          text=f"Indexed: {song_name}")

    with open(db_file, "wb") as f:
        pickle.dump(db, f)
    with open(meta_file, "wb") as f:
        pickle.dump(meta, f)

    progress.empty()
    return db, meta


# ─────────────────────────────────────────────────────────
#  STEP 5 – MATCHING
# ─────────────────────────────────────────────────────────

def match_query(query_y, query_sr, db):
    """
    Offset histogram voting:
    For each query hash found in DB, record db_time − query_time.
    The true song gets a huge spike at one offset; wrong songs scatter.
    """
    S_db, freqs, times = compute_spectrogram(query_y, query_sr)
    peaks  = extract_peaks(S_db)
    hashes = generate_hashes(peaks)

    offset_dict = {}
    for (h, qt) in hashes:
        if h not in db:
            continue
        for (song_name, db_t) in db[h]:
            offset = db_t - qt
            offset_dict.setdefault(song_name, {})
            offset_dict[song_name][offset] = \
                offset_dict[song_name].get(offset, 0) + 1

    if not offset_dict:
        return None, 0, {}, S_db, freqs, times, peaks

    best_song, best_count = None, 0
    for song, offsets in offset_dict.items():
        top = max(offsets, key=offsets.get)
        if offsets[top] > best_count:
            best_count = offsets[top]
            best_song  = song

    return best_song, best_count, offset_dict, S_db, freqs, times, peaks


def plot_offset_histogram(offset_dict, best_song):
    """Bar chart — true match has one huge bar, others are scattered."""
    top_songs = sorted(offset_dict.items(),
                       key=lambda kv: max(kv[1].values()),
                       reverse=True)[:3]
    fig, axes = plt.subplots(1, len(top_songs), figsize=(12, 3))
    if len(top_songs) == 1:
        axes = [axes]
    for ax, (song, offsets) in zip(axes, top_songs):
        color = "crimson" if song == best_song else "steelblue"
        ax.bar(list(offsets.keys()), list(offsets.values()),
               width=1, color=color, alpha=0.8)
        ax.set_title(song[:25] + ("…" if len(song) > 25 else ""),
                     fontsize=8, color=color)
        ax.set_xlabel("Time offset (frames)")
        ax.set_ylabel("Hash count")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
#  STEP 6 – BATCH MODE
# ─────────────────────────────────────────────────────────

def run_batch_mode(query_files, db):
    """Match multiple uploaded files, return list of result dicts."""
    rows = []
    for uploaded in query_files:
        y, sr = load_audio(uploaded.read())
        best_song, *_ = match_query(y, sr, db)
        rows.append({"filename": uploaded.name,
                     "prediction": best_song if best_song else "unknown"})
    return rows


def write_results_csv(rows, path="results.csv"):
    """Write results.csv with columns: filename, prediction."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    return path


# ─────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="EE200: Audio Fingerprinting",
        page_icon="🎵",
        layout="wide"
    )

    # ── Custom CSS for dark card style ───────────────────
    st.markdown("""
    <style>
    .song-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 10px;
        text-align: center;
    }
    .song-card img {
        width: 100%;
        border-radius: 4px;
        display: block;
    }
    .song-title {
        color: #e6edf3;
        font-size: 13px;
        font-weight: 600;
        margin-top: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .song-hashes {
        color: #8b949e;
        font-size: 11px;
        margin-top: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────
    st.markdown("## 🎵 EE**200**: Audio Fingerprinting")
    st.caption("SIGNALS, SYSTEMS & NETWORKS · PROJECT DEMO")
    st.write("Index a library of songs as spectrogram fingerprints, "
             "then identify any short clip against it.")
    st.markdown("---")

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        songs_path = st.text_input("Songs folder", value=SONGS_FOLDER)
        if st.button("🔄 Re-index (rebuild DB)"):
            for f in [DB_FILE, META_FILE]:
                if os.path.exists(f):
                    os.remove(f)
            st.success("Database cleared. Reload the page to re-index.")
        st.markdown("---")
        st.caption(f"N_FFT={N_FFT} | Hop={HOP_LENGTH} | SR={SAMPLE_RATE}")

    # ── Load / build database ────────────────────────────
    db_exists   = os.path.exists(DB_FILE) and os.path.exists(META_FILE)
    dir_exists  = os.path.isdir(songs_path)

    if not db_exists and not dir_exists:
        st.error(f"Songs folder `{songs_path}` not found and no database exists.")
        st.stop()

    if not dir_exists:
        os.makedirs(songs_path, exist_ok=True)

    with st.spinner("Loading fingerprint database…"):
        db, meta = build_database(songs_folder=songs_path,
                                  db_file=DB_FILE,
                                  meta_file=META_FILE)

    st.sidebar.success(f"DB loaded – {len(db):,} unique hashes, "
                       f"{len(meta)} songs.")

    # ── Three tabs matching demo video ───────────────────
    tab_lib, tab_id, tab_batch = st.tabs(
        ["♦ LIBRARY", "⊙ IDENTIFY", "▦ BATCH"]
    )

    # ════════════════════════════════════════════════════
    #  TAB 1 – LIBRARY  (constellation thumbnails grid)
    # ════════════════════════════════════════════════════
    with tab_lib:
        st.subheader("LIBRARY")
        st.caption("Song indexing is managed by the admin. "
                   "Drop a clip in the Identify tab to test the library.")
        st.markdown("---")
        st.markdown("**IN THE DATABASE**")

        if not meta:
            st.info("No songs indexed yet. Add .mp3 files to the songs folder.")
        else:
            # Display songs in a 4-column grid with constellation thumbnails
            songs_list = sorted(meta.keys())
            cols_per_row = 4

            for row_start in range(0, len(songs_list), cols_per_row):
                row_songs = songs_list[row_start: row_start + cols_per_row]
                cols = st.columns(cols_per_row)

                for col, song_name in zip(cols, row_songs):
                    info = meta[song_name]
                    thumb_bytes = info.get("thumbnail", None)
                    hash_count  = info.get("hash_count", 0)

                    with col:
                        if thumb_bytes:
                            # Convert PNG bytes → base64 for st.markdown img
                            b64 = base64.b64encode(thumb_bytes).decode()
                            st.markdown(
                                f"""
                                <div class="song-card">
                                  <img src="data:image/png;base64,{b64}"/>
                                  <div class="song-title"
                                       title="{song_name}">{song_name}</div>
                                  <div class="song-hashes">
                                    {hash_count:,} hashes
                                  </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="song-card">
                                  <div style="height:80px;background:#21262d;
                                    border-radius:4px;display:flex;
                                    align-items:center;justify-content:center;
                                    color:#484f58;font-size:24px;">♪</div>
                                  <div class="song-title">{song_name}</div>
                                  <div class="song-hashes">
                                    {hash_count:,} hashes
                                  </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

    # ════════════════════════════════════════════════════
    #  TAB 2 – IDENTIFY  (single clip)
    # ════════════════════════════════════════════════════
    with tab_id:
        st.subheader("SEARCH")
        st.markdown("### Identify a clip")

        uploaded = st.file_uploader(
            "Upload a short clip",
            type=["mp3", "wav", "flac", "ogg"],
            key="single",
            label_visibility="collapsed"
        )

        if uploaded:
            query_bytes = uploaded.read()
            st.audio(query_bytes)

            with st.spinner("Analysing…"):
                y, sr = load_audio(query_bytes)
                best, count, offsets, S_db, freqs, times, peaks = \
                    match_query(y, sr, db)

            # ── Intermediate steps ───────────────────────
            st.markdown("#### Intermediate steps")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**1 · Spectrogram**")
                st.caption("Each column = one DFT window. "
                           "Bright = loud. Timing is preserved.")
                fig = plot_spectrogram(
                    S_db, freqs, times,
                    title=f"Spectrogram – {uploaded.name}")
                st.pyplot(fig); plt.close(fig)

            with col2:
                st.markdown("**2 · Constellation of Peaks**")
                st.caption("Sparse set of loud local maxima — "
                           "robust to background noise.")
                fig = plot_constellation(
                    S_db, freqs, times, peaks,
                    title="Constellation (cyan dots)")
                st.pyplot(fig); plt.close(fig)

            # ── Offset histogram ─────────────────────────
            if offsets:
                st.markdown("**3 · Offset Histogram**")
                st.caption("True match → one very tall bar. "
                           "Wrong songs → scattered low bars.")
                fig = plot_offset_histogram(offsets, best)
                st.pyplot(fig); plt.close(fig)

            # ── Result ───────────────────────────────────
            st.markdown("---")
            if best:
                st.success(f"**MATCH FOUND**")
                st.markdown(f"## 🎵 {best}")
                st.metric("Aligned hashes", count)

                # Show the library thumbnail of the matched song
                if best in meta and meta[best].get("thumbnail"):
                    b64 = base64.b64encode(
                        meta[best]["thumbnail"]).decode()
                    st.markdown(
                        f'<img src="data:image/png;base64,{b64}" '
                        f'style="width:300px;border-radius:8px;'
                        f'border:1px solid #30363d"/>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No match found in the database.")

    # ════════════════════════════════════════════════════
    #  TAB 3 – BATCH MODE
    # ════════════════════════════════════════════════════
    with tab_batch:
        st.subheader("BATCH")
        st.markdown("### Identify many clips at once")
        st.markdown(
            "Upload a set of query clips. Each is identified against the "
            "**currently indexed library**, and the results are written to a "
            "standardised `results.csv` with columns `filename`, `prediction`. "
            "The `prediction` is the matched track's filename without its "
            "extension, or `unknown` when no candidate clears the confidence "
            "threshold."
        )

        batch_files = st.file_uploader(
            "Upload clips",
            type=["mp3", "wav", "flac", "ogg"],
            accept_multiple_files=True,
            key="batch",
            label_visibility="collapsed"
        )

        if batch_files and st.button("▶ Run batch"):
            with st.spinner("Processing…"):
                rows = run_batch_mode(batch_files, db)
            st.dataframe(rows, use_container_width=True)
            csv_path = write_results_csv(rows)
            with open(csv_path, "rb") as f:
                st.download_button(
                    "⬇ Download results.csv",
                    data=f.read(),
                    file_name="results.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()
