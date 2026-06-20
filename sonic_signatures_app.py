"""
=============================================================
Q3A & Q3B – Sonic Signatures / Signals to Softwares
Audio fingerprinting – works on Python 3.12 / 3.13 / 3.14
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

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import spectrogram as scipy_spectrogram

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

# ─────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────

SAMPLE_RATE    = 22050   # Hz – all audio resampled here
N_FFT          = 2048    # STFT window size in samples
HOP_LENGTH     = 512     # step between windows
N_PEAKS        = 10      # max constellation peaks per time frame
FAN_VALUE      = 5       # pairs per anchor peak
MIN_TIME_DELTA = 1       # min frame gap between anchor and target
MAX_TIME_DELTA = 100     # max frame gap (target zone)
FREQ_RANGE     = 200     # max freq-bin gap between anchor and target
DB_FILE        = "fingerprint_db.pkl"
SONGS_FOLDER   = "songs"

# ─────────────────────────────────────────────────────────
#  AUDIO LOADING via ffmpeg subprocess
#  Works on Python 3.14 – no audioop, no pydub needed
# ─────────────────────────────────────────────────────────

def _ffmpeg_to_pcm(input_path):
    """
    Call ffmpeg as a subprocess to decode any audio file to
    raw 16-bit signed PCM at SAMPLE_RATE Hz, mono channel.

    ffmpeg handles: mp3, wav, flac, ogg, m4a, aac, and more.
    Returns raw bytes of 16-bit PCM samples.
    """
    cmd = [
        "ffmpeg",
        "-v", "quiet",          # suppress ffmpeg console output
        "-i", input_path,       # input file path
        "-f", "s16le",          # output format: signed 16-bit little-endian PCM
        "-acodec", "pcm_s16le", # audio codec: raw PCM
        "-ar", str(SAMPLE_RATE),# resample to our target sample rate
        "-ac", "1",             # mix down to mono (1 channel)
        "pipe:1"                # send output to stdout (pipe) instead of a file
    ]
    # Run ffmpeg and capture its stdout (the raw PCM bytes)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {result.stderr.decode('utf-8', errors='ignore')}"
        )
    return result.stdout


def load_audio(source):
    """
    Load audio from a file path (str) or raw bytes / BytesIO.
    Returns (y, sr) where y is a float32 numpy array in [-1, 1].

    Strategy:
      - If source is bytes/BytesIO: write to a temp file first,
        because ffmpeg needs a real file path to read from.
      - Then call _ffmpeg_to_pcm() to get raw PCM bytes.
      - Convert int16 bytes → float32 array normalised to [-1, 1].
    """
    # ── Handle bytes / BytesIO (uploaded file from Streamlit) ──
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    if hasattr(source, "read"):
        # It's a file-like object; write it to a temp file on disk
        source.seek(0)
        raw_data = source.read()
        # Use .mp3 suffix so ffmpeg knows the format
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(raw_data)
            tmp_path = tmp.name
        try:
            pcm_bytes = _ffmpeg_to_pcm(tmp_path)
        finally:
            os.unlink(tmp_path)   # always delete the temp file
    else:
        # It's already a file path string
        pcm_bytes = _ffmpeg_to_pcm(str(source))

    # Convert raw 16-bit PCM bytes to numpy int16 array
    # Each sample is 2 bytes (little-endian signed 16-bit integer)
    n_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
    y = np.array(samples, dtype=np.float32)

    # Normalise to [-1.0, 1.0] — int16 range is [-32768, 32767]
    y /= 32768.0

    return y, SAMPLE_RATE


# ─────────────────────────────────────────────────────────
#  STEP 1 – SPECTROGRAM
#  Converts raw audio → 2-D time-frequency image
# ─────────────────────────────────────────────────────────

def compute_spectrogram(y, sr):
    """
    Short-Time Fourier Transform (STFT).
    Slides an N_FFT-sample window along the signal,
    takes the DFT of each slice, stacks them into a 2-D matrix.
    Returns S_db (freq × time in dB), freqs (Hz), times (s).
    """
    freqs, times, Sxx = scipy_spectrogram(
        y,
        fs=sr,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        scaling="spectrum",
    )
    # Convert power to decibels; +1e-10 avoids log(0)
    S_db = 10 * np.log10(Sxx + 1e-10)
    return S_db, freqs, times


def plot_spectrogram(S_db, freqs, times, title="Spectrogram"):
    """Plot the spectrogram as a colour map and return the figure."""
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
    """
    Find local maxima in the spectrogram (the constellation).
    A local max is a cell larger than all neighbours in a
    20-cell window.  We keep only the top n_peaks per time frame.
    Returns list of (freq_bin, time_frame) tuples.
    """
    local_max = maximum_filter(S_db, size=20) == S_db
    strong    = S_db > S_db.max() - 60      # at least –60 dB below peak
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
    """Overlay constellation peaks (cyan dots) on the spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma", alpha=0.7)
    pt = [times[t] if t < len(times) else times[-1] for (f, t) in peaks]
    pf = [freqs[f] if f < len(freqs) else freqs[-1] for (f, t) in peaks]
    ax.scatter(pt, pf, s=15, c="cyan", marker="o", linewidths=0.5,
               label="Constellation peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
#  STEP 3 – FINGERPRINTING  (hash generation)
# ─────────────────────────────────────────────────────────

def generate_hashes(peaks, fan_value=FAN_VALUE):
    """
    Convert constellation into (hash, anchor_time) pairs.
    Each hash encodes (f1, f2, delta_t) as a single integer:
        hash = f1 * 10^10 + f2 * 10^5 + delta_t
    Using pairs is much more specific than single peaks alone.
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
#  STEP 4 – DATABASE
# ─────────────────────────────────────────────────────────

def build_database(songs_folder=SONGS_FOLDER, db_file=DB_FILE):
    """
    Index every .mp3 / .wav file and save fingerprint database.
    Database format: { hash_int: [(song_name, anchor_time), ...] }
    Saved to disk as a pickle file so next run loads instantly.
    """
    # Load from disk if already built
    if os.path.exists(db_file):
        with open(db_file, "rb") as f:
            return pickle.load(f)

    db = {}
    os.makedirs(songs_folder, exist_ok=True)
    audio_files = [fn for fn in os.listdir(songs_folder)
                   if fn.lower().endswith((".mp3", ".wav", ".flac", ".ogg"))]

    if not audio_files:
        st.error(f"No audio files found in '{songs_folder}' folder.")
        return db

    progress = st.progress(0, text="Indexing songs…")
    for idx, filename in enumerate(audio_files):
        song_name = os.path.splitext(filename)[0]
        path      = os.path.join(songs_folder, filename)
        try:
            y, sr  = load_audio(path)
            S_db, freqs, times = compute_spectrogram(y, sr)
            peaks  = extract_peaks(S_db)
            hashes = generate_hashes(peaks)
            for (h, t) in hashes:
                db.setdefault(h, []).append((song_name, t))
        except Exception as e:
            st.warning(f"Skipped {filename}: {e}")
        progress.progress((idx + 1) / len(audio_files),
                          text=f"Indexed: {song_name}")

    with open(db_file, "wb") as f:
        pickle.dump(db, f)
    progress.empty()
    return db


# ─────────────────────────────────────────────────────────
#  STEP 5 – MATCHING
# ─────────────────────────────────────────────────────────

def match_query(query_y, query_sr, db):
    """
    Identify which song the query clip came from.
    For every query hash found in the database, vote for
    (song, db_time − query_time).  The true song gets a spike
    at one offset; wrong songs get only scattered votes.
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
    """
    Bar chart of offset votes for top 3 candidate songs.
    True match → one huge bar. Wrong song → scattered small bars.
    """
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
    """Run matching on multiple uploaded files, return list of dicts."""
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
    st.set_page_config(page_title="Sonic Signatures – EE200",
                       page_icon="🎵", layout="wide")
    st.title("🎵 Sonic Signatures")
    st.markdown("**EE200 Q3A & Q3B** – Audio fingerprinting via spectrogram "
                "peaks & hash matching.")

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        songs_path = st.text_input("Songs folder path", value=SONGS_FOLDER)
        if st.button("🔄 Re-index songs"):
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
                st.success("Database cleared. Will re-index now.")
        st.markdown("---")
        st.caption(f"N_FFT={N_FFT} | Hop={HOP_LENGTH} | SR={SAMPLE_RATE}")

    # ── Load / build database ────────────────────────────
    db_exists   = os.path.exists(DB_FILE)
    dir_exists  = os.path.isdir(songs_path)

    if not db_exists and not dir_exists:
        st.error(f"Folder `{songs_path}` not found and no pre-built database "
                 f"exists. Add your .mp3 files to the `{songs_path}/` folder.")
        st.stop()

    if not dir_exists:
        os.makedirs(songs_path, exist_ok=True)

    with st.spinner("Loading fingerprint database…"):
        db = build_database(songs_folder=songs_path, db_file=DB_FILE)

    st.sidebar.success(f"DB loaded – {len(db):,} unique hashes.")

    # ── Tabs ─────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🎤 Single Clip", "📂 Batch Mode"])

    # ════════════════════════════════════════════════════
    #  TAB 1 – SINGLE CLIP
    # ════════════════════════════════════════════════════
    with tab1:
        st.subheader("Upload a Query Clip")
        uploaded = st.file_uploader(
            "Upload a short .mp3 or .wav clip",
            type=["mp3", "wav", "flac", "ogg"],
            key="single")

        if uploaded:
            query_bytes = uploaded.read()
            st.audio(query_bytes)

            st.markdown("### 1 · Spectrogram")
            st.caption("Bright = loud. Each column is one DFT window. "
                       "A single DFT of the whole song loses all timing info.")
            y, sr = load_audio(query_bytes)
            S_db, freqs, times = compute_spectrogram(y, sr)
            fig = plot_spectrogram(S_db, freqs, times,
                                   title=f"Spectrogram – {uploaded.name}")
            st.pyplot(fig); plt.close(fig)

            st.markdown("### 2 · Constellation of Peaks")
            st.caption("Only loud local maxima survive – sparse and noise-robust.")
            peaks = extract_peaks(S_db)
            fig = plot_constellation(S_db, freqs, times, peaks,
                                     title="Constellation (cyan dots)")
            st.pyplot(fig); plt.close(fig)

            st.markdown("### 3 · Match Result")
            with st.spinner("Matching against database…"):
                best, count, offsets, *_ = match_query(y, sr, db)

            if best:
                st.success(f"**Match found:** `{best}`  ({count} aligned hashes)")
            else:
                st.warning("No match found.")

            if offsets:
                st.markdown("### 4 · Offset Histogram")
                st.caption("True match → one very tall bar. "
                           "Wrong songs → scattered low bars.")
                fig = plot_offset_histogram(offsets, best)
                st.pyplot(fig); plt.close(fig)

    # ════════════════════════════════════════════════════
    #  TAB 2 – BATCH MODE
    # ════════════════════════════════════════════════════
    with tab2:
        st.subheader("Batch Identification")
        batch_files = st.file_uploader(
            "Upload multiple clips",
            type=["mp3", "wav", "flac", "ogg"],
            accept_multiple_files=True,
            key="batch")

        if batch_files and st.button("▶ Run Batch"):
            with st.spinner("Processing…"):
                rows = run_batch_mode(batch_files, db)
            st.dataframe(rows, use_container_width=True)
            csv_path = write_results_csv(rows)
            with open(csv_path, "rb") as f:
                st.download_button("⬇ Download results.csv",
                                   data=f.read(),
                                   file_name="results.csv",
                                   mime="text/csv")


if __name__ == "__main__":
    main()
