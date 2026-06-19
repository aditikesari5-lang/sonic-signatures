"""
=============================================================
Q3A & Q3B – Sonic Signatures / Signals to Softwares
'Magical Mystery Tune' / 'Zapptain America'
=============================================================
HOW TO RUN LOCALLY:
  1. pip3 install streamlit numpy scipy matplotlib soundfile audioread --break-system-packages
  2. sudo apt install ffmpeg -y
  3. Put all your .mp3 song files in a folder called  songs/
  4. streamlit run sonic_signatures_app.py

HOW TO DEPLOY (Streamlit Cloud):
  - requirements.txt: numpy, scipy, matplotlib, soundfile, audioread, streamlit
  - packages.txt:     ffmpeg
=============================================================
"""

# ── Standard library ──────────────────────────────────────
import os          # file-path operations
import csv         # writing results.csv in batch mode
import pickle      # saving/loading the fingerprint database
import io          # in-memory byte streams (for uploaded files)
import tempfile    # create temporary files on disk for audioread

# ── Numerical / scientific ────────────────────────────────
import numpy as np                         # array maths
from scipy.ndimage import maximum_filter   # local-max peak detection
from scipy.signal import spectrogram as scipy_spectrogram  # STFT wrapper

# ── Plotting ──────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (required for Streamlit)
import matplotlib.pyplot as plt

# ── Audio I/O (no librosa, no pydub, no numba) ───────────
# soundfile : reads WAV/FLAC/OGG natively and very fast
# audioread : reads MP3 via ffmpeg – works on Python 3.12/3.13/3.14
import soundfile as sf    # pip install soundfile
import audioread          # pip install audioread  (needs ffmpeg on PATH)

# ── Web app ───────────────────────────────────────────────
import streamlit as st

# ─────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS  (tweak these to see Q3A experiments)
# ─────────────────────────────────────────────────────────

SAMPLE_RATE      = 22050   # target sample-rate (Hz) – all audio is resampled here
N_FFT            = 2048    # FFT window length in samples
HOP_LENGTH       = 512     # samples between consecutive windows
N_PEAKS          = 10      # how many constellation peaks to keep per time-frame
FAN_VALUE        = 5       # number of target-zone pairs per anchor peak
MIN_TIME_DELTA   = 1       # minimum time gap (frames) between anchor and target
MAX_TIME_DELTA   = 100     # maximum time gap (frames)
FREQ_RANGE       = 200     # frequency range (bins) around anchor for target search
DB_FILE          = "fingerprint_db.pkl"
SONGS_FOLDER     = "songs"

# ─────────────────────────────────────────────────────────
#  AUDIO LOADING  (soundfile for WAV, audioread for MP3)
# ─────────────────────────────────────────────────────────

def _resample(y, orig_sr, target_sr):
    """
    Simple linear resampling using numpy.
    Converts audio sampled at orig_sr Hz to target_sr Hz.
    We use linear interpolation: for each output sample position we
    compute where it maps to in the input array and interpolate between
    the two nearest input samples.
    """
    if orig_sr == target_sr:
        return y   # no work needed

    # Compute the ratio: how many input samples per output sample
    ratio        = orig_sr / target_sr
    # Total number of output samples after resampling
    n_out        = int(len(y) / ratio)
    # x-coordinates in the original signal for each output sample
    x_old        = np.linspace(0, len(y) - 1, n_out)
    # numpy interpolation: maps old positions → new values
    y_resampled  = np.interp(x_old, np.arange(len(y)), y)
    return y_resampled.astype(np.float32)


def load_audio(source):
    """
    Load audio from a file path (str) or raw bytes / BytesIO object.
    Returns a mono float32 numpy array normalised to [-1, 1] at SAMPLE_RATE Hz.

    Strategy:
      1. If source is a file path ending in .wav/.flac/.ogg → use soundfile
         (fast, pure Python, no external tools needed)
      2. Everything else (especially .mp3) → write to a temp file then use
         audioread, which calls ffmpeg under the hood to decode the audio.
         audioread returns raw 16-bit PCM chunks which we stitch together.
    """

    # ── Case 1: file path for WAV/FLAC/OGG ──────────────
    if isinstance(source, (str, os.PathLike)):
        ext = os.path.splitext(str(source))[1].lower()
        if ext in (".wav", ".flac", ".ogg"):
            y, sr = sf.read(str(source), dtype="float32", always_2d=False)
            # If stereo, average the two channels to get mono
            if y.ndim == 2:
                y = y.mean(axis=1)
            return _resample(y, sr, SAMPLE_RATE), SAMPLE_RATE

    # ── Case 2: MP3 file path ────────────────────────────
    if isinstance(source, (str, os.PathLike)):
        path_str = str(source)
        with audioread.audio_open(path_str) as f:
            sr        = f.samplerate    # original sample rate from the file
            n_ch      = f.channels      # number of channels (1=mono, 2=stereo)
            raw_chunks = []
            for block in f:             # f yields raw 16-bit PCM byte blocks
                raw_chunks.append(block)

        raw_bytes = b"".join(raw_chunks)
        # Convert bytes → int16 numpy array
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        # Normalise to [-1, 1]
        samples /= 32768.0
        # If stereo, reshape to (n_samples, n_channels) and average
        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)
        return _resample(samples, sr, SAMPLE_RATE), SAMPLE_RATE

    # ── Case 3: bytes or BytesIO (uploaded file) ─────────
    # audioread needs a real file on disk, so we write a temp file
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    # Read all bytes from the buffer
    source.seek(0)
    raw_data = source.read()

    # Write to a named temporary file with .mp3 extension so ffmpeg
    # knows the format; delete=False so we can close then re-open it
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(raw_data)
        tmp_path = tmp.name

    try:
        # Now load from the temp file path (falls into Case 2 above)
        y, sr = load_audio(tmp_path)
    finally:
        os.unlink(tmp_path)   # always clean up the temp file

    return y, sr


# ─────────────────────────────────────────────────────────
#  STEP 1 – SPECTROGRAM
# ─────────────────────────────────────────────────────────

def compute_spectrogram(y, sr):
    """
    Compute the magnitude spectrogram using Short-Time Fourier Transform (STFT).
    Returns S_db (freq × time in dB), freqs (Hz), times (s).
    """
    freqs, times, Sxx = scipy_spectrogram(
        y,
        fs=sr,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        scaling="spectrum",
    )
    S_db = 10 * np.log10(Sxx + 1e-10)
    return S_db, freqs, times


def plot_spectrogram(S_db, freqs, times, title="Spectrogram"):
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
    Find local maxima in the spectrogram (the constellation of peaks).
    Returns list of (freq_bin, time_frame) tuples.
    """
    neighbourhood_size = 20
    local_max = maximum_filter(S_db, size=neighbourhood_size) == S_db
    threshold = S_db.max() - 60
    strong    = S_db > threshold
    peak_mask = local_max & strong

    freq_indices, time_indices = np.where(peak_mask)
    peaks = []
    for t in np.unique(time_indices):
        mask_t   = time_indices == t
        f_at_t   = freq_indices[mask_t]
        strengths = S_db[f_at_t, t]
        top_idx  = np.argsort(strengths)[::-1][:n_peaks]
        for idx in top_idx:
            peaks.append((int(f_at_t[idx]), int(t)))
    return peaks


def plot_constellation(S_db, freqs, times, peaks, title="Constellation"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma", alpha=0.7)
    peak_times = [times[t] if t < len(times) else times[-1] for (f, t) in peaks]
    peak_freqs = [freqs[f] if f < len(freqs) else freqs[-1] for (f, t) in peaks]
    ax.scatter(peak_times, peak_freqs, s=15, c="cyan", marker="o",
               linewidths=0.5, label="Constellation peaks")
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
    Convert constellation into (hash, time_offset) pairs.
    hash = f1 * 10^10 + f2 * 10^5 + delta_t
    """
    hashes = []
    peaks_sorted = sorted(peaks, key=lambda x: x[1])

    for i, (f1, t1) in enumerate(peaks_sorted):
        count = 0
        for j in range(i + 1, len(peaks_sorted)):
            f2, t2   = peaks_sorted[j]
            delta_t  = t2 - t1
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

def build_database(songs_folder=SONGS_FOLDER, db_file=DB_FILE):
    """
    Index every .mp3/.wav file and save the fingerprint database to disk.
    On subsequent runs, loads from the .pkl file instantly.
    """
    if os.path.exists(db_file):
        with open(db_file, "rb") as f:
            db = pickle.load(f)
        return db

    db = {}
    all_files   = os.listdir(songs_folder)
    audio_files = [fn for fn in all_files
                   if fn.endswith(".mp3") or fn.endswith(".wav")]

    if not audio_files:
        st.error(f"No .mp3 or .wav files found in '{songs_folder}' folder.")
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
                if h not in db:
                    db[h] = []
                db[h].append((song_name, t))
        except Exception as e:
            st.warning(f"Skipped {filename}: {e}")

        progress.progress((idx + 1) / len(audio_files),
                          text=f"Indexed: {song_name}")

    with open(db_file, "wb") as f:
        pickle.dump(db, f)
    progress.empty()
    return db


# ─────────────────────────────────────────────────────────
#  STEP 5 – MATCHING  (offset histogram)
# ─────────────────────────────────────────────────────────

def match_query(query_y, query_sr, db):
    """
    Identify which song a query clip came from using offset histogram voting.
    Returns best_song, best_count, offset_dict, and intermediate visuals.
    """
    query_S_db, query_freqs, query_times = compute_spectrogram(query_y, query_sr)
    query_peaks  = extract_peaks(query_S_db)
    query_hashes = generate_hashes(query_peaks)

    offset_dict = {}
    for (h, qt) in query_hashes:
        if h not in db:
            continue
        for (song_name, db_t) in db[h]:
            offset = db_t - qt
            if song_name not in offset_dict:
                offset_dict[song_name] = {}
            offset_dict[song_name][offset] = \
                offset_dict[song_name].get(offset, 0) + 1

    if not offset_dict:
        return None, 0, {}, query_S_db, query_freqs, query_times, query_peaks

    best_song, best_count = None, 0
    for song, offsets in offset_dict.items():
        top_offset = max(offsets, key=offsets.get)
        count      = offsets[top_offset]
        if count > best_count:
            best_count = count
            best_song  = song

    return best_song, best_count, offset_dict, \
           query_S_db, query_freqs, query_times, query_peaks


def plot_offset_histogram(offset_dict, best_song, title="Offset Histogram"):
    fig, axes = plt.subplots(1, min(3, len(offset_dict)),
                             figsize=(12, 3), sharey=False)
    if len(offset_dict) == 1:
        axes = [axes]
    top_songs = sorted(offset_dict.items(),
                       key=lambda kv: max(kv[1].values()), reverse=True)[:3]
    for ax, (song, offsets) in zip(axes, top_songs):
        color = "crimson" if song == best_song else "steelblue"
        ax.bar(list(offsets.keys()), list(offsets.values()),
               width=1, color=color, alpha=0.8)
        ax.set_title(song[:25] + ("…" if len(song) > 25 else ""),
                     fontsize=8, color=color)
        ax.set_xlabel("Time offset (frames)")
        ax.set_ylabel("Hash count")
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
#  STEP 6 – BATCH MODE  (results.csv)
# ─────────────────────────────────────────────────────────

def run_batch_mode(query_files, db):
    rows = []
    for uploaded in query_files:
        raw_bytes  = uploaded.read()
        y, sr      = load_audio(raw_bytes)
        best_song, best_count, _, _, _, _, _ = match_query(y, sr, db)
        rows.append({"filename": uploaded.name,
                     "prediction": best_song if best_song else "unknown"})
    return rows


def write_results_csv(rows, path="results.csv"):
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
    st.markdown(
        "**EE200 Q3A & Q3B** – Audio fingerprinting via spectrogram peaks & "
        "hash matching. Upload a short clip and the app will identify the song."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        songs_path = st.text_input("Songs folder path", value=SONGS_FOLDER)
        if st.button("🔄 Re-index songs (rebuild DB)"):
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
                st.success("Old database removed. Will re-index now.")
        st.markdown("---")
        st.subheader("STFT parameters")
        st.caption(f"N_FFT={N_FFT} | Hop={HOP_LENGTH} | SR={SAMPLE_RATE}")

    if not os.path.isdir(songs_path):
        st.error(f"Folder `{songs_path}` not found. "
                 f"Place your .mp3 files there and refresh.")
        st.stop()

    with st.spinner("Loading fingerprint database…"):
        db = build_database(songs_folder=songs_path, db_file=DB_FILE)

    st.sidebar.success(f"DB loaded – {len(db):,} unique hashes.")

    tab_single, tab_batch = st.tabs(["🎤 Single Clip", "📂 Batch Mode"])

    # ── Single Clip ──────────────────────────────────────
    with tab_single:
        st.subheader("Upload a Query Clip")
        uploaded = st.file_uploader(
            "Upload a short .mp3 or .wav clip",
            type=["mp3", "wav"], key="single")

        if uploaded is not None:
            query_bytes = uploaded.read()
            y, sr = load_audio(query_bytes)
            st.audio(query_bytes, format="audio/mpeg")

            st.markdown("### 1 · Spectrogram")
            st.caption("Bright = loud. Each column is one DFT window sliding through time.")
            S_db, freqs, times = compute_spectrogram(y, sr)
            fig = plot_spectrogram(S_db, freqs, times, title=f"Query – {uploaded.name}")
            st.pyplot(fig); plt.close(fig)

            st.markdown("### 2 · Constellation of Peaks")
            st.caption("Only the loudest local maxima survive – sparse and noise-robust.")
            peaks = extract_peaks(S_db)
            fig = plot_constellation(S_db, freqs, times, peaks,
                                     title="Constellation Peaks (cyan dots)")
            st.pyplot(fig); plt.close(fig)

            st.markdown("### 3 · Matching")
            with st.spinner("Matching against database…"):
                best_song, best_count, offset_dict, _, _, _, _ = \
                    match_query(y, sr, db)

            if best_song:
                st.success(f"**Match found:** `{best_song}`  ({best_count} aligned hashes)")
            else:
                st.warning("No match found in the database.")

            if offset_dict:
                st.markdown("### 4 · Offset Histogram")
                st.caption("True match → one very tall bar. Wrong songs → scattered low bars.")
                fig = plot_offset_histogram(offset_dict, best_song)
                st.pyplot(fig); plt.close(fig)

    # ── Batch Mode ───────────────────────────────────────
    with tab_batch:
        st.subheader("Batch Identification")
        st.caption("Upload multiple clips → download results.csv")
        batch_files = st.file_uploader(
            "Upload query clips", type=["mp3", "wav"],
            accept_multiple_files=True, key="batch")

        if batch_files:
            if st.button("▶ Run Batch"):
                with st.spinner("Processing…"):
                    rows = run_batch_mode(batch_files, db)
                st.dataframe(rows, use_container_width=True)
                csv_path = write_results_csv(rows)
                with open(csv_path, "rb") as f:
                    csv_bytes = f.read()
                st.download_button("⬇ Download results.csv",
                                   data=csv_bytes,
                                   file_name="results.csv",
                                   mime="text/csv")


if __name__ == "__main__":
    main()
