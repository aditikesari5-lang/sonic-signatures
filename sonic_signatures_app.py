"""
=============================================================
Q3A & Q3B – Sonic Signatures / Signals to Softwares
'Magical Mystery Tune' / 'Zapptain America'
=============================================================
HOW TO RUN:
  1. pip3 install streamlit numpy scipy matplotlib soundfile pydub --break-system-packages
  2. sudo apt install ffmpeg -y          ← needed by pydub to decode .mp3
  3. Put all your .mp3 song files in a folder called  songs/
  4. streamlit run sonic_signatures_app.py
=============================================================
"""

# ── Standard library ──────────────────────────────────────
import os          # file-path operations
import csv         # writing results.csv in batch mode
import pickle      # saving/loading the fingerprint database
import io          # in-memory byte streams (for uploaded files)

# ── Numerical / scientific ────────────────────────────────
import numpy as np                         # array maths
from scipy.ndimage import maximum_filter   # local-max peak detection
from scipy.signal import spectrogram as scipy_spectrogram  # STFT wrapper

# ── Plotting ──────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (required for Streamlit)
import matplotlib.pyplot as plt

# ── Audio I/O  (NO librosa / NO numba) ───────────────────
# pydub: decodes .mp3 → raw PCM samples using ffmpeg under the hood
# soundfile: reads .wav / .flac / .ogg natively in Python
from pydub import AudioSegment   # pip3 install pydub  +  sudo apt install ffmpeg

# ── Web app ───────────────────────────────────────────────
import streamlit as st

# ─────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS  (tweak these to see Q3A experiments)
# ─────────────────────────────────────────────────────────

SAMPLE_RATE      = 22050   # target sample-rate (Hz) – all audio is resampled here
N_FFT            = 2048    # FFT window length in samples
                           #   shorter  → better time resolution, worse freq resolution
                           #   longer   → better freq resolution, worse time resolution
HOP_LENGTH       = 512     # samples between consecutive windows (overlap = N_FFT – HOP)
N_PEAKS          = 10      # how many constellation peaks to keep per time-frame
FAN_VALUE        = 5       # number of target-zone pairs per anchor peak
MIN_TIME_DELTA   = 1       # minimum time gap (frames) between anchor and target
MAX_TIME_DELTA   = 100     # maximum time gap (frames) – defines the target zone
FREQ_RANGE       = 200     # frequency range (bins) around anchor for target search
DB_FILE          = "fingerprint_db.pkl"   # where we persist the indexed database
SONGS_FOLDER     = "songs"                # folder containing .mp3 files

# ─────────────────────────────────────────────────────────
#  AUDIO LOADING  (replaces librosa.load — no numba needed)
# ─────────────────────────────────────────────────────────

def load_audio(source):
    """
    Load audio from a file path (str) or a bytes/BytesIO object.
    Returns a mono float32 numpy array normalised to [-1, 1] at SAMPLE_RATE Hz.

    How it works:
      - pydub.AudioSegment reads the file using ffmpeg.
        ffmpeg handles virtually every audio format: mp3, wav, flac, ogg, m4a …
      - We then:
          1. Convert to mono (average the channels)
          2. Resample to SAMPLE_RATE using pydub's built-in resampler
          3. Extract raw 16-bit PCM samples as a numpy array
          4. Normalise to float32 in [-1, 1] by dividing by 32768
    """
    if isinstance(source, (str, os.PathLike)):
        # source is a file path on disk
        seg = AudioSegment.from_file(str(source))
    else:
        # source is a bytes buffer or BytesIO (uploaded file)
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        seg = AudioSegment.from_file(source)

    # Convert stereo → mono by averaging channels
    seg = seg.set_channels(1)

    # Resample to our target sample rate
    seg = seg.set_frame_rate(SAMPLE_RATE)

    # Set sample width to 2 bytes (16-bit PCM)
    seg = seg.set_sample_width(2)

    # Extract raw PCM bytes and convert to numpy int16 array
    samples = np.frombuffer(seg.raw_data, dtype=np.int16)

    # Normalise to float32 in [-1.0, 1.0]
    y = samples.astype(np.float32) / 32768.0

    return y, SAMPLE_RATE


# ─────────────────────────────────────────────────────────
#  STEP 1 – SPECTROGRAM
#  Converts raw audio → 2-D time-frequency image
# ─────────────────────────────────────────────────────────

def compute_spectrogram(y, sr):
    """
    Compute the magnitude spectrogram using Short-Time Fourier Transform (STFT).

    The STFT slides a window of length N_FFT samples along the signal with
    a step of HOP_LENGTH samples, takes the DFT of each window, and stacks
    the magnitude columns into a 2-D matrix.

    Parameters
    ----------
    y  : 1-D numpy array of audio samples
    sr : sample rate (Hz) – needed only for returning the axes

    Returns
    -------
    S_db   : magnitude spectrogram in dB  (shape: freq_bins × time_frames)
    freqs  : frequency axis (Hz)
    times  : time axis (s)
    """
    # scipy.signal.spectrogram returns (freqs, times, Sxx)
    # Sxx contains the *power* spectrum; we take sqrt for magnitude
    freqs, times, Sxx = scipy_spectrogram(
        y,
        fs=sr,
        nperseg=N_FFT,                 # each slice is N_FFT samples wide
        noverlap=N_FFT - HOP_LENGTH,   # overlap between consecutive slices
        scaling="spectrum",            # raw power (not power-density)
    )

    # Convert power to decibels:  10·log10(power)  ← avoids log(0) with +1e-10
    S_db = 10 * np.log10(Sxx + 1e-10)
    return S_db, freqs, times


def plot_spectrogram(S_db, freqs, times, title="Spectrogram"):
    """
    Returns a Matplotlib figure of the spectrogram for display in Streamlit.
    Brighter colours = stronger frequencies at that moment in time.
    """
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
#  Keeps only the loudest peaks → robust to noise
# ─────────────────────────────────────────────────────────

def extract_peaks(S_db, n_peaks=N_PEAKS):
    """
    Find *local maxima* in the spectrogram (the 'constellation' of peaks).

    A local maximum is a cell that is strictly larger than all neighbours in a
    small surrounding window.  We then keep only the top-N_PEAKS loudest peaks
    in each time frame so the constellation stays sparse.

    Parameters
    ----------
    S_db   : magnitude spectrogram in dB
    n_peaks: max peaks to keep per time column

    Returns
    -------
    peaks  : list of (freq_bin, time_frame) tuples
    """
    # maximum_filter replaces each cell with the maximum of its neighbourhood.
    # A cell is a local max if it equals its own neighbourhood maximum.
    neighbourhood_size = 20        # window size (bins × frames)
    local_max = maximum_filter(S_db, size=neighbourhood_size) == S_db

    # Suppress very quiet peaks (below a floor of -60 dB relative to global max)
    threshold = S_db.max() - 60
    strong = S_db > threshold

    # Boolean mask: True only at genuine loud local maxima
    peak_mask = local_max & strong

    # Convert mask to list of (freq_bin, time_frame) coordinates
    freq_indices, time_indices = np.where(peak_mask)

    peaks = []
    # Group peaks by time frame and keep only the n_peaks loudest in each frame
    for t in np.unique(time_indices):
        # find all peaks at this time frame
        mask_t = time_indices == t
        f_at_t = freq_indices[mask_t]
        # sort by dB value descending and take top-N
        strengths = S_db[f_at_t, t]
        top_idx = np.argsort(strengths)[::-1][:n_peaks]
        for idx in top_idx:
            peaks.append((int(f_at_t[idx]), int(t)))   # (freq_bin, time_frame)

    return peaks


def plot_constellation(S_db, freqs, times, peaks, title="Constellation"):
    """
    Overlays the peak dots (circled in cyan) on the spectrogram.
    This is Figure 3 from the assignment brief.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma", alpha=0.7)

    # Map peak indices back to Hz / seconds for the plot axes
    peak_times = [times[t] if t < len(times) else times[-1] for (f, t) in peaks]
    peak_freqs = [freqs[f] if f < len(freqs) else freqs[-1] for (f, t) in peaks]

    ax.scatter(peak_times, peak_freqs,
               s=15, c="cyan", marker="o", linewidths=0.5,
               label="Constellation peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
#  STEP 3 – FINGERPRINTING  (hash generation)
#  Each hash encodes TWO nearby peaks + their time gap
# ─────────────────────────────────────────────────────────

def generate_hashes(peaks, fan_value=FAN_VALUE):
    """
    Convert the constellation into a list of (hash, time_offset) pairs.

    For every *anchor* peak, we look at the next `fan_value` peaks that fall
    inside a time-frequency 'target zone'.  Each (anchor, target) pair is
    hashed into a compact integer:

        hash = f1 * 10^10  +  f2 * 10^5  +  delta_t

    where f1, f2 are the two frequency bins and delta_t is their time gap.

    Using PAIRS (rather than a single peak) makes hashes far more specific:
    two random peaks at the same frequency by chance is rare, but two peaks
    at the same *pair* of frequencies with the *same gap* is extremely unlikely
    unless the audio content is actually the same.

    Parameters
    ----------
    peaks    : list of (freq_bin, time_frame) from extract_peaks()
    fan_value: how many targets per anchor

    Returns
    -------
    list of (hash_int, anchor_time_frame)
    """
    hashes = []
    peaks_sorted = sorted(peaks, key=lambda x: x[1])   # sort by time frame

    for i, (f1, t1) in enumerate(peaks_sorted):
        # Look at the next `fan_value` peaks that are within the target zone
        count = 0
        for j in range(i + 1, len(peaks_sorted)):
            f2, t2 = peaks_sorted[j]
            delta_t = t2 - t1

            # Enforce the target zone boundaries
            if delta_t < MIN_TIME_DELTA:
                continue
            if delta_t > MAX_TIME_DELTA:
                break   # peaks are sorted by time → no later peak will qualify

            # Only pair peaks that are also close in frequency
            if abs(f2 - f1) > FREQ_RANGE:
                continue

            # Build the hash: pack (f1, f2, delta_t) into one integer
            h = int(f1) * 10**10 + int(f2) * 10**5 + int(delta_t)
            hashes.append((h, t1))   # store hash alongside anchor time
            count += 1
            if count >= fan_value:
                break   # reached the fan limit for this anchor

    return hashes


# ─────────────────────────────────────────────────────────
#  STEP 4 – DATABASE  (index & persist)
# ─────────────────────────────────────────────────────────

def build_database(songs_folder=SONGS_FOLDER, db_file=DB_FILE):
    """
    Index every .mp3 (or .wav) file in `songs_folder` and save a fingerprint
    database.

    The database is a Python dict:
        { hash_int : [(song_name, anchor_time), ...] }

    One hash can appear in multiple songs (hash collision), so each key maps
    to a *list* of (song, time) tuples.  When matching we count how many
    hashes from the query align to the same time offset in a candidate song.

    Returns the loaded database dict.
    """
    if os.path.exists(db_file):
        # If we already indexed, load from disk (avoids re-processing every run)
        with open(db_file, "rb") as f:
            db = pickle.load(f)
        return db

    db = {}   # hash_int → [(song_name, anchor_time), ...]

    # Accept both .mp3 and .wav files
    all_files = os.listdir(songs_folder)
    audio_files = [fn for fn in all_files
                   if fn.endswith(".mp3") or fn.endswith(".wav")]

    if not audio_files:
        st.error(f"No .mp3 or .wav files found in '{songs_folder}' folder.")
        return db

    progress = st.progress(0, text="Indexing songs…")

    for idx, filename in enumerate(audio_files):
        song_name = os.path.splitext(filename)[0]   # strip .mp3 / .wav extension
        path = os.path.join(songs_folder, filename)

        try:
            # Load audio using pydub (no numba, no librosa)
            y, sr = load_audio(path)

            S_db, freqs, times = compute_spectrogram(y, sr)
            peaks  = extract_peaks(S_db)
            hashes = generate_hashes(peaks)

            # Insert each hash into the database
            for (h, t) in hashes:
                if h not in db:
                    db[h] = []
                db[h].append((song_name, t))

        except Exception as e:
            st.warning(f"Skipped {filename}: {e}")

        progress.progress((idx + 1) / len(audio_files),
                          text=f"Indexed: {song_name}")

    # Persist the database so future runs are instant
    with open(db_file, "wb") as f:
        pickle.dump(db, f)

    progress.empty()
    return db


# ─────────────────────────────────────────────────────────
#  STEP 5 – MATCHING  (offset histogram)
# ─────────────────────────────────────────────────────────

def match_query(query_y, query_sr, db):
    """
    Identify which song a query clip came from.

    Algorithm
    ---------
    1. Fingerprint the query clip exactly as we fingerprinted the database songs.
    2. For every query hash that exists in the database, record (candidate_song,
       db_anchor_time − query_anchor_time) as a "time offset vote".
    3. For a TRUE match, *all* the query hashes land at the SAME time offset
       inside the database song (the song started playing N frames before t=0
       of the query).  Counting votes per (song, offset) pair → one bin
       explodes with a large count while all others stay small.
    4. The song with the tallest offset-histogram bin wins.

    Returns
    -------
    best_song     : str  (or None if no match found)
    best_count    : int  (number of aligned hashes)
    offset_dict   : dict  { song_name : {offset : count} }  (for visualisation)
    query_S_db    : spectrogram of query (for display)
    query_freqs   : frequency axis
    query_times   : time axis
    query_peaks   : constellation peaks
    """
    query_S_db, query_freqs, query_times = compute_spectrogram(query_y, query_sr)
    query_peaks  = extract_peaks(query_S_db)
    query_hashes = generate_hashes(query_peaks)

    # offset_dict[song][delta_t] = count of matching hashes at that offset
    offset_dict = {}

    for (h, qt) in query_hashes:
        if h not in db:
            continue   # this hash didn't appear in any indexed song
        for (song_name, db_t) in db[h]:
            offset = db_t - qt   # time displacement between database and query
            if song_name not in offset_dict:
                offset_dict[song_name] = {}
            offset_dict[song_name][offset] = \
                offset_dict[song_name].get(offset, 0) + 1

    if not offset_dict:
        return None, 0, {}, query_S_db, query_freqs, query_times, query_peaks

    # Find the (song, offset) with the highest vote count
    best_song  = None
    best_count = 0
    for song, offsets in offset_dict.items():
        top_offset = max(offsets, key=offsets.get)
        count = offsets[top_offset]
        if count > best_count:
            best_count = count
            best_song  = song

    return best_song, best_count, offset_dict, \
           query_S_db, query_freqs, query_times, query_peaks


def plot_offset_histogram(offset_dict, best_song, title="Offset Histogram"):
    """
    Visualise the offset-histogram for the top-3 candidate songs.
    A true match shows one very tall bar (all hashes align at the same offset).
    A wrong song shows only scattered low bars (random coincidences).
    """
    fig, axes = plt.subplots(1, min(3, len(offset_dict)),
                             figsize=(12, 3), sharey=False)
    if len(offset_dict) == 1:
        axes = [axes]

    # Sort candidates by their peak count (descending) and take top 3
    top_songs = sorted(offset_dict.items(),
                       key=lambda kv: max(kv[1].values()), reverse=True)[:3]

    for ax, (song, offsets) in zip(axes, top_songs):
        xs = list(offsets.keys())
        ys = list(offsets.values())
        color = "crimson" if song == best_song else "steelblue"
        ax.bar(xs, ys, width=1, color=color, alpha=0.8)
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
    """
    Accept a list of uploaded query files, run matching on each,
    and return a list of {'filename': ..., 'prediction': ...} dicts.

    This is also used to write results.csv in exactly the format the
    assignment requires:
        filename,prediction
        clip01.mp3,song_name_without_extension
    """
    rows = []
    for uploaded in query_files:
        raw_bytes = uploaded.read()
        y, sr = load_audio(raw_bytes)
        best_song, best_count, _, _, _, _, _ = match_query(y, sr, db)

        filename   = uploaded.name
        prediction = best_song if best_song else "unknown"
        rows.append({"filename": filename, "prediction": prediction})

    return rows


def write_results_csv(rows, path="results.csv"):
    """Write the batch results to a CSV file with columns: filename, prediction."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    return path


# ─────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────

def main():
    # ── Page config ─────────────────────────────────────
    st.set_page_config(
        page_title="Sonic Signatures – EE200",
        page_icon="🎵",
        layout="wide",
    )

    # ── Title & description ──────────────────────────────
    st.title("🎵 Sonic Signatures")
    st.markdown(
        "**EE200 Q3A & Q3B** – Audio fingerprinting via spectrogram peaks & "
        "hash matching.  Upload a short clip and the app will tell you which "
        "song it came from."
    )

    # ── Sidebar: indexing controls ───────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        songs_path = st.text_input("Songs folder path", value=SONGS_FOLDER)
        rebuild = st.button("🔄 Re-index songs (rebuild DB)")
        if rebuild and os.path.exists(DB_FILE):
            os.remove(DB_FILE)   # force a fresh build
            st.success("Old database removed. Will re-index now.")

        st.markdown("---")
        st.subheader("STFT parameters")
        st.caption(f"N_FFT = {N_FFT}  |  Hop = {HOP_LENGTH}  |  SR = {SAMPLE_RATE}")
        st.caption("Edit constants at the top of the .py file to experiment.")

    # ── Load / build database ────────────────────────────
    if not os.path.isdir(songs_path):
        st.error(f"Folder `{songs_path}` not found.  "
                 f"Place your .mp3 files there and refresh.")
        st.stop()

    with st.spinner("Loading fingerprint database…"):
        db = build_database(songs_folder=songs_path, db_file=DB_FILE)

    st.sidebar.success(f"DB loaded – {len(db):,} unique hashes across all songs.")

    # ── Tabs: Single-clip mode  |  Batch mode ───────────
    tab_single, tab_batch = st.tabs(["🎤 Single Clip", "📂 Batch Mode"])

    # ════════════════════════════════════════════════════
    #  TAB 1 – SINGLE CLIP MODE
    # ════════════════════════════════════════════════════
    with tab_single:
        st.subheader("Upload a Query Clip")
        uploaded = st.file_uploader(
            "Upload a short .mp3 or .wav clip (a few seconds is enough)",
            type=["mp3", "wav"],
            key="single",
        )

        if uploaded is not None:
            # Load the clip
            query_bytes = uploaded.read()
            y, sr = load_audio(query_bytes)

            st.audio(query_bytes, format="audio/mpeg")

            # ── Spectrogram display ──────────────────────
            st.markdown("### 1 · Spectrogram")
            st.caption(
                "Each vertical slice is one DFT window.  "
                "Bright = loud.  "
                "A single Fourier transform (no sliding) would lose all timing info."
            )
            S_db, freqs, times = compute_spectrogram(y, sr)
            fig_spec = plot_spectrogram(S_db, freqs, times,
                                        title=f"Query – {uploaded.name}")
            st.pyplot(fig_spec)
            plt.close(fig_spec)

            # ── Constellation display ────────────────────
            st.markdown("### 2 · Constellation of Peaks")
            st.caption(
                "Only the loudest local maxima are kept.  "
                "This sparse set is robust to background noise."
            )
            peaks = extract_peaks(S_db)
            fig_const = plot_constellation(S_db, freqs, times, peaks,
                                           title="Constellation Peaks (cyan dots)")
            st.pyplot(fig_const)
            plt.close(fig_const)

            # ── Matching ─────────────────────────────────
            st.markdown("### 3 · Matching")
            with st.spinner("Matching against database…"):
                best_song, best_count, offset_dict, _, _, _, _ = \
                    match_query(y, sr, db)

            if best_song:
                st.success(f"**Match found:** `{best_song}`  "
                           f"({best_count} aligned hashes)")
            else:
                st.warning("No match found in the database.")

            # ── Offset histogram display ─────────────────
            if offset_dict:
                st.markdown("### 4 · Offset Histogram")
                st.caption(
                    "A true match produces a single very tall bar – all hashes "
                    "from the query line up at the *same* time offset in the "
                    "database song.  Wrong songs show only scattered low bars."
                )
                fig_hist = plot_offset_histogram(offset_dict, best_song)
                st.pyplot(fig_hist)
                plt.close(fig_hist)

    # ════════════════════════════════════════════════════
    #  TAB 2 – BATCH MODE
    # ════════════════════════════════════════════════════
    with tab_batch:
        st.subheader("Batch Identification")
        st.caption(
            "Upload multiple query clips.  The app will produce a `results.csv` "
            "with columns: `filename, prediction`."
        )
        batch_files = st.file_uploader(
            "Upload query .mp3 or .wav clips",
            type=["mp3", "wav"],
            accept_multiple_files=True,
            key="batch",
        )

        if batch_files:
            if st.button("▶ Run Batch"):
                with st.spinner("Processing…"):
                    rows = run_batch_mode(batch_files, db)

                # Display results table in Streamlit
                st.dataframe(rows, use_container_width=True)

                # Write results.csv and offer a download button
                csv_path = write_results_csv(rows)
                with open(csv_path, "rb") as f:
                    csv_bytes = f.read()
                st.download_button(
                    "⬇ Download results.csv",
                    data=csv_bytes,
                    file_name="results.csv",
                    mime="text/csv",
                )


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    main()
