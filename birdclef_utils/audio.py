import numpy as np
import librosa
import torch
import torchaudio.transforms as T

from .constants import (
    SR, N_SAMPLES, N_FFT, HOP_LENGTH, N_MELS, F_MIN, F_MAX, TOP_DB,
)

_mel_transform = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    power=2.0,
)
_amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=TOP_DB)


def load_audio(filepath, max_duration=30.0):
    """Load a mono waveform from disk.

    Parameters
    ----------
    filepath : str
        Path to the audio file (.ogg, .wav, etc.).
    max_duration : float or None
        Maximum seconds to read from the start of the file. None loads the
        whole file. Default 30.0 - long enough for window variability,
        short enough to avoid decoding multi-minute recordings in full.

    Returns
    -------
    (np.ndarray, int) or (None, None)
        A 1-D float32 waveform and its sample rate (always SR on success),
        or (None, None) if the file is missing, corrupt, or empty.
    """
    try:
        y, sr = librosa.load(
            filepath, sr=SR, mono=True, duration=max_duration
        )
    except Exception as e:
        print(f"load_audio failed for {filepath}: {e}")
        return None, None

    # librosa returns mono 1-D by default, but be defensive.
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)

    # Guard against empty / fully-corrupt decodes.
    if y.size == 0:
        print(f"load_audio: {filepath} produced an empty waveform")
        return None, None

    return y, sr


def normalize_waveform(y):
    """Peak-normalize a waveform to roughly [-1, 1].

    Safe on silence: a fully-zero waveform has peak 0, so it is returned
    unchanged rather than triggering a divide-by-zero.

    Parameters
    ----------
    y : np.ndarray
        1-D waveform.

    Returns
    -------
    np.ndarray
        Peak-normalized 1-D float32 waveform.
    """
    y = np.asarray(y, dtype=np.float32)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak
    return y


def fix_length(y, target_len=N_SAMPLES, crop_mode='start'):
    """Force a 1-D waveform to be EXACTLY `target_len` samples.

    Short waveforms are zero-padded at the end. Long waveforms are cropped
    to a `target_len` window. This guarantees every downstream spectrogram
    has the same shape, so batching never hits a size mismatch.

    Parameters
    ----------
    y : np.ndarray
        1-D waveform of any length (>= 1 sample).
    target_len : int
        Desired output length in samples. Default N_SAMPLES.
    crop_mode : {'start', 'center', 'random'}
        How to pick the window when `y` is longer than `target_len`:
          - 'start'  : first `target_len` samples (deterministic; inference)
          - 'center' : middle window (deterministic; validation)
          - 'random' : random window (augmentation; training)

    Returns
    -------
    np.ndarray
        1-D float32 waveform of length exactly `target_len`.
    """
    y = np.asarray(y, dtype=np.float32)
    n = len(y)

    # Exact length already - nothing to do.
    if n == target_len:
        return y

    # Too short - pad with zeros at the end.
    if n < target_len:
        return np.pad(y, (0, target_len - n), mode='constant')

    # Too long - crop a window of `target_len` samples.
    if crop_mode == 'start':
        start = 0
    elif crop_mode == 'center':
        start = (n - target_len) // 2
    elif crop_mode == 'random':
        start = np.random.randint(0, n - target_len + 1)
    else:
        raise ValueError(
            f"Unknown crop_mode {crop_mode!r}; "
            f"expected 'start', 'center', or 'random'."
        )

    return y[start:start + target_len]


def audio_to_melspec_torch(waveform):
    """Convert a 1-D waveform to a normalized log-mel spectrogram tensor.

    Uses torchaudio.transforms.MelSpectrogram. Equivalent to audio_to_melspec
    but returns a float32 torch.Tensor of shape (1, N_MELS, time_frames) so
    the dataset can use it directly without a numpy round-trip.

    Parameters
    ----------
    waveform : np.ndarray
        1-D mono float32 waveform. Must not be None.

    Returns
    -------
    torch.Tensor
        Shape (1, N_MELS, time_frames), dtype float32, values in [0, 1].
    """
    if waveform is None:
        raise ValueError(
            "audio_to_melspec_torch received None - check the return value of "
            "load_audio before calling this."
        )

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(
            f"audio_to_melspec_torch expects a 1-D waveform, got shape "
            f"{waveform.shape}."
        )

    wav_t = torch.from_numpy(waveform).unsqueeze(0)
    mel = _mel_transform(wav_t)
    log_mel = _amplitude_to_db(mel)

    # Shift so the loudest bin is 0 dB, matching librosa's ref=np.max.
    log_mel = log_mel - log_mel.max()

    log_mel = (log_mel + TOP_DB) / TOP_DB             # normalize to [0, 1]
    log_mel = torch.nan_to_num(log_mel, nan=0.0, posinf=1.0, neginf=0.0)
    return log_mel.clamp(0.0, 1.0)                    # (1, N_MELS, time_frames)


def audio_to_melspec(waveform):
    """Convert a 1-D waveform to a normalized log-mel spectrogram.

    Output is normalized to [0, 1] and guaranteed free of NaN/Inf, so it is
    safe to feed straight into a model. A fully-silent input produces a
    valid (all-zero-ish) spectrogram rather than NaN.

    Parameters
    ----------
    waveform : np.ndarray
        1-D mono waveform. Must not be None.

    Returns
    -------
    np.ndarray
        Log-mel spectrogram of shape (N_MELS, time_frames), dtype float32,
        values in [0, 1].

    Raises
    ------
    ValueError
        If `waveform` is None or not 1-D. (A None here almost always means
        a load_audio failure was not checked by the caller.)
    """
    if waveform is None:
        raise ValueError(
            "audio_to_melspec received None - check the return value of "
            "load_audio before calling this."
        )

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(
            f"audio_to_melspec expects a 1-D waveform, got shape "
            f"{waveform.shape}."
        )

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
    )

    # Convert power -> dB. For a silent clip (mel all zeros) power_to_db
    # with ref=np.max can yield non-finite values, handled just below.
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Map from roughly [-TOP_DB, 0] into [0, 1].
    log_mel = (log_mel + TOP_DB) / TOP_DB

    # np.clip alone does NOT remove NaN - replace non-finite values first,
    # then clip into the valid range.
    log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=1.0, neginf=0.0)
    log_mel = np.clip(log_mel, 0.0, 1.0)

    return log_mel.astype(np.float32)