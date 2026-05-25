"""
Augmentation pipeline for BirdCLEF+ 2026.

    1. MixUp          — batch-level audio mixing with element-wise max labels
    2. SpecAugment    — time and frequency masking on mel spectrograms
    3. BackgroundMixer — overlay real background audio (soundscapes / ESC-50)
    4. RandomFiltering — simplified random EQ to simulate channel distortions

"""

import os
import glob
import random

import numpy as np
import torch
import torchaudio.transforms as T
import random
import numpy as np
import librosa
from birdclef_utils.constants import SR

class SpectrogramAugmenter:
    """Apply SpecAugment with configurable masking.

    Parameters
    ----------
    p : float
        Probability of applying the full SpecAugment transform.
    num_freq_masks : int
        Number of frequency masks to apply.
    freq_mask_param : int
        Maximum width of each frequency mask (in mel bins).
    num_time_masks : int
        Number of time masks to apply.
    time_mask_param : int
        Maximum width of each time mask (in time frames).
    """

    def __init__(self, p=0.5, num_freq_masks=2, freq_mask_param=15,
                 num_time_masks=2, time_mask_param=25):
        self.p = p
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)

    def __call__(self, spec):
        """
        Parameters
        ----------
        spec : torch.Tensor
            Spectrogram of shape (1, N_MELS, T) or (N_MELS, T).

        Returns
        -------
        torch.Tensor
            Masked spectrogram (same shape).
        """
        if random.random() < self.p:
            for _ in range(self.num_freq_masks):
                spec = self.freq_mask(spec)
            for _ in range(self.num_time_masks):
                spec = self.time_mask(spec)
        return spec


# ---------------------------------------------------------------------------
# 3. Background Mixing  (waveform-level)
# ---------------------------------------------------------------------------

class BackgroundMixer:
    """Overlay real background audio onto the training waveform.

    This bridges the domain gap between clean focal recordings and the noisy
    soundscape environment the model is evaluated on.

    Parameters
    ----------
    bg_dirs : list[str]
        Directories containing background audio files (.ogg, .wav, .flac).
        The paper uses soundscape recordings and ESC-50 clips.
    p : float
        Probability of applying background mixing per sample.
    snr_range : tuple[float, float]
        Range of Signal-to-Noise Ratios (in dB) for mixing.  A lower SNR
        means louder background noise.  (3, 15) gives a good spread from
        heavy noise to light ambience.
    sample_rate : int
        Expected sample rate.  Files will be loaded at this rate.
    """

    def __init__(self, bg_dirs, p=0.5, snr_range=(3, 15), sample_rate=SR):
        self.p = p
        self.snr_range = snr_range
        self.sample_rate = sample_rate

        # Build a flat list of all background audio file paths at init time
        self.bg_files = []
        extensions = ('*.ogg', '*.wav', '*.flac', '*.mp3')
        for d in bg_dirs:
            for ext in extensions:
                self.bg_files.extend(glob.glob(os.path.join(d, '**', ext),
                                               recursive=True))

        if len(self.bg_files) == 0:
            print(f"BackgroundMixer WARNING: no audio files found in "
                  f"{bg_dirs}.  Background mixing will be skipped.")

    def _load_random_bg(self, target_len):
        """Load a random segment of background audio matching target_len."""
        if len(self.bg_files) == 0:
            return np.zeros(target_len, dtype=np.float32)

        # Try a few files in case one fails to load
        for _ in range(5):
            path = random.choice(self.bg_files)
            try:
                y, _ = librosa.load(path, sr=self.sample_rate, mono=True)
                y = np.asarray(y, dtype=np.float32)
                if y.size == 0:
                    continue

                # Pick a random segment of the required length
                if len(y) >= target_len:
                    start = random.randint(0, len(y) - target_len)
                    return y[start:start + target_len]
                else:
                    # Pad if background clip is shorter
                    return np.pad(y, (0, target_len - len(y)),
                                  mode='constant')
            except Exception:
                continue

        # Fallback: silence
        return np.zeros(target_len, dtype=np.float32)

    def __call__(self, y):
        """
        Parameters
        ----------
        y : np.ndarray
            1-D waveform (float32).

        Returns
        -------
        np.ndarray
            Waveform with background audio mixed in (same length).
        """
        if random.random() >= self.p or len(self.bg_files) == 0:
            return y

        bg = self._load_random_bg(len(y))

        # Compute desired SNR and scale background accordingly
        snr_db = random.uniform(*self.snr_range)

        # RMS of signal and background (avoid divide-by-zero)
        rms_signal = max(np.sqrt(np.mean(y ** 2)), 1e-8)
        rms_bg = max(np.sqrt(np.mean(bg ** 2)), 1e-8)

        # Scale background to achieve desired SNR:
        #   SNR_db = 20 * log10(rms_signal / rms_bg_scaled)
        #   rms_bg_scaled = rms_signal / 10^(SNR_db/20)
        target_rms = rms_signal / (10 ** (snr_db / 20))
        bg = bg * (target_rms / rms_bg)

        mixed = y + bg
        mixed = np.clip(mixed, -1.0, 1.0)
        return mixed.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Random Filtering  (waveform-level)
# ---------------------------------------------------------------------------

class RandomFilterAugmenter:
    """Simplified random equaliser to simulate channel distortions.

    Splits the spectrum into ``num_bands`` randomly-sized bands and applies
    a random gain (in dB) to each band.  This simulates the frequency
    response variations caused by different microphones, distances, and
    environmental reflections.

    Parameters
    ----------
    p : float
        Probability of applying the filter per sample.
    num_bands : int
        Number of frequency bands to split the spectrum into.
    db_range : tuple[float, float]
        Range of gain (in dB) applied to each band.  (-6, 6) gives
        moderate distortion; wider ranges give harsher effects.
    sample_rate : int
        Audio sample rate (needed for FFT frequency mapping).
    """

    def __init__(self, p=0.5, num_bands=3, db_range=(-6, 6),
                 sample_rate=SR):
        self.p = p
        self.num_bands = num_bands
        self.db_range = db_range
        self.sample_rate = sample_rate

    def __call__(self, y):
        """
        Parameters
        ----------
        y : np.ndarray
            1-D waveform (float32).

        Returns
        -------
        np.ndarray
            Frequency-modified waveform (same length).
        """
        if random.random() >= self.p:
            return y

        n = len(y)

        # FFT
        spectrum = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        n_bins = len(freqs)

        # Create random band boundaries (sorted)
        max_freq = self.sample_rate / 2
        boundaries = sorted(
            [0.0] +
            [random.uniform(0, max_freq) for _ in range(self.num_bands - 1)] +
            [max_freq]
        )

        # Apply random gain to each band
        for i in range(self.num_bands):
            lo = boundaries[i]
            hi = boundaries[i + 1]
            gain_db = random.uniform(*self.db_range)
            gain_linear = 10 ** (gain_db / 20)

            mask = (freqs >= lo) & (freqs < hi)
            spectrum[mask] *= gain_linear

        # Inverse FFT
        y_filtered = np.fft.irfft(spectrum, n=n)
        y_filtered = np.clip(y_filtered, -1.0, 1.0)
        return y_filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Waveform Augmenter  (supplementary — for ablation studies)
# ---------------------------------------------------------------------------

class WaveformAugmenter:
    """Gaussian noise injection and random gain for waveforms.

    Not part of the paper's baseline augmentations, but useful as an
    ablation variant ("SpecAugment + Noise Injection") and as a supplement
    to BackgroundMixer.

    Parameters
    ----------
    p : float
        Probability of applying each sub-augmentation independently.
    noise_level : float
        Standard deviation of additive Gaussian noise.
    gain_range : tuple[float, float]
        Range of multiplicative gain factors.
    """

    def __init__(self, p=0.5, noise_level=0.015, gain_range=(0.5, 1.5)):
        self.p = p
        self.noise_level = noise_level
        self.gain_range = gain_range

    def __call__(self, y):
        """
        Parameters
        ----------
        y : np.ndarray
            1-D waveform (float32).

        Returns
        -------
        np.ndarray
            Augmented waveform (same length).
        """
        # 1. Gaussian noise
        if random.random() < self.p:
            noise = np.random.normal(0, self.noise_level, y.shape)
            y = y + noise

        # 2. Random gain
        if random.random() < self.p:
            gain = np.random.uniform(self.gain_range[0], self.gain_range[1])
            y = y * gain

        y = np.clip(y, -1.0, 1.0)
        return y.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. AugmentationPipeline  (convenience chainer)
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """Chains multiple waveform-level augmentations into a single callable.

    Usage in the Dataset:
        pipeline = AugmentationPipeline(bg_mixer, rand_filter)
        dataset = FocalDataset(..., waveform_transform=pipeline)

    Parameters
    ----------
    *transforms : callable
        One or more waveform-level augmentation callables.  Each must accept
        and return a 1-D NumPy array.
    """

    def __init__(self, *transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, y):
        for t in self.transforms:
            y = t(y)
        return y


# ---------------------------------------------------------------------------
# 7. Ablation Configuration Helper
# ---------------------------------------------------------------------------

def get_augmentation_config(variation, bg_dirs=None):
    """Return augmentation objects for a named ablation variation.

    Parameters
    ----------
    variation : str
        One of:
          - ``'none'``          — no augmentation at all
          - ``'mixup'``         — MixUp only (batch-level)
          - ``'specaugment'``   — SpecAugment only (spec-level)
          - ``'random_filter'`` — RandomFiltering only (waveform-level)
          - ``'specaugment_noise'`` — SpecAugment + Gaussian noise injection
          - ``'full'``          — full combined pipeline (paper baseline)
    bg_dirs : list[str] or None
        Directories with background audio (required for ``'full'``).
        If None and variation is ``'full'``, BackgroundMixer is skipped
        with a warning.

    Returns
    -------
    dict
        Keys: ``'waveform_transform'``, ``'spec_transform'``,
        ``'use_mixup'`` (bool), ``'mixup_alpha'`` (float).
        Pass the first two to the Dataset and handle MixUp in the
        training loop based on the boolean flag.

    Example
    -------
    >>> cfg = get_augmentation_config('full', bg_dirs=['/path/to/soundscapes'])
    >>> dataset = FocalDataset(
    ...     ..., waveform_transform=cfg['waveform_transform'],
    ...     spec_transform=cfg['spec_transform'])
    >>> # In training loop:
    >>> if cfg['use_mixup']:
    ...     specs, labels = apply_mixup(specs, labels, cfg['mixup_alpha'])
    """

    config = {
        'waveform_transform': None,
        'spec_transform': None,
        'use_mixup': False,
        'mixup_alpha': 0.4,
    }

    variation = variation.lower().strip()

    if variation == 'none':
        pass  # all None

    elif variation == 'mixup':
        config['use_mixup'] = True

    elif variation == 'specaugment':
        config['spec_transform'] = SpectrogramAugmenter(p=0.5)

    elif variation == 'random_filter':
        config['waveform_transform'] = RandomFilterAugmenter(p=0.5)

    elif variation == 'specaugment_noise':
        config['waveform_transform'] = WaveformAugmenter(p=0.5)
        config['spec_transform'] = SpectrogramAugmenter(p=0.5)

    elif variation == 'full':
        # Build the full paper pipeline
        waveform_transforms = []

        # Background mixing (requires bg_dirs)
        if bg_dirs and len(bg_dirs) > 0:
            waveform_transforms.append(BackgroundMixer(bg_dirs, p=0.5))
        else:
            print("get_augmentation_config('full'): bg_dirs not provided, "
                  "skipping BackgroundMixer.")

        # Random filtering
        waveform_transforms.append(RandomFilterAugmenter(p=0.5))

        config['waveform_transform'] = AugmentationPipeline(
            *waveform_transforms
        )
        config['spec_transform'] = SpectrogramAugmenter(p=0.5)
        config['use_mixup'] = True

    else:
        raise ValueError(
            f"Unknown augmentation variation: {variation!r}. "
            f"Expected one of: 'none', 'mixup', 'specaugment', "
            f"'random_filter', 'specaugment_noise', 'full'."
        )

    return config