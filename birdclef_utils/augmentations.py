import torch
import torchaudio.transforms as T
import random
import numpy as np

class SpectrogramAugmenter:
    """
    Applies SpecAugment with a given probability, using multiple smaller masks
    to prevent destroying entire acoustic features.
    """
    def __init__(self, p=0.8, num_freq_masks=2, freq_mask_param=15, num_time_masks=2, time_mask_param=15):
        self.p = p 
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)

    def __call__(self, spec):
        if random.random() < self.p:
            for _ in range(self.num_freq_masks):
                spec = self.freq_mask(spec)
            for _ in range(self.num_time_masks):
                spec = self.time_mask(spec)
        return spec

def apply_mixup(batch_images, batch_labels, alpha=0.4):
    """
    Applies MixUp augmentation to a batch of spectrograms and their multi-hot labels.
    Alpha controls the blending intensity (0.4 is standard for bioacoustics).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = batch_images.size()[0]
    index = torch.randperm(batch_size)

    # Blend the images and labels
    mixed_images = lam * batch_images + (1 - lam) * batch_images[index, :]
    mixed_labels = lam * batch_labels + (1 - lam) * batch_labels[index, :]

    return mixed_images, mixed_labels

class WaveformAugmenter:
    """
    Applies 1D audio augmentations to the raw waveform before Mel Spectrogram conversion.
    - Gaussian Noise: Simulates environmental static, wind, or cheap microphones.
    - Random Gain: Simulates varying distances of the bird from the microphone.
    """
    def __init__(self, p=0.5, noise_level=0.015, gain_range=(0.5, 1.5)):
        self.p = p
        self.noise_level = noise_level
        self.gain_range = gain_range

    def __call__(self, y):
        """
        y: 1D NumPy array representing the audio wave (e.g., shape: [160000])
        """
        # 1. Inject Gaussian Noise (Static)
        if random.random() < self.p:
            # Generate random static based on a normal distribution
            noise = np.random.normal(0, self.noise_level, y.shape)
            y = y + noise # Mathematically add the noise to the bird call
            
        # 2. Apply Random Gain (Volume/Distance Scaling)
        if random.random() < self.p:
            # Pick a random volume multiplier between 0.5 (half as loud) and 1.5 (louder)
            gain = np.random.uniform(self.gain_range[0], self.gain_range[1])
            y = y * gain
            
        # 3. Audio Safety Catch (Crucial)
        y = np.clip(y, -1.0, 1.0)
        
        return y