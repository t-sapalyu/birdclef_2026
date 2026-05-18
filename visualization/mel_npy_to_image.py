import os
import numpy as np
import matplotlib.pyplot as plt

def npy_to_image(npy_path, save_path=None, show_plot=False):
    """
    Loads a .npy file containing a 2D Mel spectrogram and conditionally saves/displays it.
    
    Parameters:
    - npy_path (str): Path to the .npy file.
    - save_path (str, optional): Path to save the resulting image. If None, it won't save.
    - show_plot (bool): If True, displays the plot on screen using plt.show(). Default is False.
    """
    mel_spectrogram = np.load(npy_path)
    
    if mel_spectrogram.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {mel_spectrogram.shape}")

    # Saving the PURE IMAGE (No axes, no titles, no borders)
    if save_path:
        # plt.imsave writes the raw 2D array directly to an image file.
        # origin='lower' ensures the lowest frequencies (Mel band 0) are at the bottom.
        plt.imsave(save_path, mel_spectrogram, cmap='viridis', origin='lower')
        print(f"Pure image saved successfully to: {save_path}")

    # 3. Handle Displaying the FORMATTED PLOT on screen
    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spectrogram, origin='lower', aspect='auto', cmap='viridis')
        plt.title(f'Mel Spectrogram: {os.path.basename(npy_path)}')
        plt.ylabel('Mel Bands (0 to 127)')
        plt.xlabel('Time Frames')
        plt.colorbar(label='Normalized Amplitude [0, 1]')
        plt.tight_layout()
        plt.show()
        plt.close()

# --- Example Usage ---
# View a single file on screen:
# npy_to_image('path/to/your/audio_feature.npy')

# Save a file to your disk:
npy_to_image('./dataset/epi_spectrogram/spec_focal/22930__iNat317238.npy', save_path='visualization/output_image.png')