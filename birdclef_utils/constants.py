# Audio
SR          = 32000   # competition standard; all data resampled to 32 kHz
DURATION    = 5       # seconds; matches the 5-second test chunk size
N_SAMPLES   = SR * DURATION   # 160000 samples per training clip

# Mel spectrogram
N_FFT       = 2048    # FFT window; 2048 favors frequency resolution
HOP_LENGTH  = 512     # hop between frames; gives ~313 frames for 5 s
N_MELS      = 128     # number of mel bands
F_MIN       = 20      # Hz; low bins are mostly noise — see Methods
F_MAX       = 16000   # Hz; Nyquist limit for 32 kHz audio
TOP_DB      = 80      # dynamic range for dB normalization

# Task
NUM_CLASSES = 234     # target species/sonotypes (taxonomy.csv)