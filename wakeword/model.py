"""A small wake-word detector: audio to a log-mel spectrogram to a tiny convolutional network.

It is kept small enough for a phone or edge device, and exports to TorchScript so it can run
without the training code.
"""
import torch
from torch import nn

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 3 // 2  # 1.5 s
N_FFT = 400
HOP = 160
N_MELS = 40


def mel_filterbank(device=None):
    """Triangular filters spaced evenly on the mel scale, which matches how hearing works."""
    def to_mel(hz):
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    freqs = torch.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1, device=device)
    points = to_hz(torch.linspace(
        to_mel(torch.tensor(20.0)), to_mel(torch.tensor(SAMPLE_RATE / 2)), N_MELS + 2, device=device,
    ))
    lower, center, upper = points[:-2].unsqueeze(1), points[1:-1].unsqueeze(1), points[2:].unsqueeze(1)
    rising = (freqs - lower) / (center - lower)
    falling = (upper - freqs) / (upper - center)
    return torch.clamp(torch.min(rising, falling), min=0.0)


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        # Stored on the module because TorchScript can't read module-level constants.
        self.n_fft = N_FFT
        self.hop = HOP
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.register_buffer("filters", mel_filterbank())

    def forward(self, audio):
        spectrum = torch.stft(audio, self.n_fft, self.hop, self.n_fft, self.window, return_complex=True)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        mel = torch.matmul(self.filters, power)
        log_mel = torch.log(mel + 1e-6)
        # Per-clip normalization, so microphone volume doesn't matter.
        mean = log_mel.mean(dim=(1, 2), keepdim=True)
        std = log_mel.std(dim=(1, 2), keepdim=True)
        return (log_mel - mean) / (std + 1e-6)


class WakeWordNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = LogMel()

        def block(channels_in, channels_out):
            return nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.net = nn.Sequential(block(1, 16), block(16, 32), block(32, 48), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 1))

    def forward(self, audio):
        """Takes audio of shape (batch, samples). Returns one logit per clip."""
        return self.net(self.features(audio).unsqueeze(1)).squeeze(1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
