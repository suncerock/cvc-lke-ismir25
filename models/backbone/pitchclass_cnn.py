from typing import List
import torch
import torch.nn as nn


class PitchClassCNNBasic(torch.nn.Module):
    """
    Basic CNN as in Johannes Zeitler's report, for HCQT input with 75 (-1) context frames
    and variable amount of pitch bins (num octaves * 12 * 3 bins per semitone)
    The number of input channels, channels in the hidden layers, and output
    dimensions (e.g. for pitch output) can be parameterized.
    Layer normalization is only performed over frequency and channel dimensions,
    not over time (in order to work with variable length input).

    Args (Defaults: BasicCNN by Johannes Zeitler but with 6 input channels):
        n_chan_input:   Number of input channels (harmonics in HCQT)
        n_chan_layers:  Number of channels in the hidden layers (list)
        n_bins_in:      Number of input bins (12 * number of octaves)
        n_bins_out:     Number of output bins (12 for pitch class, 72 for pitch)
        a_lrelu:        alpha parameter (slope) of LeakyReLU activation function
        p_dropout:      Dropout probability
    """
    def __init__(
        self,
        bins_per_octave: int = 36,
        num_octaves: int = 5,
        num_harmonics: int = 6,
        
        channel_p: int = 20,
        channel_b: int = 20,
        channel_t: int = 10,

        activation="GELU",
        dropout=0.2
    ):
        super(PitchClassCNNBasic, self).__init__()

        act_fn = nn.__dict__.get(activation, "GELU")

        bins_per_semitone = bins_per_octave // 12

        # Prefiltering
        self.conv_p = nn.Sequential(
            nn.Conv2d(num_harmonics, channel_p, kernel_size=15, stride=1, padding=7),
            act_fn(),
            # nn.MaxPool2d((1, 2), ceil_mode=True),
            nn.Dropout(dropout)
        )

        # Binning to MIDI pitches
        self.conv_b = nn.Sequential(
            nn.Conv2d(channel_p, channel_b, kernel_size=(bins_per_semitone, 3), padding=(0, 1), stride=(bins_per_semitone, 1)),
            act_fn(),
            nn.MaxPool2d((1, 3), stride=1, padding=(0, 1)),
            nn.Dropout(dropout)
        )

        # Time reduction
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channels=channel_b, out_channels=channel_t, kernel_size=(1, 7), padding=(0, 3), stride=(1, 1)),
            act_fn(),
            nn.Dropout(dropout)
        )

        # Chroma reduction
        self.conv_c = nn.Sequential(
            nn.Conv2d(channel_t, 1, 1),
            act_fn(),
            nn.Dropout(dropout),
        )

        self.output_dim = 12 * num_octaves  # Output dimension for pitch class (12 for each octave)

    def forward(self, x):
        # x: B * 6 * 180 * (T=1001)

        x = self.conv_p(x)  # B * C_P * 180 * (T // 2 + 1 = 501)
        x = self.conv_b(x)  # B * C_B * 60 * (T // 10 + 1 = 101)
        x = self.conv_t(x)  # B * C_T * 60 * (T // 10 + 1 = 101)
        x = self.conv_c(x)  # B * 1 * 60 * (T // 10 + 1 = 101)

        x = x.squeeze(1)  # B * 60 * (T // 10 + 1 = 101)
        x = x.transpose(1, 2)

        return x

class PitchClassCNNRes(torch.nn.Module):
    """
    Basic CNN as in Johannes Zeitler's report, for HCQT input with 75 (-1) context frames
    and variable amount of pitch bins (num octaves * 12 * 3 bins per semitone)
    The number of input channels, channels in the hidden layers, and output
    dimensions (e.g. for pitch output) can be parameterized.
    Layer normalization is only performed over frequency and channel dimensions,
    not over time (in order to work with variable length input).

    Args (Defaults: BasicCNN by Johannes Zeitler but with 6 input channels):
        n_chan_input:   Number of input channels (harmonics in HCQT)
        n_chan_layers:  Number of channels in the hidden layers (list)
        n_bins_in:      Number of input bins (12 * number of octaves)
        n_bins_out:     Number of output bins (12 for pitch class, 72 for pitch)
        a_lrelu:        alpha parameter (slope) of LeakyReLU activation function
        p_dropout:      Dropout probability
    """
    def __init__(
        self,
        bins_per_octave: int = 36,
        num_octaves: int = 5,
        num_harmonics: int = 6,
        
        n_prefilt_layers=5,
        channel_p: int = 20,
        channel_b: int = 20,
        channel_t: int = 10,

        activation="LeakyReLU",
        dropout=0.2
    ):
        super(PitchClassCNNRes, self).__init__()

        act_fn = nn.__dict__.get(activation, "GELU")

        bins_per_semitone = bins_per_octave // 12

        # Prefiltering
        self.conv_p = nn.ModuleList([nn.Sequential(
            nn.Conv2d(num_harmonics, channel_p, kernel_size=15, stride=1, padding=7),
            act_fn(),
            # nn.MaxPool2d((1, 2), ceil_mode=True),
            nn.Dropout(dropout)
        )])
        for _ in range(n_prefilt_layers - 1):
            self.conv_p.append(nn.Sequential(
                nn.Conv2d(channel_p, channel_p, kernel_size=(15, 7), stride=1, padding=(7, 3)),
                act_fn(),
                # nn.MaxPool2d((1, 3), stride=1, padding=(0, 1)),
                nn.Dropout(dropout)
            ))

        # Binning to MIDI pitches
        self.conv_b = nn.Sequential(
            nn.Conv2d(channel_p, channel_b, kernel_size=(bins_per_semitone, 3), padding=(0, 1), stride=(bins_per_semitone, 1)),
            act_fn(),
            nn.MaxPool2d((1, 3), stride=1, padding=(0, 1)),
            nn.Dropout(dropout)
        )

        # Time reduction
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channels=channel_b, out_channels=channel_t, kernel_size=(1, 7), padding=(0, 3), stride=(1, 1)),
            act_fn(),
            nn.Dropout(dropout)
        )

        # Chroma reduction
        self.conv_c = nn.Sequential(
            nn.Conv2d(channel_t, 1, 1),
            act_fn(),
            nn.Dropout(dropout),
        )

        self.output_dim = 12 * num_octaves  # Output dimension for pitch class (12 for each octave)

    def forward(self, x):
        # x: B * 6 * 180 * T

        x = self.conv_p[0](x)  # B * C_P * 180 * T
        for conv in self.conv_p[1:]:
            x = x + conv(x)
        x = self.conv_b(x)  # B * C_B * 60 * T
        x = self.conv_t(x)  # B * C_T * 60 * T
        x = self.conv_c(x)  # B * 1 * 60 * T

        x = x.squeeze(1)  # B * 60 * T
        x = x.transpose(1, 2)

        return x