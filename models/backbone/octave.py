import torch
import torch.nn as nn
import torch.nn.functional as F


class Octave(nn.Module):
    def __init__(
        self,
        num_channels=32,
        
        bins_per_octave=12,
        num_octaves=6,

        activation="GELU",
        dropout=0.2
    ) -> None:
        super().__init__()

        self.bins_per_octave = bins_per_octave
        self.num_octaves = num_octaves

        self.pointwise_conv = nn.Conv2d(self.num_octaves, num_channels, 1)
        self.pointwise_norm = nn.LayerNorm((self.bins_per_octave, num_channels))

        self.aggregation = nn.Conv2d(num_channels, num_channels, kernel_size=(self.bins_per_octave, 1), groups=num_channels)
        self.token_norm = nn.LayerNorm(num_channels)

        self.rnn = nn.LSTM(
            num_channels, num_channels, num_layers=2,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_channels * 2

        self.act = nn.__dict__.get(activation, "GELU")()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Input
        ----------
        x : torch.Tensor
            Input CQT of shape (batch_size, num_octaves * bins_per_octave, seq_len)

        Output
        ----------
        torch.Tensor
            Output of the model of shape (batch_size, seq_len, num_channels)
        """

        x = x.reshape(x.shape[0], self.num_octaves, self.bins_per_octave, x.shape[-1])  # batch_size, num_octaves, bins_per_octave, seq_len

        x = self.pointwise_conv(x)  # batch_size, num_channels, bins_per_octave, seq_len
        x = self.pointwise_norm(x.transpose(1, -1)).transpose(1, -1)
        x = self.act(x)

        x = self.aggregation(x).squeeze(dim=2).transpose(-1, -2)  # N, T, C
        x = self.token_norm(x)
        x = self.dropout(x)

        x, _ = self.rnn(x)  # N, T, C * 2
        x = self.dropout(x)

        return x
    
    def step_forward(self, x: torch.Tensor):
        return self(x)


class OctaveFull(nn.Module):
    def __init__(self, num_channels=16, bins_per_octave=24, num_octaves=6, dropout=0.2) -> None:
        super().__init__()

        self.bins_per_octave = bins_per_octave
        self.num_octaves = num_octaves

        self.conv_octave = nn.Conv2d(self.bins_per_octave, num_channels, 1)
        self.ln_octave = nn.LayerNorm((self.num_octaves, num_channels))

        self.agg_octave = nn.Conv2d(num_channels, num_channels, kernel_size=(self.num_octaves, 1))
        self.norm_octave = nn.LayerNorm(num_channels)
        self.rnn_octave = nn.LSTM(num_channels, num_channels, num_layers=1, batch_first=True, bidirectional=True)

        self.conv_chroma = nn.Conv2d(self.num_octaves, num_channels, 1)
        self.ln_chroma = nn.LayerNorm((self.bins_per_octave, num_channels))

        self.agg_chroma = nn.Conv2d(num_channels, num_channels, kernel_size=(self.bins_per_octave, 1))
        self.norm_chroma = nn.LayerNorm(num_channels)
        self.rnn_chroma = nn.LSTM(num_channels, num_channels, num_layers=1, batch_first=True, bidirectional=True)

        self.fuse = nn.Sequential(
            nn.Linear(num_channels * 4, num_channels * 2),
            nn.ReLU(),
            nn.Linear(num_channels * 2, num_channels * 2),
        )

        self.rnn = nn.LSTM(
            num_channels * 2, num_channels * 2, 1,
            batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_channels * 4

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):

        assert x.shape[2] == self.bins_per_octave * self.num_octaves

        x = x.reshape(x.shape[0], self.num_octaves, self.bins_per_octave, x.shape[-1])

        x_chroma = x
        x_octave = x.transpose(1, 2)

        x_octave = self.conv_octave(x_octave)
        x_octave = F.gelu(self.ln_octave(x_octave.transpose(1, -1))).transpose(1, -1)
        x_octave = self.agg_octave(x_octave).squeeze(dim=2).transpose(-1, -2)
        x_octave = self.norm_octave(x_octave)
        x_octave = self.dropout(x_octave)

        x_octave, _ = self.rnn_octave(x_octave)
        x_octave = self.dropout(x_octave)

        x_chroma = self.conv_chroma(x_chroma)
        x_chroma = F.gelu(self.ln_chroma(x_chroma.transpose(1, -1))).transpose(1, -1)
        x_chroma = self.agg_chroma(x_chroma).squeeze(dim=2).transpose(-1, -2)
        x_chroma = self.norm_chroma(x_chroma)
        x_chroma = self.dropout(x_chroma)

        x_chroma, _ = self.rnn_chroma(x_chroma)
        x_chroma = self.dropout(x_chroma)

        x = torch.concat([x_octave, x_chroma], dim=-1)

        x = self.dropout(self.fuse(x))

        x, _ = self.rnn(x)
        x = self.dropout(x)
        return x
