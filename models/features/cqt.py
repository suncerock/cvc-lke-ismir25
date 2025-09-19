import torch
import torch.nn as nn

SEMITONES_PER_OCTAVE = 12


class HCQTNormalizer(nn.Module):
    def __init__(self,  bins_per_octave=12, num_octaves=5):
        super().__init__()

        self.bins_per_octave = bins_per_octave
        self.bins_per_semitone = bins_per_octave // SEMITONES_PER_OCTAVE
        self.num_octaves = num_octaves

    def forward(self, x, y=None, shift=True):
        if y is None:
            assert not shift

        with torch.no_grad():

            shift = torch.randint(low=-4, high=7+1, size=(len(x), 1)) if shift else torch.zeros((len(x), 1), dtype=torch.int64)

            cqt_list = []
            for h in range(x.shape[1]):
                s = x[:, h]
                s = torch.log(s + 1e-6)

                s = s[torch.arange(len(s))[:, None], self.bins_per_semitone * (4 + shift) + torch.arange(self.num_octaves * self.bins_per_octave)]
                s = (s - torch.mean(s, dim=(1, 2), keepdim=True)) / (s.var(dim=(1, 2), keepdim=True) + 1e-6).sqrt()
                cqt_list.append(s.unsqueeze(dim=1))

            s = torch.cat(cqt_list, dim=1)
            shift = shift.to(x.device)

            if y is not None:
                if len(y.shape) == 3:
                    shift = shift.unsqueeze(dim=-1)
                y = torch.where(y != -1, (y - shift * 2) % (SEMITONES_PER_OCTAVE * 2), y)  # * 2 for major and minor

        return s, y