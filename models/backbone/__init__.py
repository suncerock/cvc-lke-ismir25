import torch.nn as nn

from .octave import Octave, OctaveFull
from .pitchclass_cnn import PitchClassCNNBasic, PitchClassCNNRes
from .cnn import DeepSquare

ALL_BACKBONES = dict(
    octave=Octave,
    octave_full=OctaveFull,
    pitchclass=PitchClassCNNBasic,
    pitchclass_res=PitchClassCNNRes,
    cnn=DeepSquare,
)