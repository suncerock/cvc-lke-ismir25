import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.backbone import ALL_BACKBONES
from models.model import LKEModel

def test_cqt_cnn():
    x = torch.randn(32, 1, 120, 100)
    model = ALL_BACKBONES['cnn'](in_channels=1, num_channels=8, dropout=0.3)
    tgt_shape = (32, 100, model.output_dim)
    output = model(x)
    assert output.shape == tgt_shape, f"Expected output shape {tgt_shape}, but got {output.shape}"

def test_hcqt_cnn():
    x = torch.randn(32, 6, 120, 100)
    model = ALL_BACKBONES['cnn'](in_channels=6, num_channels=8, dropout=0.3)
    tgt_shape = (32, 100, model.output_dim)
    output = model(x)
    assert output.shape == tgt_shape, f"Expected output shape {tgt_shape}, but got {output.shape}"

def test_octave():
    x = torch.randn(32, 1, 144, 100)
    model = ALL_BACKBONES['octave'](num_channels=32, bins_per_octave=24, num_octaves=6, dropout=0.2)
    tgt_shape = (32, 100, model.output_dim)
    output = model(x)
    assert output.shape == tgt_shape, f"Expected output shape {tgt_shape}, but got {output.shape}"

def test_octave_full():
    x = torch.randn(32, 1, 144, 100)
    model = ALL_BACKBONES['octave_full'](num_channels=32, bins_per_octave=24, num_octaves=6, dropout=0.2)
    tgt_shape = (32, 100, model.output_dim)
    output = model(x)
    assert output.shape == tgt_shape, f"Expected output shape {tgt_shape}, but got {output.shape}"

def test_pitchclass():
    x = torch.randn(32, 6, 120, 100)
    model = ALL_BACKBONES['pitchclass'](bins_per_octave=24, num_octaves=5, dropout=0.2)
    tgt_shape = (32, 100, model.output_dim)
    output = model(x)
    assert output.shape == tgt_shape, f"Expected output shape {tgt_shape}, but got {output.shape}"

def test_pitchclass_res():
    x = torch.randn(32, 6, 120, 100)
    model = ALL_BACKBONES['pitchclass_res'](n_prefilt_layers=5, bins_per_octave=24, num_octaves=5, dropout=0.2)
    tgt_shape = (32, 100, model.output_dim)
    output = model(x)
    assert output.shape == tgt_shape, f"Expected output shape {tgt_shape}, but got {output.shape}"

def test_model():
    model = LKEModel(
        feature=dict(
            name="hcqt_normalizer",
            args={"bins_per_octave": 24, "num_octaves": 6}
        ),
        backbone=dict(
            name="octave",
            args={"num_channels": 32, "bins_per_octave": 24, "num_octaves": 6, "dropout": 0.2}
        )
    )
    x = torch.randn(8, 1, 168, 100).abs()
    y = torch.randint(0, 24, (8, 100))
    model.training_step({"x": x, "y": y}, 0)
    model.validation_step({"x": x, "y": y}, 0)

    x = torch.randn(1, 1, 168, 256).abs()
    y = torch.randint(0, 24, (1, 256))
    model.test_step({"x": x, "y": y}, 0)