import torch.nn as nn

from .cqt import HCQTNormalizer


ALL_FEATURES = dict(
    hcqt_normalizer=HCQTNormalizer,
)

