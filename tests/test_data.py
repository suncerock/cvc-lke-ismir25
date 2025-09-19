import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datamodule import LocalKeyDataModule

def test_dm():
    dm = LocalKeyDataModule(
        train_val_feature_folders=["../datasets/Schubert_Winterreise_Dataset_v2-1/99_Features/hcqt_h2205_b36_o7_h1/"],
        train_val_label_folders=["../datasets/Schubert_Winterreise_Dataset_v2-1/02_Annotations/ann_audio_localkey-ann3/"],
        train_val_datasets=["swd"],

        test_feature_folders=["../datasets/Schubert_Winterreise_Dataset_v2-1/99_Features/hcqt_h2205_b36_o7_h1/"],
        test_label_folders=["../datasets/Schubert_Winterreise_Dataset_v2-1/02_Annotations/ann_audio_localkey-ann3/"],
        test_datasets=["swd"],

        feature_fps=5,
        label_fps=5,

        seg_length=20.0,
        seg_shift_length=1.0,

        batch_size=8,
        num_workers=0
    )

    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loaders = dm.test_dataloader()

    batch = next(iter(train_loader))
    x, y = batch["x"], batch["y"]
    assert x.shape == (8, 1, 252, 100), f"Expected input shape (8, 6, 120, 101), but got {x.shape}"
    assert y.shape == (8, 100), f"Expected label shape (8, 101), but got {y.shape}"

    batch = next(iter(val_loader))
    x, y = batch["x"], batch["y"]
    assert x.shape == (8, 1, 252, 100), f"Expected input shape (8, 6, 120, 101), but got {x.shape}"
    assert y.shape == (8, 100), f"Expected label shape (8, 101), but got {y.shape}"

    for test_loader in test_loaders:
        batch = next(iter(test_loader))
        x, y = batch["x"], batch["y"]
        assert x.shape[0] == 1, f"Expected batch size 1, but got {x.shape[0]}"
        assert x.shape[1:-1] == (1, 252), f"Expected input shape (1, 6, 120, *), but got {x.shape}"
        assert y.shape[0] == 1, f"Expected batch size 1, but got {y.shape[0]}"
    