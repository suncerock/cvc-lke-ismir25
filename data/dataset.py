from typing import Optional

import soundfile as sf

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

from .utils import KEY_TO_INDEX, FN_KEY_REPLACE


class LocalKeySinglePieceDataset(Data.Dataset):
    def __init__(
        self,

        feature_path: str,
        feature_fps: float,
        label_path: str,
        label_fps: float,
        dataset: str,

        seg_length: float = 20.0,
        seg_shift_length: float = 1.0,


    ) -> None:
        """
        Dataset for a single piece, concatenated later into a single dataset

        Parameters
        ----------
        feature_path : str
            Path to the feature of the audio file
        feature_fps : float
            Frame rate of the feature in Hz
        label_path : str
            Path to the label file for local key estimation
        label_fps : float
            Frame rate of the label in Hz
        dataset : str
            Name of the dataset
        
        seg_length : float, optional
            Length of the segment in seconds, by default 20
        seg_shift_length : float, optional
            Shift length of the segment in seconds, by default 1
        """
        super().__init__()

        self.track_name = feature_path.split("/")[-1].replace(".pt", "")

        self.feature_path = feature_path
        self.feature_fps = feature_fps
        self.label_path = label_path

        self.seg_length = seg_length
        self.seg_shift_length = seg_shift_length
        self.seg_frames = int(self.feature_fps * self.seg_length) if seg_length > 0.0 else -1
        self.seg_shift_frames = int(self.feature_fps * seg_shift_length)

        self.label_fps = label_fps
        self.label_hop_frames = int(self.feature_fps / self.label_fps)

        # Load the label file
        self.dataset = dataset
        if dataset == "bsqd":
            self.label_df = pd.read_csv(self.label_path, sep=";", header=None, names=["start", "end", "key"])
        elif dataset == "swd" or dataset == "bpsd":
            self.label_df = pd.read_csv(self.label_path, sep=";", header=0)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.load_feature()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x, time_frame = self.read_feature(index)

        time_frame = time_frame[self.label_hop_frames // 2:len(time_frame):self.label_hop_frames]
        
        labels = -1 * np.ones(len(time_frame), dtype=np.int64)
        for _, row in self.label_df.iterrows():
            labels[(time_frame - 0.5 / self.label_fps > row["start"]) & (time_frame + 0.5 / self.label_fps < row["end"])] = KEY_TO_INDEX[FN_KEY_REPLACE(row["key" if self.dataset != "bpsd" else "localkey"])]

        return dict(x=x, y=labels, track_name=self.track_name)

    def load_feature(self):
        self.feature = torch.load(self.feature_path, weights_only=True, map_location="cpu")
        if self.seg_length > 0.0:
            # Compute how many segments are in the feature
            self.length = (self.feature.shape[-1] - self.seg_frames) // self.seg_shift_frames
        else:
            self.length = 1

    def read_feature(self, index):
        if self.seg_length > 0:
            start_frame = index * self.seg_shift_frames
            feature = self.feature[..., start_frame:start_frame+self.seg_frames]
            frame = np.arange(start_frame, start_frame+self.seg_frames)
        else:
            feature = self.feature
            frame = np.arange(0, feature.shape[-1])
        time_frame = frame / self.feature_fps
        return feature, time_frame
