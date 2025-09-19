import os
from typing import Literal, Optional, List

import torch.utils.data as Data
from lightning import LightningDataModule

from .dataset import LocalKeySinglePieceDataset
from .bsqd_split import get_split_list as get_split_list_BSQD
from .bpsd_split import get_split_list as get_split_list_BPSD
from .swd_split import get_split_list as get_split_list_SWD

def get_split_list(dataset):
    if dataset == "swd":
        return get_split_list_SWD()
    elif dataset == "bpsd":
        return get_split_list_BPSD()
    elif dataset == "bsqd":
        return get_split_list_BSQD()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


class LocalKeyDataModule(LightningDataModule):
    def __init__(
        self,

        train_val_feature_folders: List[str],
        train_val_label_folders: List[str],
        train_val_datasets: List[Literal["swd", "bpsd", "bsqd"]],

        test_feature_folders: List[str],
        test_label_folders: List[str],
        test_datasets: List[Literal["swd", "bpsd", "bsqd"]],

        feature_fps: Optional[int] = None,
        label_fps: Optional[int] = None,

        seg_length: float = 10.0,
        seg_shift_length: float = 2.0,

        batch_size: int = 32,
        num_workers: int = 8
    ) -> None:
        super().__init__()

        self.train_val_datasets = train_val_datasets
        self.train_val_feature_folders = train_val_feature_folders
        self.train_val_label_folders = train_val_label_folders

        self.test_datasets = test_datasets
        self.test_feature_folders = test_feature_folders
        self.test_label_folders = test_label_folders
        self.num_test_datasets = len(test_datasets)

        self.seg_length = seg_length
        self.seg_shift_length = seg_shift_length

        self.feature_fps = feature_fps
        self.label_fps = label_fps

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        train_datasets, val_datasets = [], []
        for train_val_dataset, train_val_feature_folder, train_val_label_folder in zip(
            self.train_val_datasets, self.train_val_feature_folders, self.train_val_label_folders):
            
            train_list, val_list, _ = get_split_list(train_val_dataset)

            train_dataset = self.create_dataset_from_list(
                train_list, train_val_feature_folder, train_val_label_folder, train_val_dataset, test=False
            )
            val_dataset = self.create_dataset_from_list(
                val_list, train_val_feature_folder, train_val_label_folder, train_val_dataset, test=False
            )
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        test_datasets = []
        for test_dataset, test_feature_folder, test_label_folder in zip(
            self.test_datasets, self.test_feature_folders, self.test_label_folders):
            _, _, test_lists = get_split_list(dataset=test_dataset)
            test_dataset = self.create_dataset_from_list(
                test_lists, test_feature_folder, test_label_folder, test_dataset, test=True
            )
            test_datasets.append(test_dataset)

        self.train_dataset = Data.ConcatDataset(train_datasets)
        self.val_dataset = Data.ConcatDataset(val_datasets)
        self.test_dataset = test_datasets

    def train_dataloader(self):
        return Data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return Data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return [Data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers
        ) for test_dataset in self.test_dataset]

    def create_dataset_from_list(self, track_list, feature_folder, label_folder, dataset, test=False):
        dataset_list = []
        for track in track_list:
            single_piece_dataset = LocalKeySinglePieceDataset(
                feature_path=os.path.join(feature_folder, f"{track}.pt"),
                feature_fps=self.feature_fps,
                label_path=os.path.join(label_folder, f"{track}.csv"),
                label_fps=self.label_fps,
                dataset=dataset,
                seg_length=self.seg_length if not test else -1,
                seg_shift_length=self.seg_shift_length
            )
            dataset_list.append(single_piece_dataset)
        return Data.ConcatDataset(dataset_list)
