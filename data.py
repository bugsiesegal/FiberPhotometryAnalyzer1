import os

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from config import Config
from glob import glob
import pandas as pd


class FiberTrackingDataset(Dataset):
    def __init__(self, data, window_size, normalize_fiber=True, normalize_tracking=True, use_fiber=True, use_tracking=True):
        self.data = data
        self.window_size = window_size

        len_data = 0

        for fiber, tracking in self.data:
            len_data += fiber.shape[0] - window_size - 1

        self.len_data = len_data

        self.normalize_fiber = normalize_fiber
        self.normalize_tracking = normalize_tracking

        self.use_fiber = use_fiber
        self.use_tracking = use_tracking

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        for fiber, tracking in self.data:
            if idx < fiber.shape[0] - self.window_size - 1:
                tracking_idx_fiber = torch.arange(0, tracking.shape[0], step=tracking.shape[0] / fiber.shape[0])
                tracking_idx_fiber_subset = tracking_idx_fiber[idx:idx + self.window_size]
                tracking_idx_fiber_subset = tracking_idx_fiber_subset.round().long()
                tracking_idx_fiber_subset = tracking_idx_fiber_subset.clamp(0, tracking.shape[0] - 1)
                fiber = fiber[idx:idx + self.window_size].unsqueeze(1)
                tracking = tracking[tracking_idx_fiber_subset]
                if self.use_fiber and self.use_tracking:
                    return torch.hstack((fiber, tracking))
                elif self.use_fiber:
                    return fiber
                elif self.use_tracking:
                    return tracking

            else:
                idx -= fiber.shape[0]
                if idx < 0:
                    idx = 0


class FiberTrackingDataModule(LightningDataModule):
    def __init__(self, config: Config):
        super(FiberTrackingDataModule, self).__init__()
        self.config = config

        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.fiber_tracking_pairs = []
        self.data = []

    def load_fiber(self, fiber_path):
        # Load fiber data from csv
        fiber_df = pd.read_csv(fiber_path)
        # Get fiber channel and convert to tensor
        fiber_channel = torch.tensor(fiber_df[self.config.fiber_channel_name].to_numpy())
        # Get control channel
        control_channel = torch.tensor(fiber_df[self.config.control_channel_name].to_numpy())
        # Subtract control channel from fiber channel
        fiber_channel -= control_channel
        # Convert to float16 for memory efficiency
        fiber_channel = fiber_channel.to(torch.float32)
        return fiber_channel

    def load_tracking(self, tracking_path):
        # Load tracking data from csv
        tracking_df = pd.read_csv(tracking_path)
        # Get column names
        column_names = tracking_df.columns
        # Get columns with x, y, z
        x_columns = [column for column in column_names if '_x' in column]
        y_columns = [column for column in column_names if '_y' in column]
        z_columns = [column for column in column_names if '_z' in column]
        # Convert to tensor
        x = torch.tensor(tracking_df[x_columns].to_numpy())
        y = torch.tensor(tracking_df[y_columns].to_numpy())
        z = torch.tensor(tracking_df[z_columns].to_numpy())
        # Concatenate x, y, z
        tracking = torch.cat([x, y, z], dim=1)
        # Convert to float16 for memory efficiency
        tracking = tracking.to(torch.float32)
        return tracking

    def prepare_data(self) -> None:
        # Get fiber and tracking paths
        fiber_paths = glob(os.path.join(self.data_dir, '**/Fiber/*.csv'), recursive=True)
        tracking_paths = glob(os.path.join(self.data_dir, '**/Tracking/*.csv'), recursive=True)
        # Check if length of fiber and tracking paths are the same
        assert len(fiber_paths) == len(tracking_paths)
        # Find tracking path which includes fiber paths base name
        for fiber_path in fiber_paths:
            fiber_base_name = fiber_path.split('/')[-1].split('.')[0]
            tracking_path = [tracking_path for tracking_path in tracking_paths if fiber_base_name in tracking_path][0]
            self.fiber_tracking_pairs.append((fiber_path, tracking_path))

        # Load data
        self.data = [
            (self.load_fiber(fiber_path), self.load_tracking(tracking_path))
            for fiber_path, tracking_path in self.fiber_tracking_pairs
        ]

    def setup(self, stage=None):
        # Split data into train, val, test
        train_size = int(0.7 * len(self.data))
        val_size = int(0.15 * len(self.data))
        test_size = len(self.data) - train_size - val_size
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(
            self.data, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            FiberTrackingDataset(self.train_data, self.config.window_dim, self.config.fiber_normalize, self.config.tracking_normalize, self.config.use_fiber, self.config.use_tracking),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            FiberTrackingDataset(self.val_data, self.config.window_dim, self.config.fiber_normalize, self.config.tracking_normalize, self.config.use_fiber, self.config.use_tracking),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            FiberTrackingDataset(self.test_data, self.config.window_dim, self.config.fiber_normalize, self.config.tracking_normalize),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )