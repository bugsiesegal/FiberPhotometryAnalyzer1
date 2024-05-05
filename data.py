import os
from glob import glob

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from config import Config
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, \
    PowerTransformer


class FiberTrackingDataset(Dataset):
    """Dataset for fiber and tracking data."""

    def __init__(self, data, window_size, use_fiber=True, use_tracking=True):
        """
        Initialize the dataset with data configurations.
        :param data: List of tuples (fiber data, tracking data)
        :param window_size: Size of the data window
        :param use_fiber: Boolean, whether to include fiber data
        :param use_tracking: Boolean, whether to include tracking data
        """
        self.data = data
        self.window_size = window_size
        self.len_data = sum(fiber.shape[0] - window_size - 1 for fiber, _ in self.data)
        self.use_fiber = use_fiber
        self.use_tracking = use_tracking

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        """Returns a data point and its label."""
        for fiber, tracking in self.data:
            if idx < fiber.shape[0] - self.window_size - 1:
                return self._get_window(fiber, tracking, idx)
            idx = max(0, idx - fiber.shape[0])

    def _get_window(self, fiber, tracking, idx):
        """Extracts a window of data from the dataset."""
        # Get window
        window_end_idx = idx + self.window_size
        fiber_window = fiber[idx:window_end_idx].unsqueeze(1)
        tracking_window = self._get_tracking_window(tracking, fiber, idx)

        if self.use_fiber and self.use_tracking:
            return torch.hstack((fiber_window, tracking_window))
        elif self.use_fiber:
            return fiber_window
        elif self.use_tracking:
            return tracking_window

    def _get_tracking_window(self, tracking, fiber, idx):
        """Maps fiber indices to tracking indices and extracts tracking data."""
        tracking_indices = torch.linspace(0, tracking.shape[0], fiber.shape[0])
        subset_indices = tracking_indices[idx:idx + self.window_size].round().long().clamp(0, tracking.shape[0] - 1)
        return tracking[subset_indices]


class FiberTrackingDataModule(LightningDataModule):
    """
    A data module for loading and preparing fiber tracking data using PyTorch Lightning.
    """

    def __init__(self, config: Config):
        """
        Initialize the data module with configuration.
        :param config: Configuration object
        """
        super(FiberTrackingDataModule, self).__init__()
        self.config = config

        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.fiber_tracking_pairs = []
        self.data = []

        self.fiber_normalizer = self.normalizer()
        self.tracking_normalizer = self.normalizer()

    def normalizer(self):
        if self.config.normalization == 'min-max':
            return MinMaxScaler()
        elif self.config.normalization == 'standard':
            return StandardScaler()
        elif self.config.normalization == 'robust':
            return RobustScaler()
        elif self.config.normalization == 'max-abs':
            return MaxAbsScaler()
        elif self.config.normalization == 'quantile':
            return QuantileTransformer()
        elif self.config.normalization == 'power':
            return PowerTransformer()
        else:
            raise ValueError(f"Normalization method {self.config.normalization} not supported.")

    def load_fiber(self, fiber_path):
        """Load and process fiber data from a CSV file."""
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
        """Load and process tracking data from a CSV file."""
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

    def fit_normalization(self, data):
        """Fit normalization to data."""
        fiber_data = torch.cat([fiber for fiber, _ in data])
        tracking_data = torch.cat([tracking for _, tracking in data])
        self.fiber_normalizer.fit(fiber_data.reshape(-1, 1))
        self.tracking_normalizer.fit(tracking_data)

    def normalize_data(self, data):
        """Normalize data using the fitted normalization."""
        return [
            (torch.tensor(self.fiber_normalizer.transform(fiber.reshape(-1, 1)).reshape(-1)).to(torch.float32),
             torch.tensor(self.tracking_normalizer.transform(tracking)).to(torch.float32))
            for fiber, tracking in data
        ]

    def prepare_data(self) -> None:
        """Prepare data by loading paths and asserting correct matches."""
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

        # Fit normalization
        self.fit_normalization(self.data)

        # Normalize data
        self.data = self.normalize_data(self.data)

    def setup(self, stage=None):
        """Split data into train, val, test sets."""
        # Split data into train, val, test
        train_size = int(0.7 * len(self.data))
        val_size = int(0.15 * len(self.data))
        test_size = len(self.data) - train_size - val_size
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(
            self.data, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            FiberTrackingDataset(self.train_data, self.config.window_dim, self.config.use_fiber,
                                 self.config.use_tracking),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            FiberTrackingDataset(self.val_data, self.config.window_dim, self.config.use_fiber, self.config.use_tracking),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            FiberTrackingDataset(self.test_data, self.config.window_dim, self.config.use_fiber,
                                 self.config.use_tracking),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
