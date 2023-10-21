import lightning as pl
import glob

import numpy as np
import pandas as pd
import tdt
import torch
from torch.utils.data import Dataset, DataLoader


class TDTFiberPhotometryDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, window_size, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit":
            self.train_blocks = glob.glob(self.data_dir + "/train/*")
            self.val_blocks = glob.glob(self.data_dir + "/val/*")

            self.train_data = []
            self.val_data = []

            for block_path in self.train_blocks:
                block = tdt.read_block(block_path)

                data = block["streams"]["LMag"]["data"][0]

                data = (data - data.mean()) / data.std()

                self.train_data.append(
                    torch.tensor(data).unfold(0, self.window_size, 100).unsqueeze(2))

            for block_path in self.val_blocks:
                block = tdt.read_block(block_path)

                data = block["streams"]["LMag"]["data"][0]

                data = (data - data.mean()) / data.std()

                self.val_data.append(
                    torch.tensor(data).unfold(0, self.window_size, 100).unsqueeze(2))

            self.train_data = np.concatenate(self.train_data, axis=0)
            self.val_data = np.concatenate(self.val_data, axis=0)

            self.train_data = torch.tensor(self.train_data)
            self.val_data = torch.tensor(self.val_data)
        elif stage == "test":
            self.test_blocks = glob.glob(self.data_dir + "/test/*")

            self.test_data = []

            for block_path in self.test_blocks:
                block = tdt.read_block(block_path)

                data = block["streams"]["LMag"]["data"][0]

                data = (data - data.mean()) / data.std()

                self.test_data.append(
                    torch.tensor(data).unfold(0, self.window_size, 100).unsqueeze(2))

            self.test_data = np.concatenate(self.test_data, axis=0)

            self.test_data = torch.tensor(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)


class BehaviorDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, window_size, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit":
            self.train_data = []
            self.val_data = []

            self.train_paths = glob.glob(self.data_dir + "/train/*")
            self.val_paths = glob.glob(self.data_dir + "/val/*")

            for path in self.train_paths:
                data = pd.read_csv(path,
                                   header=1).drop("bodyparts", axis=1)
                data = data.loc[:, ~(data == 'likelihood').any()].drop(0).astype(np.float32).to_numpy()

                data = (data - data.mean()) / data.std()

                self.train_data.append(torch.tensor(data).unfold(0, self.window_size, 100).reshape(-1,
                                                                                                   self.window_size,
                                                                                                   8))

            for path in self.val_paths:
                data = pd.read_csv(path,
                                   header=1).drop("bodyparts", axis=1)
                data = data.loc[:, ~(data == 'likelihood').any()].drop(0).astype(np.float32).to_numpy()

                data = (data - data.mean()) / data.std()

                self.val_data.append(torch.tensor(data).unfold(0, self.window_size, 100).reshape(-1,
                                                                                                 self.window_size,
                                                                                                 8))

            self.train_data = np.concatenate(self.train_data, axis=0)
            self.val_data = np.concatenate(self.val_data, axis=0)

            self.train_data = torch.tensor(self.train_data)
            self.val_data = torch.tensor(self.val_data)
        elif stage == "test":
            self.test_data = []

            self.test_paths = glob.glob(self.data_dir + "/test/*")

            for path in self.test_paths:
                data = pd.read_csv(path,
                                   header=1).drop("bodyparts", axis=1)
                data = data.loc[:, ~(data == 'likelihood').any()].drop(0).astype(np.float32).to_numpy()

                data = (data - data.mean()) / data.std()

                self.test_data.append(torch.tensor(data).unfold(0, self.window_size, 100).reshape(-1,
                                                                                                  self.window_size,
                                                                                                  8))

            self.test_data = np.concatenate(self.test_data, axis=0)

            self.test_data = torch.tensor(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)
