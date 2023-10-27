from typing import List, Tuple, Optional, Union
import glob
import math
import os
import pickle
import re
import warnings
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import tdt
import torch
from JAAEC import AmazingAutoEncoder
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import click
from torch.utils.data import TensorDataset, DataLoader
from umap import UMAP

MILLISECONDS_TO_SECONDS = 1000


def get_behavior_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read behavioral data from a CSV file, clean it, and return it as a DataFrame.
    """
    behavior_data = pd.read_csv(path, skiprows=2)
    behavior_data.drop(columns=["comment"], inplace=True)
    behavior_data["Time"] = behavior_data["Time"] / MILLISECONDS_TO_SECONDS
    behavior_data["Duration"] = behavior_data["Duration"] / MILLISECONDS_TO_SECONDS
    return behavior_data


def get_fiber_data(path: Union[str, Path]) -> tdt.StructType:
    """
    Read fiber data using the tdt.read_block function.
    """
    fiber_data = tdt.read_block(str(path))
    return fiber_data


def parse_filename(filename: str) -> Optional[dict]:
    """
    Parse the filename to extract experiment parameters and details.
    """
    try:
        components = filename.split(os.path.sep)
        folder_info = components[-2] if components else ""
        folder_parts = folder_info.split(" - ")

        if len(folder_parts) < 2:
            raise ValueError(f"Folder information is incomplete in {filename}")

        pathway, drug_or_vehicle, behavior_or_fiber = folder_parts[0].split()
        genotype = folder_parts[1].split()[-1]
        animal_search = re.search(r'Animal(\d+)|D\d+-(\d+)', filename)

        if not animal_search:
            raise ValueError(f"Could not find animal information in {filename}")

        animal = animal_search.group(1) or animal_search.group(2)

        return {
            'pathway': pathway,
            'drug_or_vehicle': drug_or_vehicle,
            'genotype': genotype,
            'behavior_or_fiber': behavior_or_fiber,
            'animal': animal,
        }

    except Exception as e:
        raise ValueError(f"Could not parse filename {filename}: {e}")


def get_data(paths: List[Path]) -> pd.DataFrame:
    """
    Parallelize the parsing and data loading process.
    """
    records = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(parse_and_load_data, paths)
    records.extend(results)
    return pd.DataFrame(records)


def parse_and_load_data(path: Path) -> Optional[dict]:
    """
    Helper function to parse filenames and load corresponding data.
    """
    try:
        parsed_info = parse_filename(str(path))
        if parsed_info is None:
            return None

        if parsed_info['behavior_or_fiber'] == 'fiber':
            data = get_fiber_data(path)
        else:
            data = get_behavior_data(path)

        parsed_info['data'] = data
        return parsed_info
    except Exception as e:
        raise RuntimeError(f"Could not process path {path}: {e}")


def load_model(path: Union[str, Path]) -> AmazingAutoEncoder:
    """
    Load a model from a given path.
    """
    model = AmazingAutoEncoder.load_from_checkpoint(path)
    model.eval()
    return model


def create_sliding_windows(tensor: torch.Tensor, window_size: int, step_size: int) -> torch.Tensor:
    """
    Create sliding windows from a given tensor.
    """

    windows = tensor.unfold(0, window_size, step_size)
    windows = (windows - windows.mean()) / windows.std()
    return windows


def add_sliding_windows(data: pd.DataFrame, window_size: int, step_size: int) -> pd.DataFrame:
    """
    Add sliding windows to the data.

    Parameters:
        data (pd.DataFrame): The input DataFrame expected to have a 'data' column.
        window_size (int): The size of the sliding window.
        step_size (int): The step size for the sliding window.

    Returns:
        pd.DataFrame: A new DataFrame with an additional 'sliding_windows' column.
    """

    def _create_windows_if_possible(struct):
        if isinstance(struct, tdt.StructType):
            try:
                return create_sliding_windows(torch.tensor(struct.streams.LMag.data[0]), window_size, step_size)
            except AttributeError as e:
                print(f"AttributeError encountered: {e}")
                return None
        return None

    def _create_time_windows_if_possible(struct):
        if isinstance(struct, tdt.StructType):
            try:
                return create_sliding_windows(
                    torch.tensor(np.linspace(0, len(struct.streams.LMag.data[0]) * 1 / struct.streams.LMag.fs,
                                             len(struct.streams.LMag.data[0]))), window_size, step_size
                )
            except AttributeError as e:
                print(f"AttributeError encountered: {e}")
                return None
        return None

    new_data = data.copy()
    new_data['sliding_windows'] = new_data['data'].apply(_create_windows_if_possible)
    new_data['time_windows'] = new_data['data'].apply(_create_time_windows_if_possible)
    return new_data


def create_dataloader(data: pd.DataFrame, batch_size: int = 32) -> DataLoader:
    """
    Create a DataLoader for inference on the fiber data, keeping track of original DataFrame indices.

    Parameters:
        data (pd.DataFrame): The input DataFrame expected to have a 'sliding_windows' column containing PyTorch tensors.
        batch_size (int, optional): How many samples per batch to load. Default is 32.
    Returns:
        DataLoader: DataLoader for the fiber data for inference.
    """

    # Step 2: Extract the 'sliding_windows' and collect original DataFrame indices
    tensor_list = []
    index_list = []
    for index, row in data.iterrows():
        tensor = row['sliding_windows']
        if tensor is not None:
            tensor_list.append(tensor)
            # Extend the index_list to account for each row in the tensor
            index_list.extend([index] * len(tensor))

    # Step 3: Concatenate list of tensors along a new dimension
    concatenated_tensor = torch.vstack(tensor_list)

    # Step 3.5: Convert index_list to a tensor
    index_tensor = torch.tensor(index_list, dtype=torch.long)

    # Step 4: Create a TensorDataset including indices
    tensor_dataset = TensorDataset(concatenated_tensor, index_tensor)

    # Step 5: Create and return a DataLoader
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def perform_inference_and_update_df(data: pd.DataFrame, model: AmazingAutoEncoder,
                                    batch_size: int = 32, show_progress: bool = True) -> pd.DataFrame:
    data_copy = data.copy()
    fiber_data_rows = data_copy[data_copy['behavior_or_fiber'] == 'fiber']

    dataloader = create_dataloader(fiber_data_rows, batch_size)

    result_dict = {}
    cuda_available = torch.cuda.is_available()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Performing inference", disable=not show_progress):
            inputs, indices = batch
            if cuda_available:
                inputs = inputs.cuda()
            outputs = model.encoder(inputs.unsqueeze(-1)).squeeze(-1).cpu().numpy()

            for idx, output in zip(indices, outputs):
                if idx.item() in result_dict:
                    result_dict[idx.item()].append(output)
                else:
                    result_dict[idx.item()] = [output]

    new_column = [result_dict.get(idx, None) for idx in range(len(data_copy))]
    data_copy['inference_results'] = new_column

    return data_copy


def get_x_results(data: pd.DataFrame, column, filter_column='behavior_or_fiber', filter_value='fiber') -> List:
    filtered_data = data[data[filter_column] == filter_value]
    inference_results = filtered_data[column].tolist()
    return [result for result in inference_results if result is not None]


def get_pca(data: pd.DataFrame, n_components: int = 2) -> PCA:
    inference_results = get_inference_results(data)
    inference_results = np.concatenate(inference_results)
    pca = PCA(n_components=n_components)
    pca.fit(inference_results)
    return pca


def get_umap(data: pd.DataFrame, n_components: int = 2) -> UMAP:
    inference_results = get_inference_results(data)
    inference_results = np.concatenate(inference_results)
    umap = UMAP(n_components=n_components)
    umap.fit(inference_results)
    return umap


def get_inference_results(data: pd.DataFrame, filter_column='behavior_or_fiber', filter_value='fiber') -> List:
    filtered_data = data[data[filter_column] == filter_value]
    inference_results = filtered_data['inference_results'].tolist()
    return [result for result in inference_results if result is not None]


def perform_pca_on_inference_results(data: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    data_copy = data.copy()
    filtered_data = get_inference_results(data)
    if not filtered_data:
        return data_copy

    # Flatten the inference_results for PCA and also keep track of lengths for each original row
    flat_data = []
    lengths = []
    for result_set in filtered_data:
        flat_data.extend(result_set)
        lengths.append(len(result_set))

    flat_data = np.array(flat_data)
    pca_results = pca.transform(flat_data)

    # Create a new column for the PCA results with default value as None
    data_copy['pca_results'] = None

    # Assign the PCA results back to DataFrame based on the lengths array
    start_idx = 0
    for idx, row in data.iterrows():
        if row['behavior_or_fiber'] == 'fiber':
            end_idx = start_idx + lengths.pop(0)  # Use lengths array to get the range for each row
            data_copy.at[idx, 'pca_results'] = pca_results[start_idx:end_idx].tolist()
            start_idx = end_idx  # Update the starting index for the next iteration

    return data_copy


def perform_umap_on_inference_results(data: pd.DataFrame, umap: UMAP) -> pd.DataFrame:
    data_copy = data.copy()
    filtered_data = get_inference_results(data)

    if not filtered_data:
        return data_copy

    # Flatten the inference_results for UMAP and also keep track of lengths for each original row
    flat_data = []
    lengths = []
    for result_set in filtered_data:
        flat_data.extend(result_set)
        lengths.append(len(result_set))

    flat_data = np.array(flat_data)
    umap_results = umap.transform(flat_data)

    # Create a new column for the UMAP results with default value as None
    data_copy['umap_results'] = None

    # Assign the UMAP results back to DataFrame based on the lengths array
    start_idx = 0
    for idx, row in data.iterrows():
        if row['behavior_or_fiber'] == 'fiber':
            end_idx = start_idx + lengths.pop(0)  # Use lengths array to get the range for each row
            data_copy.at[idx, 'umap_results'] = umap_results[start_idx:end_idx].tolist()
            start_idx = end_idx  # Update the starting index for the next iteration

    return data_copy


def segment_results_on_behavior(results: List[np.ndarray], time_arrays: List[np.ndarray],
                                behavior_data: pd.DataFrame) -> pd.DataFrame:
    segments = []

    behavior_data = behavior_data[0]

    # Ensure that behavior_data is a DataFrame before sorting it.
    if isinstance(behavior_data, pd.DataFrame):
        behavior_data = behavior_data.sort_values(by='Time')
    elif isinstance(behavior_data, pd.Series):
        behavior_data = behavior_data.sort_values()

    for result, time_array in zip(results, time_arrays):
        behavior = None
        start_times = behavior_data['Time'].values
        end_times = start_times + behavior_data['Duration'].values
        track_names = behavior_data['TrackName'].values

        mask = (start_times < time_array.numpy()[0]) & (end_times > time_array.numpy()[-1])
        matching_behaviors = track_names[mask]

        if len(matching_behaviors) > 0:
            behavior = matching_behaviors[0]
        else:
            behavior = None

        segments.append({'result': result, 'behavior': behavior})

    return pd.DataFrame(segments)


def get_animal_behavior_row(data: pd.DataFrame, fiber_idx) -> pd.DataFrame:
    fiber_row = data.loc[fiber_idx]
    animal = fiber_row['animal']
    behavior_data = data[
        (data['behavior_or_fiber'] == 'behavior') &
        (data['animal'] == animal) &
        (data['pathway'] == fiber_row['pathway']) &
        (data['drug_or_vehicle'] == fiber_row['drug_or_vehicle']) &
        (data['genotype'] == fiber_row['genotype'])
        ]
    return behavior_data


def perform_segmentation(data: pd.DataFrame, segmentation_column_name="inference_results") -> pd.DataFrame:
    def segment_row(row):
        if row['behavior_or_fiber'] != 'fiber':
            return None
        animal_behavior_row = get_animal_behavior_row(data, row.name)
        return segment_results_on_behavior(row[segmentation_column_name], row['time_windows'],
                                           animal_behavior_row["data"])

    data_copy = data.copy()
    data_copy['segmentation_results'] = data_copy.apply(segment_row, axis=1)

    return data_copy


def plot_segmented_time_series(segmentation_results: pd.DataFrame, save_path: Optional[Union[str, Path]] = None):
    num_plots = len(segmentation_results['result'][0])

    fig, axs = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 100))
    fig.suptitle('Segmented Time Series')

    unique_behaviors = segmentation_results['behavior'].fillna('None').unique()
    behavior_to_int = {behavior: i for i, behavior in enumerate(unique_behaviors)}

    cmap = cm.get_cmap('tab20')

    for idx, ax in enumerate(axs):
        ax.set_ylabel(f"Component {idx + 1}")

        prev_idx = 0
        for i, behavior in enumerate(segmentation_results['behavior'].fillna('None')):
            color = cmap(behavior_to_int[behavior] % 20)
            ax.plot(range(prev_idx, i + 1), segmentation_results['result'].iloc[prev_idx:i + 1].apply(lambda x: x[idx]),
                    c=color)
            prev_idx = i

    plt.savefig(save_path)
