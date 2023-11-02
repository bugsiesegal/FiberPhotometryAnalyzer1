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

from cebra import CEBRA
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from scipy.stats import gaussian_kde

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

behavior_to_color = {
    'rear': 'dodgerblue',  # Retained from original
    'rest': 'darkblue',  # Retained from original
    'back': 'lightblue',  # Added, distinct from other blues
    'locomotion': 'cyan',  # Changed from 'c' for clarity
    'face': 'darkred',  # Retained from original
    'walk': 'blue',  # Changed to avoid confusion with 'rear'
    'immobile': 'navy',  # Changed from 'b' for specificity
    'jump': 'gold',  # Added, distinct color
    'groom': 'salmon',  # Retained from original
    'body': 'red',  # Retained from original, interpreted from 'r'
    'scratch': 'firebrick',  # Retained from original
    'None': "gray"
}


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


def get_data(paths: Union[List[Path], List[str]]) -> pd.DataFrame:
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


def load_model(path: Union[str, Path], optimize_for_inference: bool = False) -> AmazingAutoEncoder:
    """
    Load a model from a given path and optionally optimize it for inference on a single GPU.
    """
    model = AmazingAutoEncoder.load_from_checkpoint(path)
    model.eval()  # Set the model to inference mode

    if optimize_for_inference:
        # Optional: Apply JIT compilation for optimization
        model = torch.jit.script(model)

        # Optional: Apply quantization here if the model and hardware support it
        # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

        # Ensure the model is on the GPU
        model.to('cuda')

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
                return torch.tensor(np.linspace(0, len(struct.streams.LMag.data[0]) * 1 / struct.streams.LMag.fs,
                                                len(struct.streams.LMag.data[0]))).unfold(0, window_size, step_size)
            except AttributeError as e:
                print(f"AttributeError encountered: {e}")
                return None
        return None

    new_data = data.copy()
    new_data['sliding_windows'] = new_data['data'].apply(_create_windows_if_possible)
    new_data['time_windows'] = new_data['data'].apply(_create_time_windows_if_possible)
    return new_data


def create_dataloader(data: pd.DataFrame, batch_size: int = 32, prefetch_factor=2) -> DataLoader:
    """
    Create a DataLoader for inference on the fiber data, keeping track of original DataFrame indices.

    Parameters:
        data (pd.DataFrame): The input DataFrame expected to have a 'sliding_windows' column containing PyTorch tensors.
        batch_size (int, optional): How many samples per batch to load. Default is 32.
        prefetch_factor (int, optional): Number of samples loaded in advance by each worker. Default is 2.
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
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False, prefetch_factor=prefetch_factor,
                            num_workers=os.cpu_count() - 1)

    return dataloader


def perform_inference_and_update_df(data: pd.DataFrame, model: AmazingAutoEncoder,
                                    batch_size: int = 32, show_progress: bool = True) -> pd.DataFrame:
    data_copy = data.copy()
    fiber_data_rows = data_copy[data_copy['behavior_or_fiber'] == 'fiber']

    dataloader = create_dataloader(fiber_data_rows, batch_size)

    result_dict = {}
    reconstructed_dict = {}
    cuda_available = torch.cuda.is_available()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Performing inference", disable=not show_progress):
            inputs, indices = batch
            if cuda_available:
                inputs = inputs.cuda()
            outputs = model.encoder(inputs.unsqueeze(-1))
            recreated = model.decoder(outputs)

            outputs = outputs.squeeze(-1).cpu().numpy()
            recreated = recreated.squeeze(-1).cpu().numpy()

            for idx, output in zip(indices, outputs):
                if idx.item() in result_dict:
                    result_dict[idx.item()].append(output)
                else:
                    result_dict[idx.item()] = [output]

            for idx, output in zip(indices, recreated):
                if idx.item() in reconstructed_dict:
                    reconstructed_dict[idx.item()].append(output)
                else:
                    reconstructed_dict[idx.item()] = [output]

    new_column = [result_dict.get(idx, None) for idx in range(len(data_copy))]
    reconstructed_column = [reconstructed_dict.get(idx, None) for idx in range(len(data_copy))]
    data_copy['inference_results'] = new_column
    data_copy['reconstructed_results'] = reconstructed_column

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


def get_cebra(data: pd.DataFrame, n_components: int = 2, max_iterations=5000) -> CEBRA:
    cebra_model = CEBRA(
        model_architecture="offset1-model-mse",
        batch_size=512,
        learning_rate=1e-4,
        max_iterations=max_iterations,
        delta=0.1,
        conditional='delta',
        output_dimension=n_components,
        distance='euclidean',
        device="cuda_if_available",
        verbose=True,
    )

    inference_results = get_inference_results(data)
    inference_results = np.concatenate(inference_results)

    cebra_model.fit(inference_results)
    return cebra_model


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


def perform_cebra_on_inference_results(data: pd.DataFrame, cebra_model: CEBRA) -> pd.DataFrame:
    data_copy = data.copy()
    filtered_data = get_inference_results(data)

    if not filtered_data:
        return data_copy

    # Flatten the inference_results for CEBRA and also keep track of lengths for each original row
    flat_data = []
    lengths = []
    for result_set in filtered_data:
        flat_data.extend(result_set)
        lengths.append(len(result_set))

    flat_data = np.array(flat_data)
    cebra_results = cebra_model.transform(flat_data)

    # Create a new column for the CEBRA results with default value as None
    data_copy['cebra_results'] = None

    # Assign the CEBRA results back to DataFrame based on the lengths array
    start_idx = 0
    for idx, row in data.iterrows():
        if row['behavior_or_fiber'] == 'fiber':
            end_idx = start_idx + lengths.pop(0)  # Use lengths array to get the range for each row
            data_copy.at[idx, 'cebra_results'] = cebra_results[start_idx:end_idx].tolist()
            start_idx = end_idx  # Update the starting index for the next iteration

    return data_copy


def segment_results_on_behavior(results: List[np.ndarray], time_arrays: List[np.ndarray],
                                behavior_data: pd.DataFrame) -> pd.DataFrame:
    segments = []

    try:
        behavior_data = behavior_data.iloc[0]
    except IndexError:
        warnings.warn("No behavior data found for this fiber.")
        return pd.DataFrame(segments)

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
    try:
        num_plots = len(segmentation_results['result'][0])
    except KeyError:
        warnings.warn("No segmentation results found for this fiber.", RuntimeWarning)
        return

    fig, axs = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 5 * num_plots))
    fig.suptitle('Segmented Time Series')

    unique_behaviors = segmentation_results['behavior'].fillna('None').unique()
    legend_handles = [plt.Line2D([0], [0], color=behavior_to_color[ub], lw=2, label=ub) for ub in unique_behaviors]

    for idx, ax in enumerate(axs):
        ax.set_ylabel(f"Component {idx + 1}")

        prev_idx = 0
        for i, behavior in enumerate(segmentation_results['behavior'].fillna('None')):
            color = behavior_to_color[behavior]
            ax.plot(range(prev_idx, i + 1), segmentation_results['result'].iloc[prev_idx:i + 1].apply(lambda x: x[idx]),
                    c=color)
            prev_idx = i

        ax.legend(handles=legend_handles)

    if save_path:
        plt.savefig(save_path)

    plt.close()


def plot_autoencoder_results(original_series, autoencoder_results: pd.DataFrame,
                             save_path: Optional[Union[str, Path]] = None):
    try:
        original_series = original_series[0]
        reconstructed_series = autoencoder_results[0]
    except KeyError:
        warnings.warn("Original and/or reconstructed series not found.", RuntimeWarning)
        return

    # Step 1: Initialize Figure
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 15))
    fig.suptitle('Autoencoder Results')

    # Step 2: Prepare data
    # This is already done, original_series and reconstructed_series are loaded.

    # Step 3: Plot original data
    axs[0].plot(original_series)
    axs[0].set_title('Original Time Series')

    # Step 4: Plot reconstructed data
    axs[1].plot(reconstructed_series)
    axs[1].set_title('Reconstructed Time Series')

    # Step 5: Plot residuals
    residuals = np.array(original_series) - np.array(reconstructed_series)
    axs[2].plot(residuals)
    axs[2].set_title('Residuals')

    # Step 6: Optional - If you have additional attributes, you can segment and color code here

    # Step 7: Save the figure if a save path is specified
    if save_path:
        plt.savefig(save_path)

    plt.close()


def plot_combinations_of_components(segmentation_results: pd.DataFrame, exclude_behaviors: Optional[List[str]] = None,
                                    save_path: Optional[Union[str, Path]] = None):
    try:
        num_plots = len(segmentation_results['result'][0])
    except KeyError:
        warnings.warn("No segmentation results found for this fiber.", RuntimeWarning)
        return

    unique_behaviors = segmentation_results['behavior'].fillna('None').unique()
    if exclude_behaviors:
        unique_behaviors = [b for b in unique_behaviors if b not in exclude_behaviors]

    num_subplots = num_plots * (num_plots - 1) // 2
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 5 * num_subplots))
    fig.suptitle('Combinations of Components')

    subplot_idx = 0
    for idx1, idx2 in combinations(range(num_plots), 2):
        ax = axs[subplot_idx]
        ax.set_title(f"Component {idx1 + 1} vs Component {idx2 + 1}")
        ax.set_xlabel(f"Component {idx1 + 1}")
        ax.set_ylabel(f"Component {idx2 + 1}")

        for behavior in unique_behaviors:
            if behavior == 'None':
                mask = segmentation_results['behavior'].isna()
            else:
                mask = segmentation_results['behavior'] == behavior

            x_data = segmentation_results[mask]['result'].apply(lambda x: x[idx1])
            y_data = segmentation_results[mask]['result'].apply(lambda x: x[idx2])
            color = behavior_to_color[behavior]

            ax.scatter(x_data, y_data, c=[color] * len(x_data), label=behavior)

            # Plot centroids
            centroid_x = x_data.mean()
            centroid_y = y_data.mean()
            ax.scatter(centroid_x, centroid_y, c=color, marker='o', s=100, label=f"Centroid-{behavior}",
                       edgecolor='black', linewidth=1, zorder=10)

        ax.legend(title='Behavior', bbox_to_anchor=(1.05, 1), loc='upper left')

        subplot_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def plot_density_based_combinations(segmentation_results: pd.DataFrame,
                                    exclude_behaviors: Optional[List[str]] = None,
                                    save_path: Optional[Union[str, Path]] = None):
    try:
        num_plots = len(segmentation_results['result'][0])
    except KeyError:
        warnings.warn("No segmentation results found for this fiber.", RuntimeWarning)
        return

    unique_behaviors = segmentation_results['behavior'].fillna('None').unique()
    if exclude_behaviors:
        unique_behaviors = [b for b in unique_behaviors if b not in exclude_behaviors]

    num_subplots = num_plots * (num_plots - 1) // 2
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 5 * num_subplots))
    fig.suptitle('Density-Based Combinations of Components')

    subplot_idx = 0
    for idx1, idx2 in combinations(range(num_plots), 2):
        ax = axs[subplot_idx]
        ax.set_title(f"Component {idx1 + 1} vs Component {idx2 + 1}")
        ax.set_xlabel(f"Component {idx1 + 1}")
        ax.set_ylabel(f"Component {idx2 + 1}")

        all_kdes = {}
        all_data = []

        for behavior in unique_behaviors:
            if behavior == 'None':
                mask = segmentation_results['behavior'].isna()
            else:
                mask = segmentation_results['behavior'] == behavior

            x_data = segmentation_results[mask]['result'].apply(lambda x: x[idx1]).to_numpy()
            y_data = segmentation_results[mask]['result'].apply(lambda x: x[idx2]).to_numpy()

            all_data.append(np.vstack([x_data, y_data]))

            if len(x_data) < 2:  # Need at least two points for density estimation
                print(f"Not enough data points for {behavior}")
                continue

            data = np.vstack([x_data, y_data])
            try:
                kde = gaussian_kde(data)
            except np.linalg.LinAlgError:
                warnings.warn(f"Could not compute KDE for {behavior}", RuntimeWarning)
                continue

            all_kdes[behavior] = kde

        all_data = np.concatenate(all_data, axis=1)
        x_min, x_max = np.min(all_data[0, :]), np.max(all_data[0, :])
        y_min, y_max = np.min(all_data[1, :]), np.max(all_data[1, :])

        # Create grid
        x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

        max_behavior_at_position = np.empty(positions.shape[1], dtype=object)
        max_density_at_position = np.zeros(positions.shape[1])

        for behavior, kde in all_kdes.items():
            densities = kde(positions)
            mask = densities > max_density_at_position
            max_density_at_position[mask] = densities[mask]
            max_behavior_at_position[mask] = behavior

        # Plot
        for behavior in unique_behaviors:
            mask = max_behavior_at_position == behavior
            ax.scatter(positions[0, mask], positions[1, mask], c=behavior_to_color[behavior], label=behavior,
                       alpha=1)

        ax.legend(title='Behavior', bbox_to_anchor=(1.05, 1), loc='upper left')

        subplot_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
