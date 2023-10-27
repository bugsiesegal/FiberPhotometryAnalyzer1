# Imports
import glob
import math
import os
import pickle
import re
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
import tdt
import torch
from JAAEC import AmazingAutoEncoder
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import click
from pathlib import Path
from umap import UMAP


def get_paths(fiber_path: str, behavior_path: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Gets paths to each fiber block and behavior block that have the same name. The name is defined as the number after
    the first dash or Animal and before the next dash or underscore.
    :param fiber_path: Path to the fiber folder
    :param behavior_path: Path to the behavior folder
    :return: List of tuples containing the fiber path and behavior path
    """
    # Regex pattern to match the name of the fiber block and behavior block
    regex_pattern = r'(?:-|Animal)(\d+)(?:_|$)'

    # Gets paths to each fiber block
    fiber_paths = glob.glob(os.path.join(fiber_path, "*"))

    # Gets paths to each behavior block
    behavior_paths = glob.glob(os.path.join(behavior_path, "*"))

    path_tuples = []
    basepaths = []

    # Iterate through each fiber block path and behavior block path if they have the same name append them to the list
    for fiber_path in fiber_paths:
        for behavior_path in behavior_paths:
            # If the regex pattern matches the fiber path and behavior path
            if re.findall(regex_pattern, fiber_path) == re.findall(regex_pattern, behavior_path):
                path_tuples.append((fiber_path, behavior_path))
                basepaths.append(os.path.basename(fiber_path))
                break

    # Returns path tuples
    return path_tuples, basepaths


def load_fiber_data(fiber_path: str) -> List[Tuple[np.ndarray, float]]:
    """
    Loads the fiber data and the sampling rate from the fiber block
    :param fiber_path: Path to the fiber block
    :return: Fiber data and the sampling rate
    """
    # Load the block
    block = tdt.read_block(fiber_path)

    # Get the first channel of the fiber data
    data = block.streams.LMag.data[0]

    # Get the sampling rate
    sampling_rate = block.streams.LMag.fs

    # Return the data
    return data, sampling_rate


def load_behavior_data(behavior_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads the behavior data and the track names from the behavior block
    :param behavior_path: Path to the behavior file
    :return: Behavior data and the track names
    """
    # Load the behavior data
    behavior_data = pd.read_csv(behavior_path, skiprows=2)

    # Get the track names from the second line of the file
    with open(behavior_path) as f:
        input_string = f.readlines()[1]

        # Regular expression pattern to match alphanumeric track names
        pattern = r'\b\w+\b'

        # Using re.findall to extract all matching track names from the string
        track_names = re.findall(pattern, input_string)

        # Filter out the word "Tracks" to get only the track names
        track_names = [name for name in track_names if name != "Tracks"]

    # Remove the comment column from the behavior data
    behavior_data.drop(columns=["comment"], inplace=True)

    # Convert the time and duration columns to seconds (from milliseconds)
    behavior_data["Time"] = behavior_data["Time"] / 1000
    behavior_data["Duration"] = behavior_data["Duration"] / 1000

    # Return the behavior data
    return behavior_data, track_names


def get_data_pairs(
        fiber_paths: List[str],
        behavior_paths: List[str]) -> List[Tuple[np.ndarray, float, pd.DataFrame, List[str]]]:
    """
    Gets data pairs from the fiber paths and behavior paths
    :param fiber_paths: List of fiber paths
    :param behavior_paths: List of behavior paths
    :return: List of fiber data, sampling rate, behavior data, and track names
    """
    for fiber_path, behavior_path in zip(fiber_paths, behavior_paths):
        fiber_data, sampling_rate = load_fiber_data(fiber_path)
        behavior_data, track_names = load_behavior_data(behavior_path)

        yield fiber_data, sampling_rate, behavior_data, track_names


def unfold_fiber_data(
        fiber_data: np.ndarray,
        sampling_rate: float,
        window_size=1000,
        stride=100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unfolds the fiber data and time array
    :param fiber_data: Fiber data
    :param sampling_rate: Sampling rate
    :param window_size: Window size
    :param stride: Stride
    :return: Unfolded fiber data and time array
    """
    # Generate a time array
    time_array = np.arange(0, len(fiber_data) / sampling_rate, 1 / sampling_rate)

    # Convert the fiber data and time array to a tensor
    fiber_data = torch.from_numpy(fiber_data)
    time_array = torch.from_numpy(time_array)

    # Unfold the fiber data and time array
    fiber_data = fiber_data.unfold(0, window_size, stride)

    fiber_data = (fiber_data - fiber_data.mean()) / fiber_data.std()

    time_array = time_array.unfold(0, window_size, stride)

    # Return the unfolded fiber data and time array
    return fiber_data, time_array


def generate_latent_space(
        fiber_data: torch.Tensor,
        model: AmazingAutoEncoder,
        batch_size=8) -> np.ndarray:
    """
    Generates the latent space from the fiber data
    :param fiber_data: Fiber data
    :param model: Model
    :param batch_size: Batch size
    :return: Latent space
    """
    # Make DataLoader
    fiber_dataloader = torch.utils.data.DataLoader(fiber_data.cuda(), batch_size=batch_size, shuffle=False)

    # Initialize temp list
    temp_list = []

    # Iterate through each batch in the fiber photometry data
    for i, batch in enumerate(fiber_dataloader):
        print(f"Batch {i} of {len(fiber_dataloader)}")
        temp_list.append(model.encoder(batch.unsqueeze(-1)).squeeze(-1).cpu().detach().numpy())
    print("")

    # Stack latent space
    latent_space = np.vstack(temp_list)

    # Return the latent space
    return latent_space


def pca_latent_spaces(latent_spaces: List[np.ndarray], n_components=4) -> List[np.ndarray]:
    """
    Performs PCA on the latent spaces
    :param latent_spaces: List of latent spaces
    :param n_components: Number of components
    :return: List of PCA transformed latent spaces
    """

    pca = PCA(n_components=n_components)

    pca.fit(np.vstack(latent_spaces))

    # Create list to store PCA transformed latent spaces
    pca_transformed_latent_spaces = []

    # Transform each latent space
    for latent_space in latent_spaces:
        pca_transformed_latent_spaces.append(pca.transform(latent_space))

    # Return the PCA transformed latent spaces
    return pca_transformed_latent_spaces


def umap_latent_spaces(latent_spaces: List[np.ndarray], n_components=4) -> List[np.ndarray]:
    """
    Performs UMAP on the latent spaces
    :param latent_spaces: List of latent spaces
    :param n_components: Number of components
    :return: List of UMAP transformed latent spaces
    """

    umap = UMAP(n_components=n_components)

    # Fit UMAP on the aggregated latent spaces
    umap.fit(np.vstack(latent_spaces))

    # Create list to store UMAP transformed latent spaces
    umap_transformed_latent_spaces = []

    # Transform each latent space
    for latent_space in latent_spaces:
        umap_transformed_latent_spaces.append(umap.transform(latent_space))

    # Return the UMAP transformed latent spaces
    return umap_transformed_latent_spaces

def split_latent_space_by_events(latent_space: np.ndarray, time_array: np.ndarray, behavior_data: pd.DataFrame):
    """
    Splits the latent space by events
    :param latent_space: Latent space either PCA transformed or not
    :param time_array: Time array
    :param behavior_data: Behavior data
    :return: Event latent spaces and track names
    """
    # Loop through each event
    for i, event in behavior_data.iterrows():
        # Get the indices of the time array that are within the event
        indices = np.where((time_array[:, 0] > event["Time"])
                           & (time_array[:, 0] < event["Time"] + event["Duration"] + 0.5))[0]

        # Get the latent space for the event
        event_latent_space = latent_space[indices[0]:indices[-1]]

        # Get the track name
        track_name = event["TrackName"]

        # Yield the event latent space and track name
        yield event_latent_space, track_name


def process_data_pairs(
        data_pairs: List[Tuple[np.ndarray, float, pd.DataFrame, List[str]]],
        model: AmazingAutoEncoder,
        window_size=1000,
        stride=100,
        batch_size=8,
        pca=False,
        umap=False,
        pca_components=2) -> List[Tuple[np.ndarray, List[Tuple[np.ndarray, str]]]]:
    """
    Processes the data pairs
    :param pca_object_path: pca_object_path
    :param data_pairs: List of data pairs
    :param model: Model
    :param window_size: Window size
    :param stride: Stride
    :param batch_size: Batch size
    :param pca: Whether to perform PCA
    :param pca_components: Number of PCA components
    :return: List of data pairs with latent spaces
    """
    # Iterate through each data pair
    for fiber_data, sampling_rate, behavior_data, track_names in data_pairs:
        # Unfold the fiber data and time array
        fiber_data, time_array = unfold_fiber_data(fiber_data, sampling_rate, window_size, stride)

        # Convert time array to numpy array
        time_array = time_array.numpy()

        # Generate the latent space
        latent_space = generate_latent_space(fiber_data, model, batch_size)

        # If PCA is True
        if pca:
            # Perform PCA on the latent space
            latent_space = pca_latent_spaces([latent_space], pca_components)[0]
        elif umap:
            # Perform UMAP on the latent space
            latent_space = umap_latent_spaces([latent_space], pca_components)[0]

        # Split the latent space by events
        event_latent_spaces = list(split_latent_space_by_events(latent_space, time_array, behavior_data))

        # Yield the latent space and event latent spaces
        yield latent_space, event_latent_spaces


def plot_event_latent_spaces(
        event_latent_spaces: List[Tuple[np.ndarray, str]],
        track_names: List[str],
        output_file_path: str):
    """
    Plots the event latent spaces
    :param output_file_path: Output file path
    :param event_latent_spaces: List of event latent spaces
    :param track_names: List of track names
    :return: None
    """
    # Get the number of tracks
    n_tracks = len(track_names)

    # Get the number of combinations of dimensions
    combs = list(combinations(range(event_latent_spaces[0][0].shape[1]), 2))

    # Get the number of rows and columns and plots
    n_plots = len(combs)
    n_cols = int(math.ceil(math.sqrt(n_plots)))
    n_rows = int(math.ceil(n_plots / n_cols))

    # Create the figure
    fig = plt.figure(figsize=(20, 20))

    # Iterate through each combination of dimensions
    for idx, (l, j) in enumerate(combs):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        colors = cm.Set1(np.linspace(0, 1, n_tracks))

        for k, event in enumerate(track_names):
            x, y = [], []
            for event_latent_space, track_name in event_latent_spaces:
                if track_name == event:
                    x.append(event_latent_space[:, l])
                    y.append(event_latent_space[:, j])

            if len(x) == 0 or len(y) == 0:
                continue

            x = np.hstack(x)
            y = np.hstack(y)

            ax.scatter(x, y, c=colors[k], label=event, s=1)

            # Calculate and plot the centroid
            centroid = np.mean(np.vstack([x, y]), axis=1)
            ax.scatter(centroid[0], centroid[1], c=colors[k], s=100, zorder=3)

        ax.set_xlabel(f"Dimension {l}")
        ax.set_ylabel(f"Dimension {j}")
        legend = ax.legend()
        ax.set_title(f"Dimension {l} vs Dimension {j}")
        handles = legend.legend_handles

        for handle in handles:
            handle.set_sizes([150])

    # Save the figure
    plt.savefig(output_file_path)


def plot_event_latent_space_centroids(processed_data_pairs: List[Tuple[np.ndarray, List[Tuple[np.ndarray, str]]]],
                                      data_pairs: List[Tuple[np.ndarray, float, pd.DataFrame, List[str]]],
                                      output_folder_path: str):
    """
    Plots the event latent space centroids for each animal on the same plot,
    :param processed_data_pairs: Data pairs with latent spaces.
    :param data_pairs: Data pairs with track data.
    :param output_folder_path: Output folder path.
    """

    # Get the number of combinations of dimensions
    combs = list(combinations(range(processed_data_pairs[0][0].shape[1]), 2))

    # Get the number of rows and columns and plots
    n_plots = len(combs)
    n_cols = int(math.ceil(math.sqrt(n_plots)))
    n_rows = int(math.ceil(n_plots / n_cols))

    # Create the figure
    fig = plt.figure(figsize=(20, 20))

    # Get all unique track names
    track_names = list(set([track_name for _, _, _, track_names in data_pairs for track_name in track_names]))

    # Get the number of tracks
    n_tracks = len(track_names)

    # Iterate through each combination of dimensions
    for idx, (l, j) in enumerate(combs):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        colors = cm.Set1(np.linspace(0, 1, n_tracks))

        # Iterate through each animal
        for k, (_, event_latent_spaces), (_, _, _, track_names) in zip(range(len(processed_data_pairs)),
                                                                       processed_data_pairs, data_pairs):
            # Get the latent space for the current combination of dimensions
            for event_latent_space, track_name in event_latent_spaces:
                try:
                    x = event_latent_space[:, l]
                    y = event_latent_space[:, j]

                    ax.scatter(x, y, c=colors[track_names.index(track_name)], s=1)

                    # Calculate and plot the centroid
                    centroid = np.mean(np.vstack([x, y]), axis=1)
                    ax.scatter(centroid[0], centroid[1], c=colors[track_names.index(track_name)], s=10, zorder=3)
                except:
                    print(f"Error in dimension {l} vs dimension {j}")
                    continue

        ax.set_xlabel(f"Dimension {l}")
        ax.set_ylabel(f"Dimension {j}")
        ax.set_title(f"Dimension {l} vs Dimension {j}")

        # Create a legend
        handles = []
        for l, track_name in enumerate(track_names):
            handles.append(ax.scatter([], [], c=colors[l], label=track_name, s=100))

        legend = ax.legend()
        for handle in handles:
            handle.set_sizes([150])

    # Save the figure
    plt.savefig(os.path.join(output_folder_path, "event_latent_space_centroids.png"))


@click.command()
@click.argument("fiber_path", type=click.Path(exists=True))
@click.argument("behavior_path", type=click.Path(exists=True))
@click.argument("output_folder_path", type=click.Path(exists=False))
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--window_size", type=int, default=1000)
@click.option("--stride", type=int, default=100)
@click.option("--batch_size", type=int, default=8)
@click.option("--pca/--no-pca", default=False)
@click.option("--umap/--no-umap", default=False)
@click.option("--pca_components", type=int, default=2)
@click.option("--plot/--no-plot", default=False)
def main(fiber_path,
         behavior_path,
         output_folder_path,
         model_path,
         window_size,
         stride,
         batch_size,
         pca,
         umap,
         pca_components,
         plot):
    # Signal that the program has started
    print("Program started")

    # Create Path object
    path = Path(output_folder_path)

    # Create output folder if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

    # Load model
    model = AmazingAutoEncoder.load_from_checkpoint(model_path)

    # Get paths to each fiber block and behavior block
    path_tuples, basepaths = get_paths(fiber_path, behavior_path)

    # Get data pairs
    data_pairs = list(get_data_pairs(*zip(*path_tuples)))

    # Process data pairs
    processed_data_pairs = list(process_data_pairs(
        data_pairs,
        model,
        window_size,
        stride,
        batch_size,
        pca,
        umap,
        pca_components,
    ))

    # Plot event latent spaces
    if plot:
        # Iterate through each processed data pair and data pair to get the track names
        for (latent_space, event_latent_spaces), (_, _, _, track_names), basepath \
                in zip(processed_data_pairs, data_pairs, basepaths):
            # Create the output folder if it doesn't exist
            if not os.path.exists(os.path.join(output_folder_path, basepath)):
                os.mkdir(os.path.join(output_folder_path, basepath))

            # Plot the event latent spaces
            plot_event_latent_spaces(event_latent_spaces, track_names,
                                     os.path.join(output_folder_path, basepath, "event_latent_spaces.png"))

        # Plot event latent space centroids
        plot_event_latent_space_centroids(processed_data_pairs, data_pairs, output_folder_path)

    # Save the processed data pairs
    for (latent_space, event_latent_spaces), (_, _, _, track_names), basepath \
            in zip(processed_data_pairs, data_pairs, basepaths):
        # Create the output folder if it doesn't exist
        if not os.path.exists(os.path.join(output_folder_path, basepath)):
            os.mkdir(os.path.join(output_folder_path, basepath))

        # Save the latent event spaces
        np.save(os.path.join(output_folder_path, basepath, "latent_space.npy"), latent_space)
        np.save(os.path.join(output_folder_path, basepath, "event_latent_spaces.npy"), event_latent_spaces)


if __name__ == '__main__':
    main()
