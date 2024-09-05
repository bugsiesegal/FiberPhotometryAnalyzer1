import base64
import copy
import io
import os
from datetime import datetime, timedelta
from functools import partial
from glob import glob
from pathlib import Path
from threading import Thread, Lock

import dash_bootstrap_components as dbc
import dash_interactive_graphviz
import dash_mantine_components as dmc
import kneed
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls
import scipy.stats as stats
import seaborn as sns
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashProxy, Output, Input, State, Serverside, html, dcc, \
    ServersideOutputTransform, callback
from lightning.pytorch.trainer import Trainer
from matplotlib.colors import LogNorm
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchviz import make_dot
from tqdm import tqdm

from data import PathlessFiberTrackingDataModule
from models.lightning_models import *
from models.lightning_models import TransformerAutoencoderModule_4

app = DashProxy(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], transforms=[ServersideOutputTransform()])

class TrainingResults:
    def __init__(self):
        self.model = None
        self.is_training_complete = False
        self.lock = Lock()

class AnalysisResults:
    def __init__(self):
        self.tracking_inertia_fig = None
        self.tracking_cluster_fig = None
        self.encoded_inertia_fig = None
        self.encoded_cluster_fig = None
        self.contingency_table_fig = None
        self.chi2_contibution_fig = None
        self.signifcant_pairs_fig = None
        self.is_analysis_complete = False
        self.lock = Lock()

training_results = TrainingResults()
analysis_results = AnalysisResults()

def train_model_thread(model, data):
    trainer = Trainer(
        max_epochs=model.config.max_epochs,
        precision=model.config.precision,
        max_time=model.config.max_time,
        gradient_clip_val=0.5,
        gradient_clip_algorithm='value',
        limit_val_batches=1
    )
    datamodule = PathlessFiberTrackingDataModule(model.config, data)

    trainer.fit(model, datamodule)
    with training_results.lock:
        training_results.model = model
        training_results.is_training_complete = True

def analyze_data_thread(model, data, sampling_rate, predict_batch_size):
    fiber_config = copy.deepcopy(model.config)
    fiber_config.use_fiber = True
    fiber_config.use_tracking = False
    fiber_config.batch_size = predict_batch_size
    fiber_datamodule = PathlessFiberTrackingDataModule(fiber_config, data)
    fiber_datamodule.prepare_data()
    tracking_config = copy.deepcopy(model.config)
    tracking_config.use_fiber = False
    tracking_config.use_tracking = True
    tracking_config.batch_size = predict_batch_size
    tracking_datamodule = PathlessFiberTrackingDataModule(tracking_config, data)
    tracking_datamodule.prepare_data()
    tracking_data = []
    precision = torch.float16
    if model.config.precision == '32':
        precision = torch.float32
    elif model.config.precision == '16-mixed':
        precision = torch.float16
    elif model.config.precision == '16':
        precision = torch.float16
    for batch in tracking_datamodule.predict_dataloader():
        tracking_data.append(batch[:, 0, :].to(dtype=precision))
    tracking_data = np.concatenate(tracking_data)[::sampling_rate]
    trainer = Trainer(
        precision=model.config.precision,
    )
    encoded_data = trainer.predict(model, fiber_datamodule.predict_dataloader())
    encoded_data = np.concatenate(encoded_data)[::sampling_rate]
    # Find the optimal number of clusters for the tracking data
    kmeans = KMeans()
    inertia = []
    range_max = 150
    for k in tqdm(range(1, range_max)):
        kmeans.n_clusters = k
        kmeans.fit(tracking_data)
        inertia.append(kmeans.inertia_)
    # Find the elbow point
    kneedle = kneed.KneeLocator(range(1, range_max), inertia, curve='convex', direction='decreasing')
    print(f"Elbow Point: {kneedle.knee}")

    plt.plot(range(1, range_max), inertia)
    plt.axvline(kneedle.knee, color='red', linestyle='--')
    plt.title('Tracking Data Inertia')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    tracking_inertia_fig = tls.mpl_to_plotly(plt.gcf())
    plt.clf()

    clustering_method = partial(KMeans, n_clusters=kneedle.knee)
    tracking_clustering = clustering_method()
    tracking_clustering.fit(tracking_data)
    tracking_cluster_centers = tracking_clustering.cluster_centers_
    tracking_pca = PCA(n_components=2)
    tracking_data_2d = tracking_pca.fit_transform(tracking_data)
    tracking_cluster_centers_2d = tracking_pca.transform(tracking_cluster_centers)
    tracking_labels = tracking_clustering.labels_
    plt.figure(figsize=(10, 7))

    # First scatter plot
    scatter1 = sns.scatterplot(
        x=tracking_data_2d[:, 0],
        y=tracking_data_2d[:, 1],
        hue=tracking_labels,
        s=1,
        palette='tab20',
        legend=None  # Disable legend for the first scatter plot
    )

    # Second scatter plot for cluster centers
    scatter2 = sns.scatterplot(
        x=tracking_cluster_centers_2d[:, 0],
        y=tracking_cluster_centers_2d[:, 1],
        s=100,
        hue=range(tracking_clustering.n_clusters),
        palette='tab20',
        legend='brief'  # Enable legend for the second scatter plot
    )

    # Add the legend manually
    plt.legend(title='Cluster', loc='best')

    plt.title('Tracking Data Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    tracking_cluster_fig = tls.mpl_to_plotly(plt.gcf())
    plt.clf()

    # Find the optimal number of clusters for the encoded data
    kmeans = KMeans()
    inertia = []
    range_max = 150
    for k in tqdm(range(1, range_max)):
        kmeans.n_clusters = k
        kmeans.fit(encoded_data)
        inertia.append(kmeans.inertia_)

    # Find the elbow point
    kneedle = kneed.KneeLocator(range(1, range_max), inertia, curve='convex', direction='decreasing')
    print(f"Elbow Point: {kneedle.knee}")

    plt.plot(range(1, range_max), inertia)
    plt.axvline(kneedle.knee, color='red', linestyle='--')
    plt.title('Encoded Data Inertia')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    encoded_inertia_fig = tls.mpl_to_plotly(plt.gcf())

    clustering_method = partial(KMeans, n_clusters=kneedle.knee)
    encoded_clustering = clustering_method()
    encoded_clustering.fit(encoded_data)
    encoded_pca = PCA(n_components=2)
    encoded_pca_data = encoded_pca.fit_transform(encoded_data)
    encoded_pca_cluster_centers = encoded_pca.transform(encoded_clustering.cluster_centers_)
    encoded_labels = encoded_clustering.labels_
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=encoded_pca_data[:, 0], y=encoded_pca_data[:, 1], hue=encoded_clustering.labels_, s=1,
                    palette='tab20', legend=None)
    sns.scatterplot(x=encoded_pca_cluster_centers[:, 0], y=encoded_pca_cluster_centers[:, 1], s=100,
                    hue=range(encoded_clustering.n_clusters),
                    palette='tab20', legend='brief')
    plt.title('Encoded Data Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    encoded_cluster_fig = tls.mpl_to_plotly(plt.gcf())
    plt.clf()

    # Create a contingency table
    contingency_table = pd.crosstab(tracking_labels, encoded_labels, rownames=['Position Clusters'],
                                    colnames=['Brainwave Clusters'])
    print(contingency_table)
    # Compute Adjusted Rand Index and Normalized Mutual Information
    ari = adjusted_rand_score(tracking_labels, encoded_clustering.labels_)
    nmi = normalized_mutual_info_score(tracking_labels, encoded_clustering.labels_)
    print(f"Adjusted Rand Index: {ari}")
    print(f"Normalized Mutual Information: {nmi}")
    chi2_stat, p, dof, ex = stats.chi2_contingency(contingency_table)
    print(f"Chi-Square Test: chi2={chi2_stat}, p-value={p}, dof={dof}")
    # Plot the contingency table and the expected values
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    sns.heatmap(contingency_table, cmap='viridis', norm=LogNorm(), annot=True, fmt='.1g')
    plt.title('Contingency Table of Brainwave and Position Clusters')
    plt.xlabel('Brainwave Clusters')
    plt.ylabel('Position Clusters')
    plt.subplot(1, 2, 2)
    sns.heatmap(ex, cmap='viridis', norm=LogNorm(), annot=True, fmt='.1g')
    plt.title('Expected Values of Brainwave and Position Clusters')
    plt.xlabel('Brainwave Clusters')
    plt.ylabel('Position Clusters')
    plt.tight_layout()
    contingency_table_fig = tls.mpl_to_plotly(plt.gcf())
    plt.clf()

    # Plot the difference between the observed and expected values
    diff = contingency_table - ex
    plt.figure(figsize=(10, 7))
    sns.heatmap(diff, cmap='viridis', annot=True, fmt='.1g')
    plt.title('Difference Between Observed and Expected Values')
    plt.xlabel('Brainwave Clusters')
    plt.ylabel('Position Clusters')
    chi2_contibution_fig = tls.mpl_to_plotly(plt.gcf())
    plt.clf()

    # Calculate the contribution to the Chi-square statistic for each cell
    chi2_contributions = (contingency_table - ex) ** 2 / ex
    # Plot the contingency table with Chi-square contributions
    plt.figure(figsize=(10, 8))
    sns.heatmap(chi2_contributions, cmap='viridis', norm=LogNorm(), annot=True, fmt='.2f')
    plt.title('Chi-square Contributions for Brainwave and Position Clusters')
    plt.xlabel('Brainwave Clusters')
    plt.ylabel('Position Clusters')
    chi2_contibution_fig = tls.mpl_to_plotly(plt.gcf())
    plt.clf()

    significance_level = 0.0000001 / chi2_contributions.size
    critical_value = chi2.ppf(1 - significance_level, df=1)
    significant_pairs = chi2_contributions > critical_value
    # Plot the significant pairs
    plt.figure(figsize=(10, 8))
    sns.heatmap(significant_pairs, cmap='coolwarm', annot=True, cbar=False)
    plt.title('Significant Cluster Pairs (Bonferroni-corrected)')
    plt.xlabel('Brainwave Clusters')
    plt.ylabel('Position Clusters')
    signifcant_pairs_fig = tls.mpl_to_plotly(plt.gcf())
    plt.clf()

    with analysis_results.lock:
        analysis_results.tracking_inertia_fig = tracking_inertia_fig
        analysis_results.tracking_cluster_fig = tracking_cluster_fig
        analysis_results.encoded_inertia_fig = encoded_inertia_fig
        analysis_results.encoded_cluster_fig = encoded_cluster_fig
        analysis_results.contingency_table_fig = contingency_table_fig
        analysis_results.chi2_contibution_fig = chi2_contibution_fig
        analysis_results.signifcant_pairs_fig = signifcant_pairs_fig
        analysis_results.is_analysis_complete = True



# Utility Functions
def find_closest_row(data, column, target_value):
    # Calculate the absolute difference between each value and the target
    differences = abs(data[column] - target_value)

    # Find the index of the minimum difference
    closest_index = differences.idxmin()

    # Return the row with the closest value
    return data.loc[closest_index]

def check_gpu():
    if torch.cuda.is_available():
        return True, torch.cuda.get_device_name(0)
    else:
        return False, "No GPU available"

def load_fiber_data(path):
    # Load fiber data from csv
    fiber_df = pd.read_csv(path)
    fiber_channel = fiber_df['Analog In. | Ch.1 AIn-1 - Dem (AOut-2)']
    control_channel = fiber_df['Analog In. | Ch.1 AIn-1 - Dem (AOut-3)']
    fiber_channel -= control_channel
    fiber_df['Fiber Channel'] = fiber_channel
    return fiber_df

def load_tracking_data(path, recording_time):
    # Load tracking data from csv
    tracking_df = pd.read_csv(path)
    time_column = np.linspace(0, recording_time, len(tracking_df))
    tracking_df['Time(s)'] = time_column
    return tracking_df

def load_data(path):
    # Load fiber and tracking data
    fiber_paths = glob(os.path.join(path, '**/Fiber/*.csv'), recursive=True)
    tracking_paths = glob(os.path.join(path, '**/Tracking/*.csv'), recursive=True)
    data = []
    for fiber_path, tracking_path in zip(fiber_paths, tracking_paths):
        fiber_data = load_fiber_data(fiber_path)
        tracking_data = load_tracking_data(tracking_path, max(fiber_data['Time(s)']))
        data.append((fiber_data, tracking_data))
    return data

def mouse_plot(data, time):
    pos_columns = [col for col in data.columns if '_x' in col or '_y' in col or '_z' in col]
    fig = go.Figure()
    pos_row = find_closest_row(data, 'Time(s)', time)

    # Calculate overall min and max for each axis
    x_min, x_max = data[[col for col in pos_columns if '_x' in col]].min().min(), data[[col for col in pos_columns if '_x' in col]].max().max()
    y_min, y_max = data[[col for col in pos_columns if '_y' in col]].min().min(), data[[col for col in pos_columns if '_y' in col]].max().max()
    z_min, z_max = data[[col for col in pos_columns if '_z' in col]].min().min(), data[[col for col in pos_columns if '_z' in col]].max().max()

    for i in range(0, len(pos_columns), 3):
        x = pos_row[pos_columns[i]]
        y = pos_row[pos_columns[i + 1]]
        z = pos_row[pos_columns[i + 2]]
        point_name = pos_columns[i].split('_x')[0]

        trace = go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=1),
            name=point_name,
            text=[point_name],
            textposition="top center",
            textfont=dict(size=8)
        )
        fig.add_trace(trace)

    # Set the axis ranges to be constant
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            zaxis=dict(range=[z_min, z_max]),
            aspectmode='cube'  # This ensures the plot is a cube and not stretched
        )
    )

    return fig

# Progress Stepper Steps
steps = [
    dmc.StepperStep(
        label="First Step",
        description="Load data",
        children=dmc.Text("Step 1: Load data from drive or connect to fiber rig.", size="sm", c="dimmed")
    ),
    dmc.StepperStep(
        label="Second Step",
        description="Load Model",
        children=dmc.Text("Step 2: Create and train a new model or load a pre-trained model", size="sm", c="dimmed")
    ),
    dmc.StepperStep(
        label="Third Step",
        description="Encode Data",
        children=dmc.Text("Step 3: Encode the fiber photometry data using the model", size="sm", c="dimmed")
    ),
    dmc.StepperStep(
        label="Fourth Step",
        description="Cluster Data",
        children=dmc.Text("Step 4: Cluster the encoded data and tracking data", size="sm", c="dimmed")
    ),
    dmc.StepperStep(
        label="Fifth Step",
        description="Analyze Data",
        children=dmc.Text("Step 5: Analyze the clustered data", size="sm", c="dimmed")
    ),
]

# Progress Stepper
progress_stepper = dmc.Stepper(
    id="progress-stepper",
    active=0,
    children=steps,
    size="sm",
    radius="md",
)

# Gpu Status Card
gpu_status_card = dmc.AspectRatio(dmc.Paper(
    children=dmc.Stack(
        [
            dmc.Text("GPU Status", size="lg"),
            dmc.Text(id="gpu-connection-status", fw=1000, mt="md", size="xl"),
            dmc.Text(id="gpu-name", size="xs", c="dimmed", mt="xs"),
        ],
        gap="xs",
        align="center",
        justify="center",
        style={"textAlign": "center"},
    ),
    id="gpu-status-card",
    shadow="sm",
    radius="lg",
    withBorder=True,
    style={"border": "1px solid red"},
), ratio=1)

# Data Status Card
data_status_card = dmc.AspectRatio(dmc.Paper(
    children=dmc.Stack(
        [
            dmc.Text("Data Status", size="lg"),
            dmc.Text(id="data-loading-status", fw=1000, mt="md", size="xl"),
        ],
        gap="xs",
        align="center",
        justify="center",
        style={"textAlign": "center"},
    ),
    id="data-status-card",
    shadow="sm",
    radius="lg",
    withBorder=True,
), ratio=1)

# Home Status Cards
home_status_cards = dmc.SimpleGrid(
    cols={"base": 2, "sm": 3, "md": 4, "lg": 6},
    spacing="md",
    verticalSpacing="md",
    children=[
        gpu_status_card,
        data_status_card,
    ],
    mt="xl",
)

# Home Tab
home_tab = dmc.TabsPanel(
    children=[
        dmc.Container(
            [
                dmc.Title("Project Progress", order=2, mb="md"),
                progress_stepper,
                dmc.Divider(variant="solid", my="xl"),
                home_status_cards
            ],
            size="lg",
            px="xl",
            py="xl",
        )

    ],
    value="home"
)

# Load Data Button Group
load_data_button = dmc.Button(
    "Load Data",
    color="blue",
    size="md",
    radius="sm",
    fullWidth=True,
    mb="md",
    id="load-data-button",
    variant="outline",
)

# Data Tab
data_tab = dmc.TabsPanel(
    children=[
        dmc.Container(
            [
                dmc.Title("Data", order=2, mb="xs"),
                dmc.Text("Load data from drive or connect to fiber rig.", size="lg", c="dimmed", mt="xs"),
                load_data_button,
                dmc.Card(html.Div(
                    [
                        dcc.Graph(id="fiber-data-viz", style={"display": "inline-block", "width": "50%"}),
                        dcc.Graph(id="tracking-data-viz", style={"display": "inline-block", "width": "50%"}),
                    ]
                ), shadow="sm", radius="lg", mb="xl"),
                dmc.Center(dmc.Pagination(id="fiber-data-pagination", total=1, value=1)),
            ],
            size="lg",
            px="xl",
            py="xl",
        )

    ],
    value="data"
)

# Model Creation Form
window_size_input = dbc.Row(
    [
        dbc.Label("Window Size", html_for="window-size", width=2),
        dbc.Col(
            dbc.Input(id="window-size", type="number", placeholder="Enter Window Size"),
            width=10,
        ),
    ]
)

d_model_input = dbc.Row(
    [
        dbc.Label("D Model", html_for="d-model", width=2),
        dbc.Col(
            dbc.Input(id="d-model", type="number", placeholder="Enter D Model"),
            width=10,
        ),
    ]
)

nhead_input = dbc.Row(
    [
        dbc.Label("Num Heads", html_for="n-head", width=2),
        dbc.Col(
            dbc.Input(id="n-head", type="number", placeholder="Enter Num Heads"),
            width=10,
        ),
    ]
)

dim_feedforward_input = dbc.Row(
    [
        dbc.Label("Dim Feedforward", html_for="dim-feedforward", width=2),
        dbc.Col(
            dbc.Input(id="dim-feedforward", type="number", placeholder="Enter Dim Feedforward"),
            width=10,
        ),
    ]
)

num_layers_input = dbc.Row(
    [
        dbc.Label("Num Layers", html_for="num-layers", width=2),
        dbc.Col(
            dbc.Input(id="num-layers", type="number", placeholder="Enter Num Layers"),
            width=10,
        ),
    ]
)

dropout_input = dbc.Row(
    [
        dbc.Label("Dropout", html_for="dropout", width=2),
        dbc.Col(
            dbc.Input(id="dropout", type="number", placeholder="Enter Dropout"),
            width=10,
        ),
    ]
)

activation_input = dbc.Row(
    [
        dbc.Label("Activation", html_for="activation", width=2),
        dbc.Col(
            dcc.Dropdown(
                id="activation",
                options=[
                    {"label": "Linear", "value": "linear"},
                    {"label": "ReLU", "value": "relu"},
                    {"label": "Tanh", "value": "tanh"},
                    {"label": "Sigmoid", "value": "sigmoid"},
                ],
                value="relu",
            ),
            width=10,
        ),
    ]
)

latent_dim_input = dbc.Row(
    [
        dbc.Label("Latent Dim", html_for="latent-dim", width=2),
        dbc.Col(
            dbc.Input(id="latent-dim", type="number", placeholder="Enter Latent Dim"),
            width=10,
        ),
    ]
)

use_positional_encoding_input = dbc.Row(
    [
        dbc.Label("Use Positional Encoding", html_for="use-positional-encoding", width=2),
        dbc.Col(
            dbc.Checkbox(id="use-positional-encoding", value=True),
            width=10,
        ),
    ]
)

normalization_input = dbc.Row(
    [
        dbc.Label("Normalization", html_for="normalization", width=2),
        dbc.Col(
            dcc.Dropdown(
                id="normalization",
                options=[
                    {"label": "Min-Max", "value": "min-max"},
                    {"label": "Standard", "value": "standard"},
                    {"label": "Robust", "value": "robust"},
                    {"label": "Max Abs", "value": "max-abs"},
                    {"label": "Quantile", "value": "quantile"},
                    {"label": "Power", "value": "power"},
                ],
                value="min-max",
            ),
            width=10,
        ),
    ]
)

model_selection_input = dbc.Row(
    [
        dbc.Label("Model Selection", html_for="model-selection", width=2),
        dbc.Col(
            dcc.Dropdown(
                id="model-selection",
                options=[
                    {"label": "Transformer V1", "value": "transformer_v1"},
                    {"label": "Transformer V2", "value": "transformer_v2"},
                    {"label": "Transformer V3", "value": "transformer_v3"},
                    {"label": "Transformer V4", "value": "transformer_v4"},
                ],
                value="transformer_v1",
            ),
            width=10,
        ),
    ]
)


generate_model_button = dmc.ButtonGroup(
    [
        dmc.Button(
            "Generate Model",
            color="blue",
            size="md",
            radius="sm",
            fullWidth=True,
            mt="md",
            id="generate-model-button",
            variant="outline",
        ),
        dcc.Upload(dmc.Button(
            "Load Model",
            color="blue",
            size="md",
            radius="sm",
            fullWidth=True,
            mt="md",
            variant="outline",
        ), id="load-model-button")
    ]
)

model_form = dmc.Card(dbc.Form(
    [
        window_size_input,
        d_model_input,
        nhead_input,
        dim_feedforward_input,
        num_layers_input,
        dropout_input,
        activation_input,
        latent_dim_input,
        use_positional_encoding_input,
        normalization_input,
        model_selection_input,
        generate_model_button,
    ],
    id="model-form",
))

model_plot = dmc.Card(
    dash_interactive_graphviz.DashInteractiveGraphviz(
        id="model-plot",
    ),
    shadow="sm",
    radius="lg",
    mb="xl",
    h="50vh",
)

# Model Tab
model_tab = dmc.TabsPanel(
    [
        model_plot,
        model_form
    ],
    value="model"
)

# Training Form Inputs
learning_rate_input = dbc.Row(
    [
        dbc.Label("Learning Rate", html_for="learning-rate", width=2),
        dbc.Col(
            dbc.Input(id="learning-rate", type="number", placeholder="Enter Learning Rate"),
            width=10,
        ),
    ]
)

lr_factor_input = dbc.Row(
    [
        dbc.Label("LR Factor", html_for="lr-factor", width=2),
        dbc.Col(
            dbc.Input(id="lr-factor", type="number", placeholder="Enter LR Factor"),
            width=10,
        ),
    ]
)

lr_patience_input = dbc.Row(
    [
        dbc.Label("LR Patience", html_for="lr-patience", width=2),
        dbc.Col(
            dbc.Input(id="lr-patience", type="number", placeholder="Enter LR Patience"),
            width=10,
        ),
    ]
)

max_epochs_input = dbc.Row(
    [
        dbc.Label("Max Epochs", html_for="max-epochs", width=2),
        dbc.Col(
            dbc.Input(id="max-epochs", type="number", placeholder="Enter Max Epochs"),
            width=10,
        ),
    ]
)

batch_size_input = dbc.Row(
    [
        dbc.Label("Batch Size", html_for="batch-size", width=2),
        dbc.Col(
            dbc.Input(id="batch-size", type="number", placeholder="Enter Batch Size"),
            width=10,
        ),
    ]
)

num_workers_input = dbc.Row(
    [
        dbc.Label("Num Workers", html_for="num-workers", width=2),
        dbc.Col(
            dbc.Input(id="num-workers", type="number", placeholder="Enter Num Workers"),
            width=10,
        ),
    ]
)

precision_input = dbc.Row(
    [
        dbc.Label("Precision", html_for="precision", width=2),
        dbc.Col(
            dcc.Dropdown(
                id="precision",
                options=[
                    {"label": "16-mixed", "value": "16-mixed"},
                    {"label": "16", "value": "16"},
                    {"label": "32", "value": "32"},
                ],
                value="32",
            ),
            width=10,
        ),
    ]
)

seconds_time_input = dbc.Row(
    [
        dbc.Label("Seconds", html_for="seconds-time", width=2),
        dbc.Col(
            dbc.Input(id="seconds-time", type="number", placeholder="Enter Seconds"),
            width=10,
        ),
    ]
)

minutes_time_input = dbc.Row(
    [
        dbc.Label("Minutes", html_for="minutes-time", width=2),
        dbc.Col(
            dbc.Input(id="minutes-time", type="number", placeholder="Enter Minutes"),
            width=10,
        ),
    ]
)

hours_time_input = dbc.Row(
    [
        dbc.Label("Hours", html_for="hours-time", width=2),
        dbc.Col(
            dbc.Input(id="hours-time", type="number", placeholder="Enter Hours"),
            width=10,
        ),
    ]
)

days_time_input = dbc.Row(
    [
        dbc.Label("Days", html_for="days-time", width=2),
        dbc.Col(
            dbc.Input(id="days-time", type="number", placeholder="Enter Days"),
            width=10,
        ),
    ]
)

train_button = dmc.Button(
    "Train Model",
    color="blue",
    size="md",
    radius="sm",
    fullWidth=True,
    mt="md",
    id="train-model-button",
    variant="outline",
)

training_form = dmc.Card(dbc.Form(
    [
        learning_rate_input,
        lr_factor_input,
        lr_patience_input,
        max_epochs_input,
        batch_size_input,
        num_workers_input,
        precision_input,
        seconds_time_input,
        minutes_time_input,
        hours_time_input,
        days_time_input,
        train_button,
    ],
    id="training-form",
))

training_progress_bar = dmc.Progress(
    value=0,
    color="blue",
    size="md",
    radius="sm",
    mt="md",
    id="training-progress-bar",
)

time_remaining = dmc.Text("Time Remaining: NaN", size="md", c="dimmed", mt="md", id="time-remaining")

training_progress = dmc.Card(
    [
        training_progress_bar,
        time_remaining,
    ],
    shadow="sm",
    radius="lg",
    mb="xl",
)

# Training Tab
training_tab = dmc.TabsPanel(
    [
        training_form,
        training_progress
    ],
    value="training"
)

# Start Analysis Button
start_analysis_button = dmc.Button(
    "Start Analysis",
    color="blue",
    size="md",
    radius="sm",
    fullWidth=True,
    mt="md",
    id="start-analysis-button",
    variant="outline",
)

# Analysis Form Inputs
sampling_rate_input = dbc.Row(
    [
        dbc.Label("Sampling Rate", html_for="sampling-rate", width=2),
        dbc.Col(
            dbc.Input(id="sampling-rate", type="number", placeholder="Enter Sampling Rate"),
            width=10,
        ),
    ]
)

predict_batch_size_input = dbc.Row(
    [
        dbc.Label("Predict Batch Size", html_for="predict-batch-size", width=2),
        dbc.Col(
            dbc.Input(id="predict-batch-size", type="number", placeholder="Enter Predict Batch Size"),
            width=10,
        ),
    ]
)

analysis_form = dmc.Card(dbc.Form(
    [
        sampling_rate_input,
        predict_batch_size_input,
    ],
    id="analysis-form",
))

# Analysis Graphs
tracking_inertia_graph = dmc.Card(dcc.Graph(id="tracking-inertia-graph"), mb="md", shadow="sm", radius="lg")
encoded_inertia_graph = dmc.Card(dcc.Graph(id="encoded-inertia-graph"), mb="md", shadow="sm", radius="lg")
tracking_cluster_graph = dmc.Card(dcc.Graph(id="tracking-cluster-graph"), mb="md", shadow="sm", radius="lg")
encoded_cluster_graph = dmc.Card(dcc.Graph(id="encoded-cluster-graph"), mb="md", shadow="sm", radius="lg")
chi2_contibution_graph = dmc.Card(dcc.Graph(id="chi2-contibution-graph"), mb="md", shadow="sm", radius="lg")
contingency_table_graph = dmc.Card(dcc.Graph(id="contingency-table-graph"), mb="md", shadow="sm", radius="lg")
signifcant_pairs_graph = dmc.Card(dcc.Graph(id="signifcant-pairs-graph"), mb="md", shadow="sm", radius="lg")

graph_grid = dmc.SimpleGrid(
    cols={"base": 1, "sm": 1, "md": 2, "lg": 3},
    spacing="md",
    verticalSpacing="md",
    children=[
        tracking_inertia_graph,
        encoded_inertia_graph,
        tracking_cluster_graph,
        encoded_cluster_graph,
        chi2_contibution_graph,
        contingency_table_graph,
        signifcant_pairs_graph,
    ],
    mt="xl",
)

# Analysis Tab
analysis_tab = dmc.TabsPanel(
    [
        analysis_form,
        start_analysis_button,
        graph_grid
    ],
    value="analysis"
)

# Define tabs
tabs_list = dmc.TabsList([
    dmc.TabsTab("Home", value="home"),
    dmc.TabsTab("Data", value="data"),
    dmc.TabsTab("Model", value="model"),
    dmc.TabsTab("Training", value="training"),
    dmc.TabsTab("Analysis", value="analysis"),
    dmc.TabsTab("Fiber Rig", value="fiber_rig", disabled=True),
    dmc.TabsTab("Settings", value="settings", ml="auto"),
])

app.layout = dmc.MantineProvider(
    [
        dmc.Paper(
            dmc.Tabs(
                [
                    tabs_list,
                    home_tab,
                    data_tab,
                    model_tab,
                    training_tab,
                    analysis_tab,
                ],
                color="blue",
                orientation="horizontal",
                value="home",
            ),
            p="md",
            shadow="sm",
            radius="lg",
            withBorder=True,
        ),
        dcc.Interval(id="interval-component", interval=10000, n_intervals=0),
        dcc.Store(id="data-store", storage_type="session"),
        dcc.Store(id="model-store", storage_type="session"),
        dcc.Store(id='training-start-time', storage_type='local'),
        dcc.Store(id='training-max-time', storage_type='local'),
    ]
)

# Callbacks
@callback(
    Output("gpu-connection-status", "children"),
    Output("gpu-connection-status", "c"),
    Output("gpu-name", "children"),
    Output("gpu-status-card", "style"),
    Input("interval-component", "n_intervals")
)
def update_gpu_status(n):
    is_available, gpu_name = check_gpu()
    if is_available:
        return "Connected", "green", gpu_name, {"border": "1px solid green"}
    else:
        return "Disconnected", "red", gpu_name, {"border": "1px solid red"}

@callback(
    Output("data-loading-status", "children"),
    Output("data-loading-status", "c"),
    Output("data-status-card", "style"),
    Input("data-store", "data"),
    Input("interval-component", "n_intervals"),
)
def update_data_status(data, n):
    if data is None:
        return "Not Loaded", "red", {"border": "1px solid red"}
    else:
        return "Loaded", "green", {"border": "1px solid green"}

@callback(
    Output("data-store", "data"),
    Output("progress-stepper", "active"),
    Output("fiber-data-pagination", "total", allow_duplicate=True),
    Input("load-data-button", "n_clicks"),
    State("data-store", "data"),
    State("progress-stepper", "active"),
    prevent_initial_call=True
)
def update_data(n_clicks, data, active_step):
    if n_clicks is None:
        raise PreventUpdate
    if data is not None:
        return Serverside(data), 1 if active_step == 0 else active_step, len(data)
    data = load_data(os.path.join(Path(__file__).parent.parent, "datafiles"))
    return Serverside(data), 1 if active_step == 0 else active_step, len(data)

@callback(
    Output("fiber-data-viz", "figure"),
    Input("fiber-data-pagination", "value"),
    Input("data-store", "data"),
)
def update_fiber_data_viz(page, data):
    if data is None:
        raise PreventUpdate
    fiber_data, tracking_data = data[page - 1]
    fiber_data_viz = px.line(fiber_data, x='Time(s)', y='Fiber Channel', title='Fiber Data Visualization')
    return fiber_data_viz

@callback(
    Output("tracking-data-viz", "figure"),
    Input("fiber-data-viz", "hoverData"),
    State("fiber-data-pagination", "value"),
    State("data-store", "data"),
)
def update_tracking_data_viz(hover_data, page, data):
    if data is None or hover_data is None:
        raise PreventUpdate
    fiber_data, tracking_data = data[page - 1]
    tracking_data_viz = mouse_plot(tracking_data, hover_data['points'][0]['x'])
    return tracking_data_viz

@callback(
    Output("fiber-data-pagination", "total", allow_duplicate=True),
    Input("data-store", "data"),
)
def update_fiber_data_pagination(data):
    if data is None:
        raise PreventUpdate
    return len(data)

@callback(
    Output("model-store", "data"),
    Output("progress-stepper", "active", allow_duplicate=True),
    Output("model-plot", "dot_source", allow_duplicate=True),
    Input("generate-model-button", "n_clicks"),
    State("window-size", "value"),
    State("d-model", "value"),
    State("n-head", "value"),
    State("dim-feedforward", "value"),
    State("num-layers", "value"),
    State("dropout", "value"),
    State("activation", "value"),
    State("latent-dim", "value"),
    State("use-positional-encoding", "value"),
    State("normalization", "value"),
    State("model-selection", "value"),
    State("progress-stepper", "active"),
    prevent_initial_call=True
)
def generate_model(n_clicks, window_size, d_model, n_head, dim_feedforward, num_layers, dropout, activation, latent_dim,
                    use_positional_encoding, normalization, model_selection, active_step):
    if n_clicks is None:
        raise PreventUpdate
    config = Config()
    config.window_dim = window_size
    config.d_model = d_model
    config.num_heads = n_head
    config.dim_feedforward = dim_feedforward
    config.num_layers = num_layers
    config.dropout = dropout
    config.activation = activation
    config.latent_dim = latent_dim
    config.use_positional_encoding = use_positional_encoding
    config.normalization = normalization
    config.model = model_selection

    if config.model == "transformer_v1":
        model = TransformerAutoencoderModule_1(config)
    elif config.model == "transformer_v2":
        model = TransformerAutoencoderModule_2(config)
    elif config.model == "transformer_v3":
        model = TransformerAutoencoderModule_3(config)
    elif config.model == "transformer_v4":
        model = TransformerAutoencoderModule_4(config)
    else:
        raise ValueError(f'Invalid model: {config.model}')

    dot = make_dot(model.model, params=dict(model.model.named_parameters()))
    dot.graph_attr['rankdir'] = 'LR'
    return Serverside(model), 2 if active_step < 1 else active_step, dot.source

@callback(
    Output("model-store", "data", allow_duplicate=True),
    Output("progress-stepper", "active", allow_duplicate=True),
    Output("window-size", "value"),
    Output("d-model", "value"),
    Output("n-head", "value"),
    Output("dim-feedforward", "value"),
    Output("num-layers", "value"),
    Output("dropout", "value"),
    Output("activation", "value"),
    Output("latent-dim", "value"),
    Output("use-positional-encoding", "value"),
    Output("normalization", "value"),
    Output("model-selection", "value"),
    Output("model-plot", "dot_source", allow_duplicate=True),
    Input("load-model-button", "contents"),
    State("progress-stepper", "active"),
    prevent_initial_call=True
)
def load_model(contents, active_step):
    if contents is None:
        raise PreventUpdate
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    model = torch.load(io.BytesIO(decoded))
    window_size = model.config.window_dim
    d_model = model.config.d_model
    n_head = model.config.nhead
    dim_feedforward = model.config.dim_feedforward
    num_layers = model.config.num_layers
    dropout = model.config.dropout
    activation = model.config.activation
    latent_dim = model.config.latent_dim
    use_positional_encoding = model.config.use_positional_encoding
    normalization = model.config.normalization
    model_selection = model.config.model
    example_output = model.model(torch.randn(1, window_size, model.config.input_features).to(model.device))
    dot = make_dot(example_output, params=dict(model.model.named_parameters()))
    dot.graph_attr['rankdir'] = 'LR'
    return Serverside(model), 2 if active_step < 1 else active_step, window_size, d_model, n_head, dim_feedforward, num_layers, dropout, activation, latent_dim, use_positional_encoding, normalization, model_selection, dot.source

@callback(
    Output("model-plot", "dot_source", allow_duplicate=True),
    Input("model-store", "data"),
)
def update_model_plot(model):
    if model is None:
        raise PreventUpdate
    example_output = model.model(torch.randn(1, model.config.window_dim, model.config.input_features).to(model.device))
    dot = make_dot(example_output, params=dict(model.model.named_parameters()))
    dot.graph_attr['rankdir'] = 'LR'
    return dot.source

@callback(
    Output("progress-stepper", "active", allow_duplicate=True),
    Output('train-model-button', 'loading', allow_duplicate=True),
    Output('training-start-time', 'data', allow_duplicate=True),
    Output('training-max-time', 'data', allow_duplicate=True),
    Input("train-model-button", "n_clicks"),
    State("model-store", "data"),
    State("learning-rate", "value"),
    State("lr-factor", "value"),
    State("lr-patience", "value"),
    State("max-epochs", "value"),
    State("batch-size", "value"),
    State("num-workers", "value"),
    State("seconds-time", "value"),
    State("minutes-time", "value"),
    State("hours-time", "value"),
    State("days-time", "value"),
    State("precision", "value"),
    State("data-store", "data"),
    State("progress-stepper", "active"),
    prevent_initial_call=True
)
def train_model(n_clicks, model, learning_rate, lr_factor, lr_patience, max_epochs, batch_size, num_workers, seconds, minutes, hours, days, precision, datamodule, active_step):
    if n_clicks is None:
        raise PreventUpdate
    model.config.learning_rate = learning_rate
    model.config.lr_factor = lr_factor
    model.config.lr_patience = lr_patience
    model.config.max_epochs = max_epochs
    model.config.batch_size = batch_size
    model.config.num_workers = num_workers
    model.config.max_time = {'days': days, 'hours': hours, 'minutes': minutes, 'seconds': seconds}
    model.config.precision = precision
    thread = Thread(target=train_model_thread, args=(model, datamodule))
    thread.start()
    start_time = datetime.now()
    return 3 if active_step < 2 else active_step, True, Serverside(start_time), model.config.max_time

@callback(
    Output("train-model-button", "loading", allow_duplicate=True),
    Output("training-progress-bar", "value", allow_duplicate=True),
    Output("time-remaining", "children", allow_duplicate=True),
    Output("model-store", "data", allow_duplicate=True),
    Input("interval-component", "n_intervals"),
    State("model-store", "data"),
    State("training-start-time", "data"),
    State("training-max-time", "data"),
)
def update_training_progress(n, model, start_time, max_time):
    if model is None or start_time is None or max_time is None:
        raise PreventUpdate

    with training_results.lock:
        if training_results.is_training_complete:
            return False, 100, "Training Complete", Serverside(training_results.model)

    # Convert start_time to datetime if it's not already
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)

    # Calculate total_seconds for max_time
    max_seconds = (
        max_time['days'] * 86400 +
        max_time['hours'] * 3600 +
        max_time['minutes'] * 60 +
        max_time['seconds']
    )

    elapsed_time = datetime.now() - start_time
    elapsed_seconds = elapsed_time.total_seconds()

    if elapsed_seconds >= max_seconds:
        return False, 100, "Time Limit Reached", Serverside(model)

    percent_complete = min(100, (elapsed_seconds / max_seconds) * 100)

    remaining_seconds = max(0, max_seconds - elapsed_seconds)
    remaining_time = timedelta(seconds=int(remaining_seconds))

    days, remainder = divmod(remaining_time.total_seconds(), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_left = f"Time Remaining: {int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"

    return True, int(percent_complete), time_left, Serverside(model)

@callback(
    Output("start-analysis-button", "loading", allow_duplicate=True),
    Input("start-analysis-button", "n_clicks"),
    State("model-store", "data"),
    State("data-store", "data"),
    State("sampling-rate", "value"),
    State("predict-batch-size", "value"),
    prevent_initial_call=True
)
def start_analysis(n_clicks, model, data, sampling_rate, predict_batch_size):
    if n_clicks is None or model is None or data is None:
        raise PreventUpdate

    thread = Thread(target=analyze_data_thread, args=(model, data, sampling_rate, predict_batch_size))
    thread.start()

    return True

@callback(
    Output("tracking-inertia-graph", "figure"),
    Output("tracking-cluster-graph", "figure"),
    Output("encoded-inertia-graph", "figure"),
    Output("encoded-cluster-graph", "figure"),
    Output("contingency-table-graph", "figure"),
    Output("chi2-contibution-graph", "figure"),
    Output("signifcant-pairs-graph", "figure"),
    Output("start-analysis-button", "loading"),
    Input("interval-component", "n_intervals"),
)
def update_analysis_graphs(n):
    with analysis_results.lock:
        if analysis_results.is_analysis_complete:
            tracking_inertia_fig = analysis_results.tracking_inertia_fig
            tracking_cluster_fig = analysis_results.tracking_cluster_fig
            encoded_inertia_fig = analysis_results.encoded_inertia_fig
            encoded_cluster_fig = analysis_results.encoded_cluster_fig
            contingency_table_fig = analysis_results.contingency_table_fig
            chi2_contibution_fig = analysis_results.chi2_contibution_fig
            signifcant_pairs_fig = analysis_results.signifcant_pairs_fig

            return tracking_inertia_fig, tracking_cluster_fig, encoded_inertia_fig, encoded_cluster_fig, contingency_table_fig, chi2_contibution_fig, signifcant_pairs_fig, False
        else:
            raise PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True)