import base64
import io
from typing import List, Tuple, Dict, Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State

# Constants
FIBER_CHANNEL_NAME = "Analog In. | Ch.1 AIn-1 - Dem (AOut-2)"
CONTROL_CHANNEL_NAME = "Analog In. | Ch.1 AIn-1 - Dem (AOut-3)"
PRIMARY_COLOR = "#007bff"
SECONDARY_COLOR = "#6c757d"
BACKGROUND_COLOR = "#f8f9fa"

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])

# Load Custom Index
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: ''' + BACKGROUND_COLOR + ''';
                height: 100vh;
                margin: 0;
                padding: 0;
            }
            #root {
                height: 100%;
            }
            .upload-area {
                border: 2px dashed ''' + PRIMARY_COLOR + ''';
                border-radius: 5px;
                padding: 10px;
                text-align: center;
                cursor: pointer;
            }
            .nav-buttons {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .nav-buttons button {
                margin: 0 5px;
            }
            .graph-container {
                height: calc(100vh - 250px);
                min-height: 400px;
            }
            .graph-card {
                height: 100%;
            }
            .graph-card-body {
                height: calc(100% - 40px);
                padding: 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Helper functions
def process_fiber_data(fiber_data: pd.DataFrame) -> np.ndarray:
    """Process fiber photometry data."""
    fiber_channel = fiber_data[FIBER_CHANNEL_NAME].to_numpy()
    control_channel = fiber_data[CONTROL_CHANNEL_NAME].to_numpy()
    return fiber_channel - control_channel

def process_tracking_data(tracking_data: pd.DataFrame) -> np.ndarray:
    """Process tracking data."""
    column_names = tracking_data.columns
    x_columns = [column for column in column_names if '_x' in column]
    y_columns = [column for column in column_names if '_y' in column]
    z_columns = [column for column in column_names if '_z' in column]
    x = tracking_data[x_columns].to_numpy()
    y = tracking_data[y_columns].to_numpy()
    z = tracking_data[z_columns].to_numpy()
    return np.concatenate([x, y, z], axis=1)

def parse_contents(contents: str, filename: str) -> Tuple[pd.DataFrame, str]:
    """Parse uploaded file contents."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df, filename

def plot_fiber_data(fiber_data: pd.DataFrame, filename: str) -> go.Figure:
    """Create a plot for fiber photometry data."""
    fiber_data = pd.DataFrame({
        'Time (Seconds)': fiber_data["Time(s)"],
        'Fiber': fiber_data[FIBER_CHANNEL_NAME] - fiber_data[CONTROL_CHANNEL_NAME]
    })
    fig = px.line(x="Time (Seconds)", y="Fiber", data_frame=fiber_data)
    fig.update_layout(
        title=filename,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font_color=SECONDARY_COLOR,
        margin=dict(l=50, r=50, t=30, b=30),
        autosize=True,
    )
    return fig

def plot_tracking_data(tracking_data: pd.DataFrame, filename: str) -> go.Figure:
    """Create a 3D plot for tracking data."""
    tracking_data = tracking_data.filter(regex='.*(_x|_y|_z).*')

    connections = [
        (0, 3), (3, 6), (6, 4), (4, 5), (5, 7), (7, 8),  # Head and body
        (0, 1), (0, 2),  # Nose to right and left ear
        (9, 10), (10, 5),  # Left Front Leg
        (11, 12), (12, 5),  # Right Front Leg
        (13, 14), (14, 6),  # Left Hind Leg
        (15, 16), (16, 6)  # Right Hind Leg
    ]

    fig = go.Figure()
    num_frames = len(tracking_data)
    normalized_frames = np.linspace(0, 1, num_frames)

    for frame, norm_frame in enumerate(normalized_frames):
        color = px.colors.sample_colorscale('Viridis', norm_frame)
        for start, end in connections:
            fig.add_trace(go.Scatter3d(
                x=[tracking_data.iloc[frame, start * 3], tracking_data.iloc[frame, end * 3]],
                z=[-tracking_data.iloc[frame, start * 3 + 1], -tracking_data.iloc[frame, end * 3 + 1]],
                y=[tracking_data.iloc[frame, start * 3 + 2], tracking_data.iloc[frame, end * 3 + 2]],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                opacity=0.6
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        title=filename,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font_color=SECONDARY_COLOR,
        margin=dict(l=0, r=0, t=30, b=0),
        autosize=True,
    )

    # Add color bar to show time progression
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Time', thickness=10, len=0.6, y=0.5)
        ),
        showlegend=False
    ))

    return fig

def process_upload(
        list_of_contents: List[str],
        list_of_names: List[str],
        data_dict: Dict[str, Any],
        figures: List[go.Figure],
        tracking_figures: List[go.Figure]
) -> Tuple[Dict[str, Any], List[go.Figure], List[go.Figure]]:
    """Process uploaded files and generate figures."""
    if list_of_contents is not None:
        for contents, filename in zip(list_of_contents, list_of_names):
            df, filename = parse_contents(contents, filename)
            if "fiber-" in filename:
                figures.append(plot_fiber_data(df[::10], filename))
                fiber_data = process_fiber_data(df)
                data_dict[filename.strip(".csv").strip("fiber-")] = fiber_data.tolist()
            elif "tracking-" in filename:
                tracking_figures.append(plot_tracking_data(df[::1000], filename))
                tracking_data = process_tracking_data(df)
                data_dict[filename.strip(".csv").strip("tracking-")] = tracking_data.tolist()
    return data_dict, figures, tracking_figures

def update_figure(figures: List[go.Figure], current_figure_index: int) -> go.Figure:
    """Update the displayed figure based on the current index."""
    return figures[current_figure_index]

def create_collapsible(title, content, id_prefix):
    """Create a collapsible card component."""
    return dbc.Card([
        dbc.CardHeader(
            html.Div([
                html.I(className="fas fa-chevron-down mr-2", id=f"{id_prefix}-chevron"),
                html.H5(title, className="mb-0 d-inline")
            ], id=f"{id_prefix}-header", style={"cursor": "pointer"}),
        ),
        dbc.Collapse(
            dbc.CardBody(content),
            id=f"{id_prefix}-collapse",
            is_open=True
        )
    ], className="mb-3")

# Layout components
def create_upload_area():
    """Create the upload area component."""
    return dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files', style={"color": PRIMARY_COLOR})
        ]),
        className='upload-area mb-2',
        multiple=True
    )

def create_graph_card(title, graph_id):
    """Create a card containing a graph."""
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody([
            dcc.Loading(dcc.Graph(id=graph_id, className='graph-container'))
        ], className="graph-card-body")
    ], className="h-100 mb-3")

def create_navigation_buttons():
    """Create navigation buttons for switching between graphs."""
    return html.Div([
        dbc.Button("Previous", id="prev-button", color="primary", size="sm", className="mr-1"),
        html.Span(id='current-figure-index-output', className="mx-1"),
        dbc.Button("Next", id="next-button", color="primary", size="sm", className="ml-1")
    ], className="nav-buttons")

def create_instructions():
    """Create instructions for using the dashboard."""
    return html.Div([
        html.H4("How to use this dashboard:", className="mt-4"),
        html.Ul([
            html.Li("Upload your fiber photometry and tracking data files using the upload area above."),
            html.Li("The fiber photometry graph will show the processed signal over time."),
            html.Li("The tracking graph displays the mouse's movement in 3D, with color indicating time."),
            html.Li("Use the 'Previous' and 'Next' buttons to navigate between uploaded datasets."),
            html.Li("Click on the section headers to collapse or expand the uploaded files and graphs sections.")
        ])
    ])

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("AI Dashboard", className="text-center my-2", style={"color": PRIMARY_COLOR}), width=12)
    ]),
    dbc.Row([dbc.Col(create_upload_area(), width=12)]),
    create_collapsible("Uploaded Files", html.Div(id="uploaded-files"), "uploaded-files"),
    create_collapsible("Graphs", dbc.Row([
        dbc.Col(create_graph_card("Fiber Photometry Data", 'fiber-photometry-graph'), width=6),
        dbc.Col(create_graph_card("Mouse Tracking Data", 'tracking-graph'), width=6),
    ], className="flex-fill"), "graphs"),
    dbc.Row([dbc.Col(create_navigation_buttons(), width=12)]),
    dbc.Row([dbc.Col(create_instructions(), width=12)]),
    dcc.Store(id='data-store', data={}),
    dcc.Store(id='current-figure-index', data=0),
    dcc.Store(id='figures', data=[]),
    dcc.Store(id='tracking-figures', data=[])
], fluid=True, style={"height": "100vh"})

# Callbacks
@app.callback(
    Output('uploaded-files', 'children'),
    Input('upload-data', 'filename')
)
def update_uploaded_files(uploaded_filenames):
    if not uploaded_filenames:
        return "No files uploaded yet."
    return html.Ul([html.Li(filename) for filename in uploaded_filenames], className="mb-0")

@app.callback(
    [Output(f"{id_prefix}-collapse", "is_open") for id_prefix in ["uploaded-files", "graphs"]] +
    [Output(f"{id_prefix}-chevron", "className") for id_prefix in ["uploaded-files", "graphs"]],
    [Input(f"{id_prefix}-header", "n_clicks") for id_prefix in ["uploaded-files", "graphs"]],
    [State(f"{id_prefix}-collapse", "is_open") for id_prefix in ["uploaded-files", "graphs"]]
)
def toggle_collapse(*args):
    n_clicks_list = list(args[:2])
    is_open_list = list(args[2:])
    ctx = callback_context
    if not ctx.triggered:
        return is_open_list + ["fas fa-chevron-down mr-2", "fas fa-chevron-down mr-2"]

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    index = ["uploaded-files-header", "graphs-header"].index(button_id)

    new_is_open_list = list(is_open_list)
    new_is_open_list[index] = not new_is_open_list[index]

    new_chevron_classes = [
        "fas fa-chevron-up mr-2" if is_open else "fas fa-chevron-down mr-2"
        for is_open in new_is_open_list
    ]

    return new_is_open_list + new_chevron_classes


@app.callback(
    [Output('data-store', 'data'),
     Output('figures', 'data'),
     Output('tracking-figures', 'data'),
     Output('fiber-photometry-graph', 'figure'),
     Output('tracking-graph', 'figure'),
     Output('current-figure-index-output', 'children')],
    [Input('upload-data', 'contents'),
     Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('data-store', 'data'),
     State('current-figure-index-output', 'children'),
     State('figures', 'data'),
     State('tracking-figures', 'data')]
)
def update_output(list_of_contents, prev_button, next_button, list_of_names, data_dict, current_figure_index, figures,
                  tracking_figures):
    """Update the dashboard based on user interactions."""
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    if 'upload-data' in changed_id and list_of_contents:
        data_dict, new_figures, new_tracking_figures = process_upload(list_of_contents, list_of_names, data_dict, [],
                                                                      [])
        figures.extend(new_figures)
        tracking_figures.extend(new_tracking_figures)
        current_figure_index = 0
    elif 'prev-button' in changed_id and figures:
        current_figure_index = max(int(current_figure_index) - 1, 0)
    elif 'next-button' in changed_id and figures:
        current_figure_index = min(int(current_figure_index) + 1, len(figures) - 1)

    fiber_figure = update_figure(figures, current_figure_index) if figures else px.line(
        pd.DataFrame({'x': [0], 'y': [0]}), x='x', y='y', title='No data'
    )
    tracking_figure = update_figure(tracking_figures, current_figure_index) if tracking_figures else px.scatter_3d(
        pd.DataFrame({'x': [0], 'y': [0], 'z': [0]}), x='x', y='y', z='z', title="No Data"
    )

    return data_dict, figures, tracking_figures, fiber_figure, tracking_figure, current_figure_index

if __name__ == '__main__':
    app.run_server(debug=True)