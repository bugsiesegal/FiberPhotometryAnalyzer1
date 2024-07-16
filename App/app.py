import base64
import io

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("AI Dashboard"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            className='upload-area',
            multiple=True
        ), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id='fiber-photometry-graph')), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Loading(html.Div(dcc.Graph(id='tracking-graph', className='square-graph'), className='square-graph-container')), width=12)
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("<", id="prev-button", n_clicks=0), width="auto"),
        dbc.Col(html.Div(id='current-figure-index-output'), width="auto"),
        dbc.Col(dbc.Button(">", id="next-button", n_clicks=0), width="auto")
    ]),
    dcc.Store(id='data-store', data={}),
    dcc.Store(id='current-figure-index', data=0),
    dcc.Store(id='figures', data=[]),
    dcc.Store(id='tracking-figures', data=[])
])

def process_fiber_data(fiber_data,
                       fiber_channel_name="Analog In. | Ch.1 AIn-1 - Dem (AOut-2)",
                       control_channel_name="Analog In. | Ch.1 AIn-1 - Dem (AOut-3)"):
    fiber_channel = fiber_data[fiber_channel_name].to_numpy()
    control_channel = fiber_data[control_channel_name].to_numpy()
    fiber_channel -= control_channel
    return fiber_channel


def process_tracking_data(tracking_data):
    column_names = tracking_data.columns
    x_columns = [column for column in column_names if '_x' in column]
    y_columns = [column for column in column_names if '_y' in column]
    z_columns = [column for column in column_names if '_z' in column]
    x = tracking_data[x_columns].to_numpy()
    y = tracking_data[y_columns].to_numpy()
    z = tracking_data[z_columns].to_numpy()
    tracking = np.concatenate([x, y, z], axis=1)
    return tracking


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df, filename


def plot_fiber_data(fiber_data, filename):
    fiber_data = pd.DataFrame(
        {'Time (Seconds)':fiber_data["Time(s)"], 'Fiber':fiber_data["Analog In. | Ch.1 AIn-1 - Dem (AOut-2)"] - fiber_data["Analog In. | Ch.1 AIn-1 - Dem (AOut-3)"]},
    )
    # Select only columns with x, y, z
    fig = px.line(x="Time (Seconds)", y="Fiber", data_frame=fiber_data, title=filename)
    return fig


def plot_tracking_data(tracking_data, filename):
    # Select only columns with x, y, z
    tracking_data = tracking_data.filter(regex='.*(_x|_y|_z).*')

    # Define the connections between joints as pairs of column indices
    connections = [
        (0, 3), (3, 6), (6, 4), (4, 5), (5, 7), (7, 8), # Head and body
        (0, 1), (0, 2), # Nose to right and left ear
        (9, 10), (10, 5), # Left Front Leg
        (11, 12), (12, 5), # Right Front Leg
        (13, 14), (14, 6), # Left Hind Leg
        (15, 16), (16, 6) # Right Hind Leg
    ]

    fig = go.Figure()

    # Get the number of frames (assuming each row is a frame)
    num_frames = len(tracking_data)

    # Normalize frame indices to the range [0, 1] for color mapping
    normalized_frames = np.linspace(0, 1, num_frames)

    # Add scatter plot for each frame with color indicating time
    for frame, norm_frame in enumerate(normalized_frames):
        # Map normalized frame index to a color in the Viridis colormap
        color = px.colors.sample_colorscale('Viridis', norm_frame)

        for (start, end) in connections:
            fig.add_trace(go.Scatter3d(
                x=[tracking_data.iloc[frame, start * 3], tracking_data.iloc[frame, end * 3]],
                z=[-tracking_data.iloc[frame, start * 3 + 1], -tracking_data.iloc[frame, end * 3 + 1]],  # Swap y and z
                y=[tracking_data.iloc[frame, start * 3 + 2], tracking_data.iloc[frame, end * 3 + 2]],  # Swap y and z
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                opacity=0.6
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=filename
    )

    return fig


def process_upload(list_of_contents, list_of_names, data_dict, figures, tracking_figures):
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


def update_figure(figures, current_figure_index):
    return figures[current_figure_index]


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
def update_output(list_of_contents, prev_button, next_button, list_of_names, data_dict, current_figure_index, figures, tracking_figures):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'upload-data' in changed_id:
        if list_of_contents:
            data_dict, new_figures, new_tracking_figures = process_upload(list_of_contents, list_of_names, data_dict, [], [])
            figures.extend(new_figures)
            tracking_figures.extend(new_tracking_figures)
            current_figure_index = 0
            return data_dict, figures, tracking_figures, figures[current_figure_index], tracking_figures[current_figure_index], current_figure_index

    elif 'prev-button' in changed_id and figures:
        current_figure_index = max(current_figure_index - 1, 0)
        return data_dict, figures, tracking_figures, update_figure(figures, current_figure_index), update_figure(tracking_figures, current_figure_index), current_figure_index

    elif 'next-button' in changed_id and figures:
        current_figure_index = min(current_figure_index + 1, len(figures) - 1)
        return data_dict, figures, tracking_figures, update_figure(figures, current_figure_index), update_figure(tracking_figures, current_figure_index), current_figure_index

    return data_dict, figures, tracking_figures, figures[current_figure_index] if figures else px.line([0, 0], title='No data'), tracking_figures[current_figure_index] if tracking_figures else px.scatter_3d([(0,0,0)], title="No Data"), current_figure_index


if __name__ == '__main__':
    app.run_server(debug=True)
