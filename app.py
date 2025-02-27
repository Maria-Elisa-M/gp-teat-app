# imports
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback,  dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

dirpath = os.getcwd()
input_dir = os.path.join(dirpath, "data")
file_list = os.listdir(input_dir)

file_dict = [{'label': "_".join(file.split("_")[:5]), "value": file} for file in file_list]
plot_dict = [{'label': 'udder', 'value':0},{'label': 'teat', 'value': 1}]
q_dict = [{'label': '1', 'value': 1},{'label': '2', 'value': 2},{'label': '3', 'value': 3},{'label': '4', 'value': 4}]

def filter_df(filepath, q):
    file = np.loadtxt(filepath, delimiter = ",")
    idx = np.where(file[:, 5] == q)[0]
    data = file[idx, :]
    return data

def get_teat(data):
    teat = data[:, [0, 1, 3]]
    return teat

def get_udder(data):
    udder = data[:, [0, 1, 4]]
    return udder
def get_raw(data):
    raw = data[:, [0, 1, 2]]
    return raw

def get_lines(teat, udder):
    # tidx = np.argmin(udder[:, 2])
    tidx = np.argmin(teat[:, 2])
    tip = teat[tidx, :]
    p2 = tip.copy()
    p2[2] = 0
    len1 = np.linalg.norm(tip - p2)
    p3 = tip.copy()
    p3[2] = tip[2] + 3*len1/4
    dist = np.array([np.linalg.norm(teat[t, :] - p3) for t in range(teat.shape[0])])
    distx  = np.array([np.linalg.norm(teat[t, :2] - tip[:2]) for t in range(teat.shape[0])])
    rad = np.min(dist)
    sub_i = np.where((distx < rad) & (teat[:, 2] <p3[2]))
    ipts = teat[sub_i]
    medx = np.median(ipts[:,0])
    medy = np.median(ipts[:,1])
    p4 = np.array([medx, medy, p3[2]])
    l1 = p2 - tip
    l3 = p4 - tip
    disp = p4-tip
    t = -tip[2]/disp[2]
    p5  = tip + t * disp
    point_dict = {"p1": tip, "p2":p2, "p3": p3, "p4":p4, "p5":p5}
    return point_dict, t, disp

def get_length(point_dict):
    line_1 = point_dict["p1"] - point_dict["p2"]
    line_4 = point_dict["p1"] - point_dict["p4"]
    len_1 = np.linalg.norm(line_1)
    len_4 = np.linalg.norm(line_4)
    cos_an = np.dot(line_1, line_4)/(len_1 * len_4)
    len_5 = len_1/cos_an
    return len_5

def point_to_udder(udder, point_dict, t, disp):
    new_dict = {}
    for key in ["p1", "p2", "p3"]:
        p1uix = np.where((udder[:, 1] == point_dict[key][1]) & (udder[:, 0] == point_dict[key][0]))[0]
        p1u = udder[p1uix].copy()[0]
        p1u[2] =  point_dict[key][2] + p1u[2]
        new_dict[key] = p1u
    new_dict["p5"] = new_dict["p1"] + t*disp
    return new_dict

def get_plane(teat):
    # zero plane
    x = np.arange(np.min(teat[:, 0]), np.max(teat[:, 0]), 0.001)
    y = np.arange(np.min(teat[:, 1]), np.max(teat[:, 1]), 0.001)
    x = np.arange(np.min(teat[:, 0]), np.max(teat[:, 0]), 0.001)
    y = np.arange(np.min(teat[:, 1]), np.max(teat[:, 1]), 0.001)
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    coords =  np.column_stack([np.transpose(X), np.transpose(Y), np.zeros((len(Y), 1))])
    return coords
def plot_teat(teat, coords, point_dict):
    # lines
    lines = np.row_stack([point_dict["p4"], point_dict["p3"], point_dict["p1"], point_dict["p2"], point_dict["p5"],point_dict["p4"], point_dict["p1"]])
    # plot
    fig =  go.Figure(data=[go.Scatter3d(x = teat[:,0], y = teat[:,1], z=teat[:,2], mode='markers', marker=dict(size=2, color="blue", opacity = 1), name = "teat")])
    fig.add_trace(go.Scatter3d(x = lines[:,0], y = lines[:,1], z = lines[:,2], mode='markers', marker=dict(color="red", size = 3, opacity = 1), name = "lines"))
    fig.add_trace(go.Scatter3d(x = lines[:,0], y = lines[:,1], z = lines[:,2], mode='lines', marker=dict(color="red", size = 3, opacity = 1), name = "lines"))
    fig.add_trace(go.Scatter3d(x = coords[:,0], y = coords[:,1], z = coords[:,2], mode='markers', marker=dict(color="gray", size = 1, opacity = 0.5), name = "zero"))
    fig.update_layout(scene_aspectmode='data')
    fig.update_layout(paper_bgcolor="white", font_color =  "black", plot_bgcolor = "white")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    return fig

def plot_udder(udder, coords, new_dict):
    lines = np.row_stack([new_dict["p1"],new_dict["p2"], new_dict["p5"], new_dict["p1"]])
    # plot
    fig =  go.Figure(data=[go.Scatter3d(x = udder[:,0], y = udder[:,1], z=udder[:,2], mode='markers', marker=dict(size=2, color="blue", opacity = 1), name = "udder")])
    fig.add_trace(go.Scatter3d(x = lines[:,0], y = lines[:,1], z = lines[:,2], mode='markers', marker=dict(color="red", size = 3, opacity = 1), name = "lines"))
    fig.add_trace(go.Scatter3d(x = lines[:2,0], y = lines[:2,1], z = lines[:2,2], mode='lines', marker=dict(color="red", size = 3, opacity = 1), name = "lines"))
    fig.add_trace(go.Scatter3d(x = lines[2:,0], y = lines[2:,1], z = lines[2:,2], mode='lines', marker=dict(color="green", size = 3, opacity = 1), name = "lines"))
    fig.add_trace(go.Scatter3d(x = coords[:,0], y = coords[:,1], z = coords[:,2], mode='markers', marker=dict(color="gray", size = 1, opacity = 0.8), name = "raw"))
    fig.update_layout(scene_aspectmode='data')
    fig.update_layout(paper_bgcolor="white", font_color =  "black", plot_bgcolor = "white")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    return fig 

def blank_fig():
    fig = go.Figure(go.Scatter3d(x=[], y = [], z=[]))
    fig.update_layout(paper_bgcolor="black")
    fig.update_layout(legend_font_color="white")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    fig.update_layout(legend_font_color="white", width=1500, height=1000)
    return fig


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

MENU_STYLE = {
    'backgroundColor': 'black',
    'color': 'white',
}

sidebar = html.Div(
    [
        html.H2("Udder", className="display-4"),
        html.Hr(),
        html.P(
            "choose a cow and quarter", className="lead"
        ),
        html.Label("plot:"),
        dcc.RadioItems(id = 'plot_btn', options=plot_dict, value= 1),

        html.Label("Q:"),
        dcc.RadioItems(id = 'q-btn', options=q_dict, value=1),
        
        html.Label("Cow ID:"),
        dcc.Dropdown(id='cows-dpdn',options= file_dict, value = '1023_20231117_124217_frame_100_udder.csv', style = MENU_STYLE),

    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(
[html.Div(
             [dbc.Row(
                [dbc.Col([dcc.Graph(id='graph', figure = blank_fig())])])])
], id="page-content", style=CONTENT_STYLE)
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(
    Output("graph", "figure"),
    Input('cows-dpdn', 'value'), 
    Input('plot_btn', 'value'), 
    Input( 'q-btn', 'value'))
def get_frames(file_name, p, q):
    global input_dir
    filepath = os.path.join(input_dir, file_name)
    data = filter_df(filepath, q)
    teat = get_teat(data)
    udder = get_udder(data)
    raw = get_raw(data)
    point_dict, t, disp = get_lines(teat, raw)
    teat_len = np.round(get_length(point_dict) * 100, 2)
    new_dict = point_to_udder(udder, point_dict, t, disp)
    print(teat_len)
    if p == 1:
        coords = get_plane(teat)
        fig = plot_teat(teat, coords, point_dict)
    else:
        coords = raw
        fig = plot_udder(udder, coords, new_dict)
    return fig

if __name__ == '__main__':
    app.run(debug=True)