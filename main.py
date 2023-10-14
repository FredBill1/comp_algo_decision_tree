import inspect
from typing import Optional

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_mantine_components as dmc
import plotly.express as px
from dash import Dash, Input, Output, State, callback, ctx, dash_table, dcc, html
from dash_iconify import DashIconify

from Config import *
from Elements import Elements
from sorting_algorithms import sorting_algorithms


@callback(
    Output("user-state", "data"),
    Output("cytoscape", "elements"),
    Output("cytoscape", "stylesheet"),
    Output("sorting-algorithm", "value"),
    Output("input-N", "value"),
    Output("show-full-labels", "value"),
    Output("notifications-container", "children"),
    Output("control-loading-output", "children"),
    Input("user-state", "data"),
    Input("sorting-algorithm", "value"),
    Input("input-N", "value"),
    Input("show-full-labels", "value"),
    Input("cytoscape", "tapNode"),
    Input("expand-all", "n_clicks"),
    Input("reset", "n_clicks"),
    State("cytoscape", "stylesheet"),
)
def on_data(
    user_state: dict,
    input_sorting_algorithm_i: str,
    input_N: str,
    show_full_labels: list,
    node: Optional[dict],
    _expand_all: int,
    _reset: int,
    stylesheet: list[dict],
):
    if not user_state:
        user_state = DEFAULT_USER_STATE
    notification = ""

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "sorting-algorithm":
        user_state["sorting_algorithm_i"] = int(input_sorting_algorithm_i)
    elif trigger_id == "input-N":
        user_state["N"] = int(input_N)
    elif trigger_id == "show-full-labels":
        user_state["show_full_labels"] = bool(show_full_labels)
    if trigger_id in ("sorting-algorithm", "input-N", "reset"):
        user_state["visiblity_state"] = None

    elements = Elements(**user_state)
    if trigger_id == "cytoscape":
        elements.on_tap_node(int(node["data"]["id"]))
    elif trigger_id == "expand-all":
        if not elements.expand_all():
            notification = dmc.Notification(
                id="warning-notification",
                title="Warning",
                action="show",
                message=f"Too many elements, only {MAX_ELEMENTS} elements are displayed.",
                icon=DashIconify(icon="material-symbols:warning"),
            )
    if trigger_id in ("cytoscape", "expand-all"):
        user_state["visiblity_state"] = elements.get_visiblity_state()

    sorting_algorithm_i, N, show_full_labels = user_state["sorting_algorithm_i"], user_state["N"], user_state["show_full_labels"]
    for rule in stylesheet:
        if rule["selector"] == "node":
            rule["style"]["label"] = "data(full_label)" if show_full_labels else "data(cropped_label)"
            break
    return [user_state, elements.visible_elements(), stylesheet, str(sorting_algorithm_i), str(N), [0] if show_full_labels else [], notification, ""]


@callback(
    Output("code-modal", "is_open"),
    Output("code-modal", "children"),
    Input("show-code", "n_clicks"),
    State("code-modal", "is_open"),
    State("user-state", "data"),
    prevent_initial_call=True,
)
def on_show_code(_: int, is_open: bool, user_state: dict):
    if is_open:
        return [False, []]
    sorting_algorithm_i = user_state["sorting_algorithm_i"]
    code = inspect.getsource(sorting_algorithms[sorting_algorithm_i][1]).strip()
    children = [
        dbc.ModalHeader(dbc.ModalTitle(sorting_algorithms[sorting_algorithm_i][0])),
        dbc.ModalBody(dcc.Markdown(f"```python\n{code}\n```"), style={"margin": "auto"}),
    ]
    return [True, children]


@callback(
    Output("statistics-modal", "is_open"),
    Output("statistics-graph", "figure"),
    Output("statistics-table", "data"),
    Input("show-statistics", "n_clicks"),
    State("statistics-modal", "is_open"),
    State("user-state", "data"),
    prevent_initial_call=True,
)
def on_show_statics(_: int, is_open: bool, user_state: dict):
    if is_open:
        return [False, {}, []]
    element_holder = Elements.get_element_holder(**user_state)
    data = element_holder.operation_cnts
    fig = px.histogram(x=data, title="Operation Count Distribution", labels={"x": "Operation Count"}, color=data, text_auto=True)
    fig.layout.update(showlegend=False)
    return [True, fig, [{"Best": data.min(), "Worst": data.max(), "Average": f"{data.mean():.2f}"}]]


control_panel = html.Div(
    [
        dbc.Button("Expand All", id="expand-all"),
        dbc.Button("Reset", id="reset"),
        dbc.Row(
            [
                "Sorting Algorithm:",
                dbc.Select(
                    options=[{"label": name, "value": i} for i, (name, _) in enumerate(sorting_algorithms)],
                    id="sorting-algorithm",
                    style={"width": "12rem"},
                ),
            ],
            style={"column-gap": "0", "display": "flex", "align-items": "center", "padding": "0.5rem"},
        ),
        dbc.Button("Show Code", id="show-code"),
        dbc.Row(
            [
                f"N({N_RANGE[0]}~{N_RANGE[1]}):",
                dbc.Input(id="input-N", type="number", min=N_RANGE[0], max=N_RANGE[1], step=1, style={"width": "4rem"}),
            ],
            style={"column-gap": "0", "display": "flex", "align-items": "center", "padding": "0.5rem"},
        ),
        # don't know why dbc.Switch cannot align center vertically, so use dbc.Checklist instead
        dbc.Button("Show Statistics", id="show-statistics"),
        dbc.Checklist(options=[{"label": "Show Full Labels", "value": 0}], id="show-full-labels", value=[], switch=True, inline=True),
        dcc.Loading(id="control-loading", type="default", children=html.Div(id="control-loading-output")),
    ],
    style={"column-gap": "1rem", "display": "flex", "align-items": "center", "margin": "1rem", "flex-wrap": "wrap"},
)
cyto.load_extra_layouts()
cytoscape = cyto.Cytoscape(
    id="cytoscape",
    layout=dict(
        name="dagre",
        rankDir="UD",
        spacingFactor=1.75,
        animate=True,
        animationDuration=200,
    ),
    style={"height": "98%", "width": "100%"},
    stylesheet=[
        {"selector": "edge", "style": {"label": "data(cmp_op)", "curve-style": "bezier", "target-arrow-shape": "triangle"}},
        {"selector": "node", "style": {"label": "data(cropped_label)"}},
        {"selector": ".has_hidden_child", "style": {"background-color": "red", "line-color": "red"}},
        {"selector": ".is_leaf", "style": {"background-color": "green", "line-color": "green"}},
        {"selector": "label", "style": {"color": "#0095FF"}},
    ],
    autoRefreshLayout=True,
)
code_modal = dbc.Modal(id="code-modal", is_open=False, scrollable=True)
statistics_modal = dbc.Modal(
    id="statistics-modal",
    size="xl",
    is_open=False,
    scrollable=True,
    children=[
        dbc.ModalHeader(dbc.ModalTitle("Statistics")),
        dbc.ModalBody(
            [
                dcc.Graph(id="statistics-graph"),
                dash_table.DataTable(
                    id="statistics-table",
                    style_cell={"textAlign": "center"},
                    columns=[{"name": x, "id": x} for x in ("Best", "Worst", "Average")],
                ),
            ]
        ),
    ],
)
user_state = dcc.Store(id="user-state", storage_type=USER_STATE_STORAGE_TYPE)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {
            "name": "viewport",
            "content": "user-scalable=no, initial-scale=1, maximum-scale=1, minimum-scale=1, width=device-width, height=device-height, target-densitydpi=device-dpi",
        }
    ],
)
app.layout = dmc.NotificationsProvider(
    html.Div(
        [html.Div(id="notifications-container"), control_panel, cytoscape, code_modal, statistics_modal, user_state],
        style={"height": "90vh", "width": "98vw", "margin": "auto"},
    )
)

if __name__ == "__main__":
    app.run()
