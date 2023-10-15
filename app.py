import inspect
from typing import Optional

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_mantine_components as dmc
import plotly.express as px
from dash import Dash, Input, Output, State, callback, ctx, dash_table, dcc, html
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from Config import *
from Elements import Elements
from sorting_algorithms import sorting_algorithms


@callback(
    Output("visiblity-state", "data"),
    Output("cytoscape", "elements"),
    Output("notifications-container", "children"),
    Output("control-loading-output", "children"),
    Input("visiblity-state", "data"),
    Input("sorting-algorithm", "value"),
    Input("input-N", "value"),
    Input("show-full-labels", "value"),
    Input("cytoscape", "tapNode"),
    Input("expand-all", "n_clicks"),
    Input("reset", "n_clicks"),
)
def on_data(
    visiblity_state: Optional[str],
    sorting_algorithm_i: str,
    input_N: Optional[str],
    show_full_labels: list,
    node: dict,
    _expand_all: int,
    _reset: int,
):
    if input_N is None:
        return [None, [], [], []]
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id in ("sorting-algorithm", "input-N", "reset"):
        visiblity_state = None
    elements = Elements(int(sorting_algorithm_i), int(input_N), visiblity_state)
    notification = ""
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
    return [elements.get_visiblity_state(), elements.visible_elements(bool(show_full_labels)), notification, []]


@callback(
    Output("input-N", "invalid"),
    Output("show-statistics", "disabled"),
    Input("input-N", "value"),
)
def on_input_N_invalid(input_N: Optional[str]):
    disabled = input_N is None
    return [disabled, disabled]


@callback(
    Output("code-modal", "is_open"),
    Output("code-modal", "children"),
    Output("control-loading-output", "children", allow_duplicate=True),
    Input("show-code", "n_clicks"),
    State("code-modal", "is_open"),
    State("sorting-algorithm", "value"),
    prevent_initial_call=True,
)
def on_show_code(_: int, is_open: bool, sorting_algorithm_i: str):
    if is_open:
        return [False, []]
    sorting_algorithm_i = int(sorting_algorithm_i)
    code = inspect.getsource(sorting_algorithms[sorting_algorithm_i][1]).strip()
    children = [
        dbc.ModalHeader(dbc.ModalTitle(sorting_algorithms[sorting_algorithm_i][0])),
        dbc.ModalBody(dcc.Markdown(f"```python\n{code}\n```"), style={"margin": "auto"}),
    ]
    return [True, children, ""]


@callback(
    Output("statistics-modal", "is_open"),
    Output("statistics-graph", "figure"),
    Output("statistics-table", "data"),
    Output("control-loading-output", "children", allow_duplicate=True),
    Input("show-statistics", "n_clicks"),
    State("statistics-modal", "is_open"),
    State("sorting-algorithm", "value"),
    State("input-N", "value"),
    prevent_initial_call=True,
)
def on_show_statics(_: int, is_open: bool, sorting_algorithm_i: str, input_N: Optional[str]):
    if is_open:
        return [False, {}, []]
    if input_N is None:
        raise PreventUpdate
    element_holder = Elements.get_element_holder(int(sorting_algorithm_i), int(input_N))
    data = element_holder.operation_cnts
    fig = px.histogram(x=data, title="Operation Count Distribution", labels={"x": "Operation Count"}, color=data, text_auto=True)
    fig.layout.update(showlegend=False)
    return [True, fig, [{"Best": data.min(), "Worst": data.max(), "Average": f"{data.mean():.2f}"}], ""]


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
                    value=str(SORTING_ALGORITHM_I),
                    persistence=True,
                    persistence_type=USER_STATE_STORAGE_TYPE,
                ),
            ],
            style={"column-gap": "0", "display": "flex", "align-items": "center", "padding": "0.5rem"},
        ),
        dbc.Button("Show Code", id="show-code"),
        dbc.Row(
            [
                f"N({N_RANGE[0]}~{N_RANGE[1]}):",
                dbc.Input(
                    id="input-N",
                    type="number",
                    min=N_RANGE[0],
                    max=N_RANGE[1],
                    step=1,
                    style={"width": "5rem"},
                    debounce=True,
                    value=INPUT_N,
                    persistence=True,
                    persistence_type=USER_STATE_STORAGE_TYPE,
                ),
            ],
            style={"column-gap": "0", "display": "flex", "align-items": "center", "padding": "0.5rem"},
        ),
        # don't know why dbc.Switch cannot align center vertically, so use dbc.Checklist instead
        dbc.Button("Show Statistics", id="show-statistics"),
        dbc.Checklist(
            options=[{"label": "Show Full Labels", "value": 0}],
            id="show-full-labels",
            value=[0] if SHOW_FULL_LABELS else [],
            switch=True,
            inline=True,
            persistence=True,
            persistence_type=USER_STATE_STORAGE_TYPE,
        ),
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
        {"selector": "node", "style": {"label": "data(label)"}},
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
visiblity_state = dcc.Store(id="visiblity-state", storage_type=USER_STATE_STORAGE_TYPE)

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
        [html.Div(id="notifications-container"), control_panel, cytoscape, code_modal, statistics_modal, visiblity_state],
        style={"height": "90vh", "width": "98vw", "margin": "auto"},
    )
)
server = app.server

if __name__ == "__main__":
    app.run(debug=True)
