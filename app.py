import inspect
from dataclasses import dataclass, field, fields
from math import log2
from typing import Optional

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_mantine_components as dmc
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, callback, ctx, dash_table, dcc, html
from dash_iconify import DashIconify
from flask_executor import Executor

from Config import *
from Nodes import Nodes
from sorting_algorithms.sorting_algorithms import sorting_algorithms


@dataclass
class OnDataCallbackOutput:
    progress__value: int = 0
    progress__max: int = 1
    progress__animated: bool = False
    progress__label: str = ""
    progress_interval__disabled: bool = True
    show_statistics__disabled: bool = True
    expand_all__disabled: bool = True
    reset__disabled: bool = True
    visiblity_state__data: Optional[str] = None
    cytoscape__elements: list[dict] = field(default_factory=list)
    notifications_container__children: list | dmc.Notification = field(default_factory=list)
    buffered_input__data: Optional[list] = None
    last_tree_param__data: Optional[list] = None
    statistics_modal__is_open: bool = False
    statistics_graph__figure: dict | go.Figure = field(default_factory=dict)
    statistics_table__data: list = field(default_factory=list)
    control_loading__style: dict = field(default_factory=dict)
    control_loading_output__children: list = field(default_factory=list)

    @classmethod
    def outputs(cls) -> list[Output]:
        return [Output(*field.name.split("__")) for field in fields(cls)]

    def to_list(self) -> list:
        return [getattr(self, field.name) for field in fields(self)]


@callback(
    OnDataCallbackOutput.outputs(),
    Input("visiblity_state", "data"),
    Input("sorting_algorithm", "value"),
    Input("input_N", "value"),
    Input("show_full_labels", "value"),
    Input("cytoscape", "tapNode"),
    Input("expand_all", "n_clicks"),
    Input("reset", "n_clicks"),
    Input("show_statistics", "n_clicks"),
    Input("progress_interval", "n_intervals"),
    State("buffered_input", "data"),
    State("last_tree_param", "data"),
)
def on_data(
    visiblity_state: Optional[str],
    sorting_algorithm_i: str,
    input_N: Optional[str],
    show_full_labels: list,
    node: dict,
    _expand_all: int,
    _reset: int,
    _show_statistics: int,
    _n_intervals: int,
    buffered_input: Optional[str],
    last_tree_param: Optional[list],
):
    ret = OnDataCallbackOutput()
    if input_N is None:
        return ret.to_list()
    ret.last_tree_param__data = [int(sorting_algorithm_i), int(input_N)]
    sorting_algorithm = sorting_algorithms[int(sorting_algorithm_i)]
    if ret.last_tree_param__data != last_tree_param and sorting_algorithm.max_N < int(input_N):
        ret.notifications_container__children = dmc.Notification(
            id="warning_notification",
            title="Warning",
            action="show",
            message=f"Input `N={input_N}` is too large (upper limit for `{sorting_algorithm.name}` is {sorting_algorithm.max_N}), "
            f"using the results from random sampling for {MAX_SAMPLE_TIME_MS / 1000:.1f}s instead.",
            icon=DashIconify(icon="material-symbols:warning"),
            autoClose=10000,
        )

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id in ("sorting_algorithm", "input_N", "reset"):
        visiblity_state = None

    node_holder = Nodes.get_node_holder(int(sorting_algorithm_i), int(input_N))
    ret.progress__value, ret.progress__max = i, total = node_holder.get_progress()
    ret.progress__label = f"{i}/{total}"
    if i != total:
        if not node_holder.get_and_set_initialize_scheduled():
            executor.submit(node_holder.initialize, int(sorting_algorithm_i), int(input_N))
        if trigger_id == "expand_all":
            ret.buffered_input__data = "expand_all"
        elif trigger_id == "show_statistics":
            ret.buffered_input__data = "show_statistics"
        else:
            ret.buffered_input__data = buffered_input
        ret.progress__animated = True
        ret.progress_interval__disabled = False
        ret.control_loading__style = {"visibility": "hidden"}
        return ret.to_list()
    ret.show_statistics__disabled = False
    ret.expand_all__disabled = False
    ret.reset__disabled = False

    node_holder.wait_until_initialized()
    nodes = Nodes(node_holder, visiblity_state)
    if trigger_id == "show_statistics" or buffered_input == "show_statistics":
        data = np.array(node_holder.operation_cnts, dtype=np.int32)
        ret.statistics_graph__figure = px.histogram(x=data, title="Operation Count Distribution", labels={"x": "Operation Count"}, color=data, text_auto=True)
        ret.statistics_graph__figure.layout.update(showlegend=False)
        ret.statistics_modal__is_open = True
        input_total = sorting_algorithm.input_total(int(input_N))
        output_total = sorting_algorithm.output_total(int(input_N))
        lower_bound = log2(input_total / output_total)
        ret.statistics_table__data = [
            {
                "||Input||": input_total,
                "||Output||": output_total,
                "Lower Bound": f"Ω({lower_bound:.2f})",
                "Best": data.min(),
                "Worst": data.max(),
                "Average": f"{data.mean():.2f}",
            }
        ]
    elif trigger_id == "cytoscape":
        nodes.on_tap_node(int(node["data"]["id"]))
    elif trigger_id == "expand_all" or buffered_input == "expand_all":
        if not nodes.expand_all():
            ret.notifications_container__children = dmc.Notification(
                id="warning_notification",
                title="Warning",
                action="show",
                message=f"Too many nodes, only {MAX_ELEMENTS} nodes are displayed.",
                icon=DashIconify(icon="material-symbols:warning"),
                autoClose=10000,
            )
    ret.visiblity_state__data = nodes.get_visiblity_state()
    ret.cytoscape__elements = nodes.visible_elements(bool(show_full_labels))
    return ret.to_list()


@callback(Output("input_N", "invalid"), Input("input_N", "value"))
def on_input_N_invalid(input_N: Optional[str]):
    return input_N is None


@callback(
    Output("code_modal", "is_open"),
    Output("code_modal", "children"),
    Output("control_loading_output", "children", allow_duplicate=True),
    Input("show_code", "n_clicks"),
    State("sorting_algorithm", "value"),
    prevent_initial_call=True,
)
def on_show_code(_show_code: int, sorting_algorithm_i: str):
    sorting_algo = sorting_algorithms[int(sorting_algorithm_i)]
    code = inspect.getsource(sorting_algo.func).strip()
    children = [
        dbc.ModalHeader(dbc.ModalTitle(sorting_algo.name)),
        dbc.ModalBody(dcc.Markdown(f"```python\n{code}\n```"), style={"margin": "auto"}),
    ]
    return [True, children, ""]


control_panel = html.Div(
    [
        dbc.Button("Expand All", id="expand_all", disabled=True),
        dbc.Button("Reset", id="reset", disabled=True),
        dbc.Row(
            [
                "Sorting Algorithm:",
                dbc.Select(
                    options=[{"label": sorting_algo.name, "value": i} for i, sorting_algo in enumerate(sorting_algorithms)],
                    id="sorting_algorithm",
                    style={"width": "12rem"},
                    value=str(SORTING_ALGORITHM_I),
                    persistence=True,
                    persistence_type=USER_STATE_STORAGE_TYPE,
                ),
            ],
            style={"column-gap": "0", "display": "flex", "align-items": "center", "padding": "0.5rem"},
        ),
        dbc.Button("Show Code", id="show_code"),
        dbc.Row(
            [
                f"N(>0):",
                dbc.Input(
                    id="input_N",
                    type="number",
                    min=1,
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
        dbc.Progress(id="progress", value=0, striped=True, animated=True, style={"width": "10rem", "height": "1.3rem"}),
        dcc.Interval(id="progress_interval", interval=300, n_intervals=0, disabled=True),
        dbc.Button("Show Statistics", id="show_statistics", disabled=True),
        dbc.Checklist(  # don't know why dbc.Switch cannot align center vertically, so use dbc.Checklist instead
            options=[{"label": "Show Full Labels", "value": 0}],
            id="show_full_labels",
            value=[0] if SHOW_FULL_LABELS else [],
            switch=True,
            inline=True,
            persistence=True,
            persistence_type=USER_STATE_STORAGE_TYPE,
        ),
        dcc.Loading(id="control_loading", type="default", children=html.Div(id="control_loading_output"), style={"visibility": "hidden"}),
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
        sort="function(a, b) { return a.data('pos') - b.data('pos') }",
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
code_modal = dbc.Modal(id="code_modal", is_open=False, scrollable=True)
statistics_modal = dbc.Modal(
    id="statistics_modal",
    size="xl",
    is_open=False,
    scrollable=True,
    children=[
        dbc.ModalHeader(dbc.ModalTitle("Statistics")),
        dbc.ModalBody(
            [
                dcc.Graph(id="statistics_graph"),
                dash_table.DataTable(
                    id="statistics_table",
                    style_cell={"textAlign": "center"},
                    columns=[{"name": x, "id": x} for x in ("||Input||", "||Output||", "Lower Bound", "Best", "Worst", "Average")],
                ),
            ]
        ),
    ],
)
visiblity_state = dcc.Store(id="visiblity_state", storage_type=USER_STATE_STORAGE_TYPE)
buffered_input = dcc.Store(id="buffered_input", storage_type=USER_STATE_STORAGE_TYPE)
last_tree_param = dcc.Store(id="last_tree_param", storage_type=USER_STATE_STORAGE_TYPE)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {
            "name": "viewport",
            "content": "user-scalable=no, initial-scale=1, maximum-scale=1, minimum-scale=1, width=device-width, height=device-height, target-densitydpi=device-dpi",
        }
    ],
    title="Decision Tree",
    update_title=None,
)
app.layout = dmc.NotificationsProvider(
    html.Div(
        [html.Div(id="notifications_container"), control_panel, cytoscape, code_modal, statistics_modal, visiblity_state, buffered_input, last_tree_param],
        style={"height": "90vh", "width": "98vw", "margin": "auto"},
    )
)
server = app.server
executor = Executor(server)

if __name__ == "__main__":
    app.run(debug=True)
