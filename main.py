from collections import deque
from collections.abc import Callable
from typing import Optional

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback, dcc, html
from dash_iconify import DashIconify

from decision_tree import DecisionTreeNode, decision_tree
from sorting_algorithms import *


class ElementNode:
    def __init__(self, node_data: dict) -> None:
        self.node_data = node_data
        self.left: Optional[ElementNode] = None
        self.left_edge_data: Optional[dict] = None
        self.right: Optional[ElementNode] = None
        self.right_edge_data: Optional[dict] = None
        self.crop_label()

    def set_child_visible(self) -> None:
        def dfs(node: ElementNode, depth: int) -> None:
            if depth >= DISPLAY_DEPTH:
                return
            node.node_data["data"]["visibility"] = "visible"
            if node.left is not None:
                if depth + 1 < DISPLAY_DEPTH:
                    node.left_edge_data["data"]["visibility"] = "visible"
                dfs(node.left, depth + 1)
            if node.right is not None:
                if depth + 1 < DISPLAY_DEPTH:
                    node.right_edge_data["data"]["visibility"] = "visible"
                dfs(node.right, depth + 1)

        dfs(self, 0)

    def set_child_hidden(self) -> None:
        def dfs(node: ElementNode) -> None:
            if node.node_data["data"]["visibility"] == "hidden":
                return
            node.node_data["data"]["visibility"] = "hidden"
            if node.left is not None:
                node.left_edge_data["data"]["visibility"] = "hidden"
                dfs(node.left)
            if node.right is not None:
                node.right_edge_data["data"]["visibility"] = "hidden"
                dfs(node.right)

        for child, edge in zip((self.left, self.right), (self.left_edge_data, self.right_edge_data)):
            if child is not None:
                edge["data"]["visibility"] = "hidden"
                dfs(child)

    def update_classes(self) -> None:
        if self.is_leaf():
            self.node_data["classes"] = "is_leaf"
        elif self.has_hidden_child():
            self.node_data["classes"] = "has_hidden_child"
        else:
            self.node_data["classes"] = ""

    def crop_label(self) -> None:
        label = self.node_data["data"]["full_label"]
        if len(label) > LABEL_MAX_LENGTH:
            self.node_data["data"]["croped_label"] = label[: LABEL_MAX_LENGTH - 3] + "..."
        else:
            self.node_data["data"]["croped_label"] = label

    def has_hidden_child(self) -> bool:
        return (self.left is not None and self.left.node_data["data"]["visibility"] == "hidden") or (
            self.right is not None and self.right.node_data["data"]["visibility"] == "hidden"
        )

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    __slots__ = ["node_data", "left", "left_edge_data", "right", "right_edge_data"]


class Elements:
    cached: dict[tuple[int, int], "Elements"] = {}

    def __init__(self, sorting_func: Callable[[list], None], N: int) -> None:
        DecisionTreeNode.reset_id()
        self.elements: list[dict] = []
        self.element_nodes: dict[int, ElementNode] = {}

        root = decision_tree(sorting_func, N)

        def dfs(node: DecisionTreeNode, parent: Optional[ElementNode] = None, is_left: bool = False, depth: int = 0) -> None:
            self.element_nodes[node.id] = element_node = ElementNode(
                {
                    "data": {
                        "id": str(node.id),
                        "visibility": "visible" if depth < DISPLAY_DEPTH else "hidden",
                        "full_label": node.get_arr() + " " + node.get_actuals(),
                    }
                },
            )
            if parent is not None:
                if is_left:
                    parent.left = element_node
                else:
                    parent.right = element_node
            self.elements.append(element_node.node_data)

            if node.cmp_xy is None:
                return
            x, y = [chr(ord("a") + x) for x in node.cmp_xy]
            for op, child in zip("<>", [node.left, node.right]):
                if child is not None:
                    is_left = op == "<"
                    edge_data = {
                        "data": {
                            "source": str(node.id),
                            "target": str(child.id),
                            "visibility": "visible" if depth < DISPLAY_DEPTH - 1 else "hidden",
                            "cmp_op": f"{x}{op}{y}",
                        }
                    }
                    self.elements.append(edge_data)
                    if is_left:
                        element_node.left_edge_data = edge_data
                    else:
                        element_node.right_edge_data = edge_data
                    dfs(child, element_node, is_left, depth + 1)

        dfs(root)

    def reset(self) -> None:
        self.element_nodes[1].set_child_hidden()
        self.element_nodes[1].set_child_visible()

    @classmethod
    def get(cls, sorting_algorithm_i: int, N: int) -> "Elements":
        key = (sorting_algorithm_i, N)
        if key not in cls.cached:
            cls.cached[key] = cls(sorting_algorithms[sorting_algorithm_i][1], N)
        else:
            cls.cached[key].reset()
        return cls.cached[key]

    def visible_elements(self) -> list[dict]:
        for element_node in self.element_nodes.values():
            element_node.update_classes()
        return [element for element in self.elements if element["data"]["visibility"] == "visible"]


@callback(Output("cytoscape", "elements", allow_duplicate=True), Input("cytoscape", "tapNode"), prevent_initial_call=True)
def tapNodeCb(node: Optional[dict]):
    element_node = current_elements.element_nodes[int(node["data"]["id"])]
    if element_node.has_hidden_child():
        element_node.set_child_visible()
    else:
        element_node.set_child_hidden()

    return current_elements.visible_elements()


@callback(
    Output("cytoscape", "elements", allow_duplicate=True),
    Output("control-loading-output", "children", allow_duplicate=True),
    Output("notifications-container", "children", allow_duplicate=True),
    Input("expand-all", "n_clicks"),
    prevent_initial_call=True,
)
def expandAllCb(_: int):
    tot = 1

    queue = deque([current_elements.element_nodes[1]])
    while queue:
        node = queue.popleft()
        node.node_data["data"]["visibility"] = "visible"
        tot += 1
        if tot < MAX_ELEMENTS and node.left is not None:
            node.left_edge_data["data"]["visibility"] = "visible"
            queue.append(node.left)
        if tot < MAX_ELEMENTS and node.right is not None:
            node.right_edge_data["data"]["visibility"] = "visible"
            queue.append(node.right)

    notification = ""
    if tot >= MAX_ELEMENTS:
        notification = dmc.Notification(
            id="warning-notification",
            title="Warning",
            action="show",
            message=f"Too many elements, only {MAX_ELEMENTS} elements are displayed.",
            icon=DashIconify(icon="material-symbols:warning"),
        )
    return [current_elements.visible_elements(), "", notification]


@callback(Output("cytoscape", "elements", allow_duplicate=True), Input("reset", "n_clicks"), prevent_initial_call=True)
def resetCb(_: int):
    current_elements.reset()
    return current_elements.visible_elements()


@callback(
    Output("cytoscape", "elements", allow_duplicate=True),
    Output("control-loading-output", "children", allow_duplicate=True),
    Input("sorting-algorithm", "value"),
    Input("input-N", "value"),
    prevent_initial_call=True,
)
def reconstructCb(input_sorting_algorithm_i: Optional[str], input_N: Optional[str]):
    global sorting_algorithm_i, N, current_elements
    if input_sorting_algorithm_i is not None:
        sorting_algorithm_i = int(input_sorting_algorithm_i)
    if input_N is not None:
        N = int(input_N)
    current_elements = Elements.get(sorting_algorithm_i, N)
    return [current_elements.visible_elements(), ""]


@callback(
    Output("cytoscape", "stylesheet", allow_duplicate=True),
    Output("control-loading-output", "children", allow_duplicate=True),
    Input("show-full-labels", "value"),
    State("cytoscape", "stylesheet"),
    prevent_initial_call=True,
)
def showFullLabelsCb(show_full_labels: list, stylesheet: list[dict]):
    for rule in stylesheet:
        if rule["selector"] == "node":
            rule["style"]["label"] = "data(full_label)" if len(show_full_labels) else "data(croped_label)"
            break
    return [stylesheet, ""]


DISPLAY_DEPTH = 4
LABEL_MAX_LENGTH = 50
MAX_ELEMENTS = 500
N = 3
N_RANGE = (1, 8)
sorting_algorithm_i = 0

current_elements: Optional[Elements] = None


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

if __name__ == "__main__":
    current_elements = Elements.get(sorting_algorithm_i, N)
    control_panel = html.Div(
        [
            dbc.Button("Expand All", id="expand-all"),
            dbc.Button("Reset", id="reset"),
            dbc.Row(
                [
                    "Sorting Algorithm:",
                    dbc.Select(
                        options=[{"label": name, "value": i} for i, (name, _) in enumerate(sorting_algorithms)],
                        value="0",
                        id="sorting-algorithm",
                        style={"width": "12rem"},
                    ),
                ],
                style={"column-gap": "0", "display": "flex", "align-items": "center", "padding": "0.5rem"},
            ),
            dbc.Row(
                [
                    f"N({N_RANGE[0]}~{N_RANGE[1]}):",
                    dbc.Input(id="input-N", type="number", min=N_RANGE[0], max=N_RANGE[1], step=1, style={"width": "4rem"}, value=N),
                ],
                style={"column-gap": "0", "display": "flex", "align-items": "center", "padding": "0.5rem"},
            ),
            # don't know why dbc.Switch cannot align center vertically, so use dbc.Checklist instead
            dbc.Checklist(options=[{"label": "Show Full Labels", "value": 0}], id="show-full-labels", value=[], switch=True, inline=True),
            dcc.Loading(id="control-loading", type="default", children=html.Div(id="control-loading-output")),
        ],
        style={"column-gap": "1rem", "display": "flex", "align-items": "center", "margin": "1rem", "flex-wrap": "wrap"},
    )
    cytoscape = cyto.Cytoscape(
        id="cytoscape",
        layout=dict(
            name="breadthfirst",
            directed=True,
            roots="#1",
            animate=True,
            animationDuration=200,
        ),
        elements=current_elements.visible_elements(),
        style={"height": "98%", "width": "100%"},
        stylesheet=[
            {"selector": "edge", "style": {"label": "data(cmp_op)"}},
            {"selector": "node", "style": {"label": "data(croped_label)"}},
            {"selector": ".has_hidden_child", "style": {"background-color": "red", "line-color": "red"}},
            {"selector": ".is_leaf", "style": {"background-color": "green", "line-color": "green"}},
            {"selector": "label", "style": {"color": "#0095FF"}},
        ],
        autoRefreshLayout=True,
    )

    app.layout = dmc.NotificationsProvider(
        html.Div(
            [html.Div(id="notifications-container"), control_panel, cytoscape],
            style={"height": "90vh", "width": "98vw", "margin": "auto"},
        )
    )

    app.run()
