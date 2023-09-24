from collections.abc import Callable
from typing import Optional

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import Dash, Input, Output, State, callback, html

from decision_tree import DecisionTreeNode, decision_tree
from sorting_algorithms import *

DISPLAY_DEPTH = 4
LABEL_MAX_LENGTH = 50


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


def construct_elements(sorting_func: Callable[[list], None], N: int) -> int:
    DecisionTreeNode.reset_id()
    elements.clear()
    element_nodes.clear()
    root = decision_tree(sorting_func, N)

    def dfs(node: DecisionTreeNode, parent: Optional[ElementNode] = None, is_left: bool = False, depth: int = 0) -> None:
        element_nodes[node.id] = element_node = ElementNode(
            {
                "data": {
                    "id": str(node.id),
                    "visibility": "visible" if depth < DISPLAY_DEPTH else "hidden",
                    "arr": node.get_arr(),
                    "full_label": node.get_arr() + " " + node.get_actuals(),
                }
            },
        )
        if parent is not None:
            if is_left:
                parent.left = element_node
            else:
                parent.right = element_node
        elements.append(element_node.node_data)

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
                elements.append(edge_data)
                if is_left:
                    element_node.left_edge_data = edge_data
                else:
                    element_node.right_edge_data = edge_data
                dfs(child, element_node, is_left, depth + 1)

    dfs(root)


def visible_elements() -> list[dict]:
    for element_node in element_nodes.values():
        element_node.update_classes()
    return [element for element in elements if element["data"]["visibility"] == "visible"]


@callback(Output("cytoscape", "elements", allow_duplicate=True), Input("cytoscape", "tapNode"), prevent_initial_call=True)
def tapNodeCb(node: Optional[dict]):
    element_node = element_nodes[int(node["data"]["id"])]
    if element_node.has_hidden_child():
        element_node.set_child_visible()
    else:
        element_node.set_child_hidden()

    return visible_elements()


@callback(Output("cytoscape", "elements", allow_duplicate=True), Input("expand-all", "n_clicks"), prevent_initial_call=True)
def expandAllCb(_: int):
    for node in element_nodes.values():
        node.node_data["data"]["visibility"] = "visible"
        if node.left is not None:
            node.left_edge_data["data"]["visibility"] = "visible"
        if node.right is not None:
            node.right_edge_data["data"]["visibility"] = "visible"
    return visible_elements()


@callback(Output("cytoscape", "elements", allow_duplicate=True), Input("reset", "n_clicks"), prevent_initial_call=True)
def resetCb(_: int):
    element_nodes[1].set_child_hidden()
    element_nodes[1].set_child_visible()
    return visible_elements()


@callback(
    Output("cytoscape", "stylesheet", allow_duplicate=True),
    Input("show-full-labels", "value"),
    State("cytoscape", "stylesheet"),
    prevent_initial_call=True,
)
def showFullLabelsCb(show_actuals: bool, stylesheet: list[dict]):
    for rule in stylesheet:
        if rule["selector"] == "node":
            rule["style"]["label"] = "data(full_label)" if show_actuals else "data(croped_label)"
            break
    return stylesheet


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


elements = []
element_nodes: dict[int, ElementNode] = {}

if __name__ == "__main__":
    construct_elements(LomutoQS, 5)

    app.layout = html.Div(
        [
            html.Div(
                [
                    dbc.Button("Expand All", id="expand-all"),
                    dbc.Button("Reset", id="reset"),
                    dbc.Switch(label="Show Full Labels", id="show-full-labels", value=False),
                ],
                style={"column-gap": "1rem", "display": "flex", "align-items": "center", "justify-content": "center", "margin": "1rem"},
            ),
            cyto.Cytoscape(
                id="cytoscape",
                layout=dict(
                    name="breadthfirst",
                    directed=True,
                    roots="#1",
                    animate=True,
                    animationDuration=200,
                ),
                elements=visible_elements(),
                style={"height": "100%", "width": "100%"},
                stylesheet=[
                    {"selector": "edge", "style": {"label": "data(cmp_op)"}},
                    {"selector": "node", "style": {"label": "data(croped_label)"}},
                    {"selector": ".has_hidden_child", "style": {"background-color": "red", "line-color": "red"}},
                    {"selector": ".is_leaf", "style": {"background-color": "green", "line-color": "green"}},
                ],
                autoRefreshLayout=True,
            ),
        ],
        style={"height": "95vh", "width": "100%"},
    )

    app.run(debug=True)
