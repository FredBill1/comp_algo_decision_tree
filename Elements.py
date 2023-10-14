import base64
import zlib
from collections import deque
from collections.abc import Callable
from threading import Lock
from typing import Optional

import numpy as np
import sortednp as snp

from Config import *
from decision_tree import DecisionTreeNode, decision_tree
from sorting_algorithms import sorting_algorithms


class ElementNode:
    def __init__(self, id: int, full_label: str, edge_data: Optional[dict]) -> None:
        self.id = id
        self.data = dict(
            id=str(id),
            full_label=full_label,
            cropped_label=full_label[: LABEL_MAX_LENGTH - 3] + "..." if len(full_label) > LABEL_MAX_LENGTH else full_label,
        )
        self.edge_data = edge_data
        self.left: Optional[ElementNode] = None
        self.right: Optional[ElementNode] = None

    @property
    def node_data(self) -> dict:
        return {"data": dict(self.data)}

    __slots__ = ["id", "data", "edge_data", "left", "right"]


class ElementHolder:
    def __init__(self, sorting_func: Callable[[list], None], N: int) -> None:
        self.element_nodes: list[ElementNode] = []

        root, self.operation_cnts = decision_tree(sorting_func, N)

        def dfs(
            node: DecisionTreeNode,
            parent: Optional[ElementNode] = None,
            edge_data: Optional[dict] = None,
            is_left: bool = False,
            cmp_op: Optional[str] = None,
            depth: int = 0,
        ) -> None:
            element_node = ElementNode(len(self.element_nodes), node.get_arr() + " " + node.get_actuals(), edge_data)
            self.element_nodes.append(element_node)
            if parent is not None:
                element_node.edge_data = {"data": dict(source=str(parent.id), target=str(element_node.id), cmp_op=cmp_op)}
                if is_left:
                    parent.left = element_node
                else:
                    parent.right = element_node

            if node.cmp_xy is None:
                return
            x, y = [chr(ord("a") + x) for x in node.cmp_xy]
            for op, child in zip("<>", [node.left, node.right]):
                if child is not None:
                    is_left = op == "<"
                    dfs(child, element_node, edge_data, is_left, f"{x}{op}{y}", depth + 1)

        dfs(root)


class Elements:
    cached: dict[tuple[int, int], ElementHolder] = {}
    cached_lock = Lock()

    def __init__(self, sorting_algorithm_i: int, N: int, visiblity_state: Optional[str], **kwargs) -> None:
        self.element_holder = self.get_element_holder(sorting_algorithm_i, N)
        if visiblity_state is not None:
            self.visiblity_state = self.decode_visiblity(visiblity_state)
        else:
            self.visiblity_state = np.array([0], dtype=np.int32)
            self.expand_childs(0)

    @classmethod
    def get_element_holder(cls, sorting_algorithm_i: int, N: int, **kwargs) -> ElementHolder:
        key = (sorting_algorithm_i, N)
        with cls.cached_lock:
            if key not in cls.cached:
                cls.cached[key] = ElementHolder(sorting_algorithms[sorting_algorithm_i][1], N)
            return cls.cached[key]

    def get_visiblity_state(self) -> str:
        return self.encode_visiblity(self.visiblity_state)

    @staticmethod
    def encode_visiblity(visiblity: np.ndarray) -> str:
        return base64.b64encode(zlib.compress(visiblity.tobytes())).decode()

    @staticmethod
    def decode_visiblity(visiblity: str) -> np.ndarray:
        return np.frombuffer(zlib.decompress(base64.b64decode(visiblity)), dtype=np.int32)

    def node_visiblity(self, id: int) -> bool:
        i = self.visiblity_state.searchsorted(id)
        return i < len(self.visiblity_state) and self.visiblity_state[i] == id

    def node_has_hidden_child(self, id: int) -> bool:
        node = self.element_holder.element_nodes[id]
        for child in (node.left, node.right):
            if child is not None and not self.node_visiblity(child.id):
                return True
        return False

    def node_is_leaf(self, id: int) -> bool:
        node = self.element_holder.element_nodes[id]
        return node.left is None and node.right is None

    def expand_childs(self, id: int) -> None:
        update: list[int] = []

        def dfs(node: ElementNode, depth: int) -> None:
            for child in (node.left, node.right):
                if child is not None:
                    if not self.node_visiblity(child.id):
                        update.append(child.id)
                    if depth < DISPLAY_DEPTH:
                        dfs(child, depth + 1)

        dfs(self.element_holder.element_nodes[id], 1)
        self.visiblity_state = snp.merge(self.visiblity_state, np.array(update, dtype=np.int32))

    def hide_childs(self, id: int) -> None:
        deletes: list[int] = []

        def dfs(node: ElementNode) -> None:
            for child in (node.left, node.right):
                if child is not None:
                    i = self.visiblity_state.searchsorted(child.id)
                    if i < len(self.visiblity_state) and self.visiblity_state[i] == child.id:
                        deletes.append(i)
                        dfs(child)

        dfs(self.element_holder.element_nodes[id])
        self.visiblity_state = np.delete(self.visiblity_state, deletes)

    def on_tap_node(self, id: int) -> None:
        if self.node_has_hidden_child(id):
            self.expand_childs(id)
        else:
            self.hide_childs(id)

    def expand_all(self) -> bool:
        elem: list[int] = []
        Q = deque([self.element_holder.element_nodes[0]])
        tot = 1
        while Q and tot < MAX_ELEMENTS:
            node = Q.popleft()
            elem.append(node.id)
            for child in (node.left, node.right):
                if tot < MAX_ELEMENTS and child is not None:
                    tot += 1
                    Q.append(child)
        self.visiblity_state = np.array(elem, dtype=np.int32)
        self.visiblity_state.sort()
        return tot < MAX_ELEMENTS

    def visible_elements(self) -> list[dict]:
        ret = []
        for id in self.visiblity_state:
            node: ElementNode = self.element_holder.element_nodes[id]
            node_data = node.node_data
            if self.node_is_leaf(id):
                node_data["classes"] = "is_leaf"
            elif self.node_has_hidden_child(id):
                node_data["classes"] = "has_hidden_child"
            else:
                node_data["classes"] = ""
            ret.append(node_data)
            if node.edge_data is not None:
                ret.append(node.edge_data)
        return ret
