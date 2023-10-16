import base64
import zlib
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import atomics
import numpy as np
import pandas as pd
import sortednp as snp

from Config import *
from decision_tree import DecisionTreeNode, decision_tree
from sorting_algorithms import sorting_algorithms


@dataclass
class ElementNode:
    full_label: str
    parent_id: int = -1
    cmp_op: str = ""
    left_id: int = -1
    right_id: int = -1

    def node_data(self, node_id: int, show_full_labels: bool, classes: str) -> dict:
        return {
            "data": {"id": str(node_id), "label": self.full_label if show_full_labels else ElementNode.crop_label(self.full_label)},
            "classes": classes,
        }

    def edge_data(self, node_id: int) -> dict:
        return {"data": {"source": str(self.parent_id), "target": str(node_id), "label": self.cmp_op}}

    @staticmethod
    def crop_label(full_label: str) -> str:
        if len(full_label) <= LABEL_CROP_LENGTH:
            return full_label
        return full_label[: LABEL_CROP_LENGTH - 3] + "..."


class ElementHolder:
    def __init__(self) -> None:
        self.lock = Lock()
        self.element_nodes = None
        self.progress = atomics.atomic(8, atomics.INT)
        self.set_progress(0, 1)

    def get_progress(self) -> tuple[int, int]:
        x = self.progress.load(atomics.MemoryOrder.RELAXED)
        return x >> 32, x & 0xFFFFFFFF

    def set_progress(self, i: int, total: int):
        self.progress.store((i << 32) | total, atomics.MemoryOrder.RELAXED)

    def initialized(self) -> bool:
        return self.element_nodes is not None

    def initialize(self, sorting_func: Callable[[list], None], N: int) -> None:
        tree_node, self.operation_cnts, node_cnt = decision_tree(sorting_func, N, self.set_progress)

        element_nodes: list[ElementNode] = [element_node := ElementNode(tree_node.get_label())]
        Q: deque[tuple[DecisionTreeNode, int]] = deque([(tree_node, 0)])
        self.set_progress(1, node_cnt)
        while Q:
            tree_node, element_node_id = Q.popleft()
            element_node = element_nodes[element_node_id]
            if tree_node.cmp_xy is None:  # leaf node
                continue
            x, y = [chr(ord("a") + x) for x in tree_node.cmp_xy]
            for is_right, (op, child) in enumerate(zip("<>", [tree_node.left, tree_node.right])):
                if child is None:
                    continue
                child_element_node_id = len(element_nodes)
                child_element_node = ElementNode(child.get_label(), element_node_id, f"{x}{op}{y}")
                element_nodes.append(child_element_node)
                self.set_progress(len(element_nodes), node_cnt)
                if is_right:
                    element_node.right_id = child_element_node_id
                else:
                    element_node.left_id = child_element_node_id
                Q.append((child, child_element_node_id))
        self.element_nodes = pd.DataFrame(element_nodes)

    __slots__ = ["lock", "element_nodes", "operation_cnts", "progress"]


class Elements:
    cached: defaultdict[tuple[int, int], ElementHolder] = defaultdict(ElementHolder)
    cached_lock = Lock()

    def __init__(self, sorting_algorithm_i: int, N: int, visiblity_state: Optional[str], **kwargs) -> None:
        self.element_holder = self.get_element_holder(sorting_algorithm_i, N)
        if visiblity_state is not None:
            self.visiblity_state = self.decode_visiblity(visiblity_state)
        else:
            self.visiblity_state = np.array([0], dtype=np.int32)
            self.expand_children(0)

    @classmethod
    def get_element_holder(cls, sorting_algorithm_i: int, N: int, require_initialize: bool = True, **kwargs) -> ElementHolder:
        with cls.cached_lock:
            element_holder = cls.cached[(sorting_algorithm_i, N)]
        if not require_initialize:
            return element_holder
        with element_holder.lock:
            if not element_holder.initialized():
                name, func = sorting_algorithms[sorting_algorithm_i]
                print(f"init: `{name}` with {N} elements")
                element_holder.initialize(func, N)
                print(f"fin:  `{name}` with {N} elements")
        return element_holder

    def get_visiblity_state(self) -> str:
        return self.encode_visiblity(self.visiblity_state)

    @staticmethod
    def encode_visiblity(visiblity: np.ndarray) -> str:
        return base64.b85encode(zlib.compress(np.diff(visiblity, prepend=0).astype(np.int32, copy=False).tobytes())).decode()

    @staticmethod
    def decode_visiblity(visiblity: str) -> np.ndarray:
        return np.cumsum(np.frombuffer(zlib.decompress(base64.b85decode(visiblity)), dtype=np.int32), dtype=np.int32)

    def node_visiblity(self, id: int) -> bool:
        i = self.visiblity_state.searchsorted(id)
        return i < len(self.visiblity_state) and self.visiblity_state[i] == id

    def node_has_hidden_child(self, id: int) -> bool:
        node = self.element_holder.element_nodes.iloc[id]
        for child_id in (node.left_id, node.right_id):
            if child_id != -1 and not self.node_visiblity(child_id):
                return True
        return False

    def node_is_leaf(self, id: int) -> bool:
        node = self.element_holder.element_nodes.iloc[id]
        return node.left_id == -1 and node.right_id == -1

    def expand_children(self, id: int) -> None:
        update: list[int] = []
        Q = deque([id])
        depth = 1
        while Q:
            for _ in range(len(Q)):
                node_id = Q.popleft()
                node = self.element_holder.element_nodes.iloc[node_id]
                for child_id in (node.left_id, node.right_id):
                    if child_id != -1:
                        update.append(child_id)
                        if depth < DISPLAY_DEPTH:
                            Q.append(child_id)
            depth += 1
        self.visiblity_state = snp.merge(self.visiblity_state, np.array(update, dtype=np.int32), duplicates=snp.DROP)

    def hide_children(self, id: int) -> None:
        deletes: list[int] = []
        Q = deque([id])
        while Q:
            node_id = Q.popleft()
            node = self.element_holder.element_nodes.iloc[node_id]
            for child_id in (node.left_id, node.right_id):
                if child_id != -1:
                    i = self.visiblity_state.searchsorted(child_id)
                    if i < len(self.visiblity_state) and self.visiblity_state[i] == child_id:
                        deletes.append(i)
                        Q.append(child_id)
        self.visiblity_state = np.delete(self.visiblity_state, deletes)

    def on_tap_node(self, id: int) -> None:
        if self.node_is_leaf(id):
            return
        if self.node_has_hidden_child(id):
            self.expand_children(id)
        else:
            self.hide_children(id)

    def expand_all(self) -> bool:
        elem: list[int] = []
        Q = deque([0])
        tot = 1
        while Q and tot < MAX_ELEMENTS:
            node_id = Q.popleft()
            node = self.element_holder.element_nodes.iloc[node_id]
            elem.append(node_id)
            for child_id in (node.left_id, node.right_id):
                if tot < MAX_ELEMENTS and child_id != -1:
                    tot += 1
                    Q.append(child_id)
        self.visiblity_state = np.array(elem, dtype=np.int32)
        return tot < MAX_ELEMENTS

    def visible_elements(self, show_full_labels: bool) -> list[dict]:
        ret = []
        for node_id in self.visiblity_state:
            node = self.element_holder.element_nodes.iloc[node_id]
            classes = "is_leaf" if self.node_is_leaf(node_id) else "has_hidden_child" if self.node_has_hidden_child(node_id) else ""
            ret.append(ElementNode.node_data(node, node_id, show_full_labels, classes))
            if node.parent_id != -1:
                ret.append(ElementNode.edge_data(node, node_id))
        return ret
