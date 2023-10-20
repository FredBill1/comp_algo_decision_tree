import base64
import ctypes
import traceback
import zlib
from collections import defaultdict, deque
from threading import Lock
from typing import Optional

import atomics
import numpy as np
import sortednp as snp

from Config import *
from decision_tree import DecisionTreeNode, decision_tree
from sorting_algorithms import SortingAlgorithm, sorting_algorithms


def node_data(node: DecisionTreeNode, show_full_labels: bool, classes: str) -> dict:
    label = node.label(LABEL_MAX_LENGTH if show_full_labels else LABEL_CROP_LENGTH)
    return {"data": {"id": str(id(node)), "label": label}, "classes": classes}


def node_from_id(node_id: int) -> DecisionTreeNode:
    return ctypes.cast(node_id, ctypes.py_object).value


class NodeHolder:
    def __init__(self) -> None:
        self.lock = Lock()
        self.progress = atomics.atomic(8, atomics.INT)
        self.initialize_scheduled = atomics.atomic(1, atomics.BYTES)
        self.initialized_flag = atomics.atomic(1, atomics.BYTES)
        self.set_progress(0, 1)

    def get_and_set_initialize_scheduled(self) -> bool:
        return self.initialize_scheduled.bit_test_set(0, atomics.MemoryOrder.RELAXED)

    def get_progress(self) -> tuple[int, int]:
        x = self.progress.load(atomics.MemoryOrder.RELAXED)
        return x >> 32, x & 0xFFFFFFFF

    def set_progress(self, i: int, total: int):
        self.progress.store((i << 32) | total, atomics.MemoryOrder.RELAXED)

    def initialize(self, sorting_algorithm_i: int, N: int) -> None:
        with self.lock:
            sorting_algo = sorting_algorithms[sorting_algorithm_i]
            print(f"init: `{sorting_algo.name}` with {N} elements")
            self._initialize(sorting_algo, N)
            print(f"fin:  `{sorting_algo.name}` with {N} elements")
            self.initialized_flag.store(b"\x01", atomics.MemoryOrder.RELEASE)

    def wait_until_initialized(self) -> None:
        if self.initialized_flag.load(atomics.MemoryOrder.RELAXED):
            return
        with self.lock:
            pass

    def _initialize(self, sorting_algo: SortingAlgorithm, N: int) -> None:
        try:
            self.root, self.operation_cnts, self.node_cnt = decision_tree(sorting_algo, N, self.set_progress)
        except Exception as e:
            traceback.print_exc()
            self.initialize_scheduled.store(b"\x00", atomics.MemoryOrder.RELEASE)
            raise e

    __slots__ = ["lock", "root", "operation_cnts", "node_cnt", "progress", "initialize_scheduled", "initialized_flag"]


class Nodes:
    cached: defaultdict[tuple[int, int], NodeHolder] = defaultdict(NodeHolder)
    cached_lock = Lock()

    def __init__(self, element_holder: NodeHolder, visiblity_state: Optional[str]) -> None:
        self.element_holder = element_holder
        if visiblity_state is not None:
            self.visiblity_state = self.decode_visiblity(visiblity_state)
        else:
            self.visiblity_state = np.array([id(self.element_holder.root)], dtype=np.uintp)
            self.expand_children(self.element_holder.root)

    @classmethod
    def get_node_holder(cls, sorting_algorithm_i: int, N: int) -> NodeHolder:
        with cls.cached_lock:
            return cls.cached[(sorting_algorithm_i, N)]

    def get_visiblity_state(self) -> str:
        return self.encode_visiblity(self.visiblity_state)

    @staticmethod
    def encode_visiblity(visiblity: np.ndarray) -> str:
        return base64.b85encode(zlib.compress(visiblity.tobytes())).decode()

    @staticmethod
    def decode_visiblity(visiblity: str) -> np.ndarray:
        return np.frombuffer(zlib.decompress(base64.b85decode(visiblity)), dtype=np.uintp)

    def node_visiblity(self, node_id: int) -> bool:
        i = self.visiblity_state.searchsorted(node_id)
        return i < len(self.visiblity_state) and self.visiblity_state[i] == node_id

    def node_has_hidden_child(self, node: DecisionTreeNode) -> bool:
        for child in (node.left, node.right):
            if child is not None and not self.node_visiblity(id(child)):
                return True
        return False

    def node_is_leaf(self, node: DecisionTreeNode) -> bool:
        return node.left is None and node.right is None

    def expand_children(self, node: DecisionTreeNode) -> None:
        update: list[int] = []
        Q = deque([node])
        depth = 1
        while Q:
            for _ in range(len(Q)):
                node = Q.popleft()
                for child in (node.left, node.right):
                    if child is not None:
                        update.append(id(child))
                        if depth < DISPLAY_DEPTH:
                            Q.append(child)
            depth += 1
        update = np.array(update, dtype=np.uintp)
        update.sort()
        self.visiblity_state = snp.merge(self.visiblity_state, update, duplicates=snp.DROP)

    def hide_children(self, node: DecisionTreeNode) -> None:
        deletes: list[int] = []
        Q = deque([node])
        while Q:
            node = Q.popleft()
            for child in (node.left, node.right):
                if child is not None:
                    child_id = id(child)
                    i = self.visiblity_state.searchsorted(child_id)
                    if i < len(self.visiblity_state) and self.visiblity_state[i] == child_id:
                        deletes.append(i)
                        Q.append(child)
        self.visiblity_state = np.delete(self.visiblity_state, deletes)

    def on_tap_node(self, node_id: int) -> None:
        node = node_from_id(node_id)
        if self.node_is_leaf(node):
            return
        if self.node_has_hidden_child(node):
            self.expand_children(node)
        else:
            self.hide_children(node)

    def expand_all(self) -> bool:
        elem: list[int] = []
        Q = deque([self.element_holder.root])
        tot = 1
        while Q and tot < MAX_ELEMENTS:
            node = Q.popleft()
            elem.append(id(node))
            for child in (node.left, node.right):
                if tot < MAX_ELEMENTS and child is not None:
                    tot += 1
                    Q.append(child)
        self.visiblity_state = np.array(elem, dtype=np.uintp)
        self.visiblity_state.sort()
        return tot < MAX_ELEMENTS

    def visible_elements(self, show_full_labels: bool) -> list[dict]:
        ret = []
        for node_id in self.visiblity_state:
            node = node_from_id(int(node_id))
            classes = "is_leaf" if self.node_is_leaf(node) else "has_hidden_child" if self.node_has_hidden_child(node) else ""
            ret.append(node_data(node, show_full_labels, classes))
            if node.edge_data is not None:
                ret.append(node.edge_data)
        return ret
