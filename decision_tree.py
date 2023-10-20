from collections.abc import Callable
from functools import cmp_to_key
from time import thread_time
from typing import Optional

from Config import *
from sorting_algorithms import SortingAlgorithm


class DecisionTreeNode:
    def __init__(self, parent: Optional["DecisionTreeNode"] = None, use_letter: bool = True, is_left: bool = False) -> None:
        self.arr: Optional[list[int]] = None
        self.cmp_xy: Optional[tuple[int, int]] = None
        self.actuals: list[tuple[int, ...]] = []
        self.left: Optional[DecisionTreeNode] = None
        self.right: Optional[DecisionTreeNode] = None
        self.pos = 0.0 if parent is None else parent.pos * 2 + (2 - int(is_left))
        self.edge_data = None
        if parent is not None:
            x, y = [chr(ord("a") + x) if use_letter else f"[{x}]" for x in parent.cmp_xy[:2]]
            self.edge_data = {"data": dict(source=id(parent), target=id(self), cmp_op=f"{x}<{y}" if is_left else f"{x}>{y}")}

    def get_arr(self) -> str:
        return "(" + ",".join(chr(ord("a") + x) if len(self.arr) <= 26 else str(x) for x in self.arr) + ")"

    def label(self, crop_length: Optional[int]) -> str:
        ret = [self.get_arr()]
        tot_len = len(ret[0])
        for actual in self.actuals:
            ret.append(f" ({','.join(str(x + 1) for x in actual)})")
            tot_len += len(ret[-1])
            if crop_length is not None and tot_len >= crop_length:
                return "".join(ret)[: crop_length - 3] + "..."
        return "".join(ret)

    def node_data(self, show_full_labels: bool, classes: str) -> dict:
        label = self.label(LABEL_MAX_LENGTH if show_full_labels else LABEL_CROP_LENGTH)
        return {"data": {"id": id(self), "label": label, "pos": self.pos}, "classes": classes}

    __slots__ = ["arr", "cmp_xy", "actuals", "left", "right", "pos", "edge_data"]


class NonDeterministicError(Exception):
    def __init__(self) -> None:
        super().__init__("Non-deterministic sorting algorithm")


class InvalidSortingAlgorithmError(Exception):
    def __init__(self) -> None:
        super().__init__("Invalid sorting algorithm")


def decision_tree(sorting_algo: SortingAlgorithm, N: int, callback: Optional[Callable[[int, int], None]] = None) -> tuple[DecisionTreeNode, list[int], int]:
    root = DecisionTreeNode()
    node_cnt = 1

    def cmp(x: int, y: int) -> int:
        if x > y:
            return -cmp(y, x)
        cur_cmp_xy = (x, y, actual[x] < actual[y])
        if not (arrs and cur_cmp_xy == cmp_xys[-1]):
            nonlocal operation_cnt
            operation_cnt += 1
            arrs.append([x for _, x in arr])
            cmp_xys.append((x, y, actual[x] < actual[y]))
        return 1 if actual[x] > actual[y] else -1 if actual[x] < actual[y] else 0

    do_sample = N > sorting_algo.max_N
    TOTAL = 1 if do_sample else sorting_algo.total(N)
    if callback is not None:
        if do_sample:
            callback(0, MAX_SAMPLE_TIME_MS)
        else:
            callback(0, TOTAL)
    operation_cnts = []
    key = cmp_to_key(cmp)
    start_time = thread_time()
    for I, actual in enumerate(sorting_algo.sampler(N) if do_sample else sorting_algo.generator(N)):
        arrs = []
        cmp_xys = []
        arr = [(key(x), x) for x in range(N)]
        operation_cnt = 0
        sorting_algo.func(arr)
        if not sorting_algo.validator(actual[x] for _, x in arr):
            raise InvalidSortingAlgorithmError
        operation_cnts.append(operation_cnt)
        arrs.append([x for _, x in arr])
        cmp_xys.append(None)
        node = root
        for arr, cmp_xy in zip(arrs, cmp_xys):
            if node.arr is None:
                node.arr = arr
            elif node.arr != arr:
                raise NonDeterministicError
            if len(node.actuals) < ACTUALS_MAX_LENGTH:
                node.actuals.append(actual)

            if cmp_xy is None:
                if node.cmp_xy is not None:
                    raise NonDeterministicError
                continue
            if node.cmp_xy is None:
                node.cmp_xy = cmp_xy[:2]
            elif node.cmp_xy != cmp_xy[:2]:
                raise NonDeterministicError

            if cmp_xy[2]:  # x < y
                if node.left is None:
                    node.left = DecisionTreeNode(node, len(arr) <= 26, True)
                    node_cnt += 1
                node = node.left
            else:
                if node.right is None:
                    node.right = DecisionTreeNode(node, len(arr) <= 26, False)
                    node_cnt += 1
                node = node.right
        if do_sample and (cur_time := int((thread_time() - start_time) * 1000)) >= MAX_SAMPLE_TIME_MS:
            callback(MAX_SAMPLE_TIME_MS, MAX_SAMPLE_TIME_MS)
            break
        if callback is not None:
            if do_sample:
                callback(cur_time, MAX_SAMPLE_TIME_MS)
            else:
                callback(I + 1, TOTAL)
    return root, operation_cnts, node_cnt


def print_tree(node: DecisionTreeNode, level: int = 0, op: str = "") -> None:
    if node.cmp_xy is not None:
        x, y = [chr(ord("a") + x) for x in node.cmp_xy]

    if node.left is not None:
        print_tree(node.left, level + 1, f"[{x}<{y}]")
    print("\n" + " " * (level * 8) + f"-{op}->", node.get_arr(), node.actuals, "\n")
    if node.right is not None:
        print_tree(node.right, level + 1, f"[{x}>{y}]")


if __name__ == "__main__":
    from sorting_algorithms import sorting_algorithms

    N = 3
    tree, operation_cnts, node_cnt = decision_tree(sorting_algorithms[0], N)
    # tree = decision_tree(quick_sort, N)
    # tree = decision_tree(LomutoQS, N)
    # tree = decision_tree(merge_sort, N)
    print_tree(tree)
