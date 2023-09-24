from collections.abc import Callable
from functools import cmp_to_key
from itertools import permutations
from typing import Optional


class DecisionTreeNode:
    tot = 0

    def __init__(self) -> None:
        DecisionTreeNode.tot += 1
        self.id: int = DecisionTreeNode.tot
        self.arr: Optional[list[int]] = None
        self.cmp_xy: Optional[tuple[int, int]] = None
        self.actuals: list[list[int]] = []
        self.left: Optional[DecisionTreeNode] = None
        self.right: Optional[DecisionTreeNode] = None

    def get_arr(self) -> str:
        return "(" + ",".join(chr(ord("a") + x) for x in self.arr) + ")"

    def get_actuals(self) -> str:
        return " ".join("(" + ",".join(map(str, x)) + ")" for x in self.actuals)

    @classmethod
    def reset_id(cls) -> None:
        cls.tot = 0

    __slots__ = ["id", "arr", "cmp_xy", "actuals", "left", "right"]


class NonDeterministicError(Exception):
    def __init__(self) -> None:
        super().__init__("Non-deterministic sorting algorithm")


class InvalidSortingAlgorithmError(Exception):
    def __init__(self) -> None:
        super().__init__("Invalid sorting algorithm")


def decision_tree(sorting_func: Callable[[list], None], N: int) -> DecisionTreeNode:
    root = DecisionTreeNode()

    def cmp(x: int, y: int) -> int:
        if x > y:
            return -cmp(y, x)
        cur_cmp_xy = (x, y, actual[x] < actual[y])
        if not (arrs and cur_cmp_xy == cmp_xys[-1]):
            arrs.append([x for _, x in arr])
            cmp_xys.append((x, y, actual[x] < actual[y]))
        return 1 if actual[x] > actual[y] else -1 if actual[x] < actual[y] else 0

    key = cmp_to_key(cmp)
    for actual in permutations(range(1, N + 1)):
        arrs = []
        cmp_xys = []
        arr = [(key(x), x) for x in range(N)]
        sorting_func(arr)
        if any(actual[x] != y for (_, x), y in zip(arr, range(1, N + 1))):
            raise InvalidSortingAlgorithmError
        arrs.append([x for _, x in arr])
        cmp_xys.append(None)
        node = root
        for arr, cmp_xy in zip(arrs, cmp_xys):
            if node.arr is None:
                node.arr = arr
            elif node.arr != arr:
                raise NonDeterministicError
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
                    node.left = DecisionTreeNode()
                node = node.left
            else:
                if node.right is None:
                    node.right = DecisionTreeNode()
                node = node.right

    return root


def print_tree(node: DecisionTreeNode, level: int = 0, op: str = "") -> None:
    if node.cmp_xy is not None:
        x, y = [chr(ord("a") + x) for x in node.cmp_xy]

    if node.left is not None:
        print_tree(node.left, level + 1, f"[{x}<{y}]")
    print("\n" + " " * (level * 8) + f"-{op}->", node.get_arr(), node.actuals, "\n")
    if node.right is not None:
        print_tree(node.right, level + 1, f"[{x}>{y}]")


def visualize_tree(node: DecisionTreeNode, name: str = "decision_tree"):
    from graphviz import Digraph

    dot = Digraph(graph_attr={"overlap": "false"})

    def dfs(node: DecisionTreeNode):
        dot.node(str(node.id), node.get_arr(), xlabel=node.get_actuals())
        if node.cmp_xy is None:
            return
        x, y = [chr(ord("a") + x) for x in node.cmp_xy]
        for op, child in zip("<>", [node.left, node.right]):
            if child is not None:
                dot.edge(str(node.id), str(child.id), label=f"{x}{op}{y}")
                dfs(child)

    dfs(node)
    dot.render(f"{name}.gv", view=True)


if __name__ == "__main__":
    from sorting_algorithms import *

    N = 3
    tree = decision_tree(bubble_sort, N)
    # tree = decision_tree(quick_sort, N)
    # tree = decision_tree(LomutoQS, N)
    # tree = decision_tree(merge_sort, N)
    print_tree(tree)
    visualize_tree(tree)
