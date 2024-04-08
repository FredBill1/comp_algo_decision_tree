"This module operates with Min-Heap"
from collections.abc import Callable, Generator
from functools import cache, partial
from itertools import combinations
from math import comb, factorial
from random import randint, sample
from typing import Generic, Optional, TypeVar

from comp_algo_decision_tree.decision_tree_gen.DecisionTreeNode import DecisionTreeNode

from ..CmpAlgorithm import CmpAlgorithm, IdxVal

T = TypeVar("T")


class Node(Generic[T]):
    def __init__(self, val: T, left: Optional["Node[T]"], right: Optional["Node[T]"], is_empty: bool = False) -> None:
        self.val: T = val
        self.left: Node = self if left is None else left
        self.right: Node = self if right is None else right
        self.is_empty: bool = is_empty

    __slots__ = ("val", "left", "right", "is_empty")


INF = 0x7FFFFFFF
EMPTY = Node(INF, None, None, True)


def heap_map(f: Callable[[int], T], node: Node[int], map_empty: bool = False) -> Node[T]:
    if node.is_empty:
        return Node(f(node.val), EMPTY, EMPTY, True) if map_empty else EMPTY
    return Node(f(node.val), heap_map(f, node.left), heap_map(f, node.right))


def heap_map_enumerate(f: Callable[[IdxVal], T], node: Node[int], map_empty: bool = False) -> Node[T]:
    def dfs(node: Node, idx: int) -> Node:
        if node.is_empty:
            return Node(f(IdxVal(idx, node.val)), EMPTY, EMPTY, True) if map_empty else EMPTY
        return Node(f(IdxVal(idx, node.val)), dfs(node.left, (idx << 1) + 1), dfs(node.right, (idx << 1) + 2))

    return dfs(node, 0)


# @cache
def heaps_total(N: int) -> int:
    # if N <= 1:
    #     return 1
    # return sum(comb(N - 1, L) * heaps_total(L) * heaps_total(N - L - 1) for L in range(N))
    return factorial(N)


@cache
def heaps(N: int) -> list[Node[int]]:
    if N < 1:
        return [EMPTY]
    if N == 1:
        return [Node(0, EMPTY, EMPTY)]
    res: list[Node[int]] = []
    for L in range(N):
        R = N - L - 1
        for map_l in combinations(range(1, N), L):
            map_r = tuple(filter(lambda x: x not in map_l, range(1, N)))
            for heap_l in heaps(L):
                heap_l = heap_map(lambda x: map_l[x], heap_l)
                for heap_r in heaps(R):
                    heap_r = heap_map(lambda x: map_r[x], heap_r)
                    res.append(Node(0, heap_l, heap_r))
    return res


def _heap_sample(N: int) -> Node[int]:
    if N < 1:
        return EMPTY
    if N == 1:
        return Node(0, EMPTY, EMPTY)
    L = randint(0, N - 1)
    R = N - L - 1
    map_l = sample(range(1, N), L)
    map_l.sort()
    map_r = tuple(filter(lambda x: x not in map_l, range(N)))
    heap_l, heap_r = heap_map(lambda x: map_l[x], _heap_sample(L)), heap_map(lambda x: map_r[x], _heap_sample(R))
    return Node(0, heap_l, heap_r)


def is_heap(node: Node[int]) -> bool:
    if node.is_empty:
        return True
    if (node.left is not EMPTY and node.left.val < node.val) or (node.right is not EMPTY and node.right.val < node.val):
        return False
    return is_heap(node.left) and is_heap(node.right)


"The semi-heaps here will become real heaps after the root node being pushed down"


def semi_heaps(N: int) -> Generator[Node[int], None, None]:
    if N < 1:
        yield EMPTY
        return
    if N == 1:
        yield Node(0, EMPTY, EMPTY)
        return
    for root in range(N):
        for L in range(N):
            R = N - L - 1
            for map_l in combinations(filter(lambda x: x != root, range(N)), L):
                map_r = tuple(filter(lambda x: x != root and x not in map_l, range(N)))
                for heap_l in heaps(L):
                    heap_l = heap_map(lambda x: map_l[x], heap_l)
                    for heap_r in heaps(R):
                        heap_r = heap_map(lambda x: map_r[x], heap_r)
                        yield Node(root, heap_l, heap_r)


def _semi_heap_sample(N: int) -> Node[int]:
    if N < 1:
        return EMPTY
    if N == 1:
        return Node(0, EMPTY, EMPTY)
    L = randint(0, N - 1)
    R = N - L - 1
    root = randint(0, N - 1)
    map_l = sample(tuple(filter(lambda x: x != root, range(N))), L)
    map_l.sort()
    map_r = tuple(filter(lambda x: x != root and x not in map_l, range(N)))
    heap_l, heap_r = heap_map(lambda x: map_l[x], _heap_sample(L)), heap_map(lambda x: map_r[x], _heap_sample(R))
    return Node(root, heap_l, heap_r)


def semi_heap_sampler(N: int) -> Generator[Node[int], None, None]:
    while True:
        yield _semi_heap_sample(N)


def semi_heaps_total(N: int) -> int:
    return heaps_total(N) * N if N else 1


def push_down_arbitrary(node: Node) -> None:
    nxt = node.left if node.left.val < node.right.val else node.right
    if nxt.val < node.val:
        node.val, nxt.val = nxt.val, node.val
        push_down_arbitrary(nxt)


def idx_converter(node: Node[int]) -> str:
    def dfs(node: Node[int]) -> list[str]:
        if node.is_empty:
            return None
        if node.val == 0:
            return []
        if (ret := dfs(node.left)) is not None:
            ret.append("L")
        elif (ret := dfs(node.right)) is not None:
            ret.append("R")
        return ret

    return "".join(reversed(dfs(node)))


def get_label(tree_node: DecisionTreeNode[Node[int]], idx_use_letter: bool, crop_length: int) -> str:
    def visit(node: Node) -> Generator[Node, None, None]:
        yield node
        if not node.is_empty:
            yield from visit(node.left)
            yield from visit(node.right)

    ret = [f"({tree_node.idx_array})"]
    if (tot_len := len(ret[0])) >= crop_length:
        return ret[0][: crop_length - 3] + "..."
    for val in tree_node.val_arrays:
        ret.append(f" ({','.join('_' if node.is_empty else str(node.val + 1) for node in visit(val))})")
        tot_len += len(ret[-1])
        if crop_length is not None and tot_len >= crop_length:
            return "".join(ret)[: crop_length - 3] + "..."
    return "".join(ret)


algorithm = CmpAlgorithm(
    "push down arbitrary",
    push_down_arbitrary,
    8,
    semi_heaps,
    semi_heaps_total,
    heaps_total,
    semi_heap_sampler,
    lambda node, _: is_heap(node),
    idx_converter,
    lambda N: N <= 3,
    get_label,
    partial(heap_map, map_empty=True),
    partial(heap_map_enumerate, map_empty=True),
)
