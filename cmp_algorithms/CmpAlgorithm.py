from collections.abc import Callable, Generator, Iterable, Sequence
from itertools import permutations
from math import factorial
from random import shuffle
from typing import NamedTuple

from DecisionTreeNode import DecisionTreeNode


def _sampler(N: int) -> Generator[list[int], None, None]:
    arr = list(range(N))
    while True:
        shuffle(arr)
        yield arr


def _get_label(node: DecisionTreeNode, crop_length: int) -> str:
    ret = ["(" + ",".join(chr(ord("a") + x) if len(node.arr) <= 26 else str(x) for x in node.arr) + ")"]
    tot_len = len(ret[0])
    for actual in node.actuals:
        ret.append(f" ({','.join(str(x + 1) for x in actual)})")
        tot_len += len(ret[-1])
        if crop_length is not None and tot_len >= crop_length:
            return "".join(ret)[: crop_length - 3] + "..."
    return "".join(ret)


class CmpAlgorithm(NamedTuple):
    name: str
    func: Callable[[list], None]
    max_N: int
    generator: Callable[[int], Iterable[Sequence[int]]] = lambda n: permutations(range(n))
    input_total: Callable[[int], int] = factorial
    output_total: Callable[[int], int] = lambda _: 1
    sampler: Callable[[int], Generator[Sequence[int], None, None]] = _sampler
    validator: Callable[[Iterable[int]], bool] = lambda arr: all(i == v for i, v in enumerate(arr))
    get_label: Callable[[DecisionTreeNode, int], str] = _get_label
