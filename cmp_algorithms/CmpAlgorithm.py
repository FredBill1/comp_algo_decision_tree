from collections.abc import Callable, Generator, Iterable, MutableSequence, Sequence
from itertools import permutations
from math import factorial
from random import shuffle
from typing import NamedTuple, TypeVar

from DecisionTreeNode import DecisionTreeNode


def _sampler(N: int) -> Generator[list[int], None, None]:
    arr = list(range(N))
    while True:
        shuffle(arr)
        yield arr


def _get_label(node: DecisionTreeNode, crop_length: int) -> str:
    ret = ["(" + ",".join(chr(ord("a") + x) if len(node.idx_array) <= 26 else str(x) for x in node.idx_array) + ")"]
    tot_len = len(ret[0])
    for val_array in node.val_arrays:
        ret.append(f" ({','.join(str(x + 1) for x in val_array)})")
        tot_len += len(ret[-1])
        if crop_length is not None and tot_len >= crop_length:
            return "".join(ret)[: crop_length - 3] + "..."
    return "".join(ret)


class IdxVal(NamedTuple):
    idx: int
    val: int


T = TypeVar("T")


class CmpAlgorithm(NamedTuple):
    name: str
    func: Callable[[MutableSequence[int]], None]
    max_N: int
    generator: Callable[[int], Iterable[Sequence[int]]] = lambda n: permutations(range(n))
    input_total: Callable[[int], int] = factorial
    output_total: Callable[[int], int] = lambda _: 1
    sampler: Callable[[int], Generator[Sequence[int], None, None]] = _sampler
    validator: Callable[[Iterable[int]], bool] = lambda arr: all(i == v for i, v in enumerate(arr))
    get_label: Callable[[DecisionTreeNode, int], str] = _get_label
    map: Callable[[Callable[[int], T], Sequence[int]], Sequence[T]] = lambda f, arr: [f(x) for x in arr]
    map_enumerate: Callable[[Callable[[IdxVal], T], Sequence[int]], Sequence[T]] = lambda f, arr: [f(IdxVal(i, x)) for i, x in enumerate(arr)]
