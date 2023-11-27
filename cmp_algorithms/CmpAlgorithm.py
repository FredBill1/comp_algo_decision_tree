from collections.abc import Callable, Generator, Iterable, MutableSequence, Sequence
from itertools import permutations
from math import factorial
from random import shuffle
from typing import Any, NamedTuple, Optional, TypeVar

from DecisionTreeNode import DecisionTreeNode


def _sampler(N: int) -> Generator[list[int], None, None]:
    arr = list(range(N))
    while True:
        shuffle(arr)
        yield arr


def _get_label(node: DecisionTreeNode[Sequence[int]], idx_use_letter: bool, crop_length: int) -> str:
    ret = [f"({','.join(chr(ord('a') + x) if idx_use_letter else str(x) for x in node.idx_array)})"]
    if (tot_len := len(ret[0])) >= crop_length:
        return ret[0][: crop_length - 3] + "..."
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
    func: Callable[[MutableSequence[int]], Optional[Any]]
    max_N: int
    generator: Callable[[int], Iterable[Sequence[int]]] = lambda n: permutations(range(n))
    input_total: Callable[[int], int] = factorial
    output_total: Callable[[int], int] = lambda _: 1
    sampler: Callable[[int], Generator[Sequence[int], None, None]] = _sampler
    validator: Callable[[Sequence[int], Optional[Any]], bool] = lambda arr, _: all(i == v for i, v in enumerate(arr))
    idx_converter: Callable[[Sequence[int]], Any] = lambda arr: arr
    idx_use_letter: Callable[[int], bool] = lambda n: n <= 26
    get_label: Callable[[DecisionTreeNode, bool, int], str] = _get_label
    map: Callable[[Callable[[int], T], Sequence[int]], Sequence[T]] = lambda f, arr: [f(x) for x in arr]
    map_enumerate: Callable[[Callable[[IdxVal], T], Sequence[int]], Sequence[T]] = lambda f, arr: [f(IdxVal(i, x)) for i, x in enumerate(arr)]
