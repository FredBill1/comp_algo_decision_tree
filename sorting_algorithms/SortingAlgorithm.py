from collections.abc import Callable, Generator, Iterable, Sequence
from itertools import permutations
from math import factorial
from random import shuffle
from typing import NamedTuple


def _sampler(N: int) -> Generator[list[int], None, None]:
    arr = list(range(N))
    while True:
        shuffle(arr)
        yield arr


class SortingAlgorithm(NamedTuple):
    name: str
    func: Callable[[list], None]
    max_N: int
    generator: Callable[[int], Iterable[Sequence[int]]] = lambda n: permutations(range(n))
    input_total: Callable[[int], int] = factorial
    output_total: Callable[[int], int] = lambda _: 1
    sampler: Callable[[int], Generator[Sequence[int], None, None]] = _sampler
    validator: Callable[[Iterable[int]], bool] = lambda arr: all(i == v for i, v in enumerate(arr))
