from collections.abc import Iterable
from math import factorial

from ..CmpAlgorithm import CmpAlgorithm


def bubble_max(arr: list) -> None:
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]


def _validator(arr: Iterable[int]) -> bool:
    arr = list(arr)
    if arr[-1] != len(arr) - 1:
        return False
    return all(i == v for i, v in enumerate(sorted(arr)))


algorithm = CmpAlgorithm(
    "bubble max",
    bubble_max,
    9,
    output_total=lambda N: factorial(N - 1),
    validator=_validator,
)
