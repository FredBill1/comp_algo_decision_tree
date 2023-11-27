from collections.abc import Sequence
from math import factorial

from ..CmpAlgorithm import CmpAlgorithm


def find_max(arr: list) -> None:
    for i in range(1, len(arr)):
        if arr[i] > arr[0]:
            arr[0], arr[i] = arr[i], arr[0]


def _validator(arr: Sequence[int], _) -> bool:
    if arr[0] != len(arr) - 1:
        return False
    return all(i == v for i, v in enumerate(sorted(arr)))


algorithm = CmpAlgorithm(
    "find max",
    find_max,
    9,
    output_total=lambda N: factorial(N - 1),
    validator=_validator,
)
