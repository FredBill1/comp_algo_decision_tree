from ..CmpAlgorithm import CmpAlgorithm
from .lomuto_partition import output_total


def hoare_partition(L: list) -> int:
    pivot = L[len(L) // 2]
    i = -1
    j = len(L)
    while True:
        i += 1
        while L[i] < pivot:
            i += 1
        j -= 1
        while L[j] > pivot:
            j -= 1
        if i >= j:
            return j
        L[i], L[j] = L[j], L[i]


def validator(L: list, ret: int) -> bool:
    return ret == len(L) - 1 or max(L[: ret + 1]) < min(L[ret + 1 :])


algorithm = CmpAlgorithm(
    "Hoare partition",
    hoare_partition,
    9,
    output_total=output_total,
    validator=validator,
)
