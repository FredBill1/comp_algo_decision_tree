from math import factorial

from ..CmpAlgorithm import CmpAlgorithm
from .push_down import heaps_total, is_heap


def heapify(arr: list) -> None:
    def push_down(i: int, n: int) -> None:
        k = i
        while 2 * k + 1 < n:
            j = 2 * k + 1
            if j + 1 < n and arr[j] > arr[j + 1]:
                j += 1
            if arr[k] > arr[j]:
                arr[k], arr[j] = arr[j], arr[k]
                k = j
            else:
                break

    for i in range(len(arr) // 2, -1, -1):
        push_down(i, len(arr))


algorithm = CmpAlgorithm(
    "heapify",
    heapify,
    max_N=9,
    input_total=factorial,
    output_total=heaps_total,
    validator=lambda arr, _: is_heap(arr),
)
