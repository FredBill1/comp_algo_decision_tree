from ..SortingAlgorithm import SortingAlgorithm


def selection_sort(arr: list) -> None:
    for i in range(len(arr) - 1):
        k = i
        for j in range(i + 1, len(arr)):
            if arr[k] > arr[j]:
                k = j
        if k != i:
            arr[k], arr[i] = arr[i], arr[k]


algorithm = SortingAlgorithm("selection sort", selection_sort, 8)
