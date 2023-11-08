from ..SortingAlgorithm import SortingAlgorithm


def bubble_sort(arr: list) -> None:
    for i in range(len(arr) - 2, -1, -1):
        for j in range(i + 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


algorithm = SortingAlgorithm("bubble sort", bubble_sort, 8)
