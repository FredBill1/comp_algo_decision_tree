from ..SortingAlgorithm import SortingAlgorithm


def insertion_sort(arr: list) -> None:
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j - 1] > arr[j]:
            arr[j - 1], arr[j] = arr[j], arr[j - 1]
            j -= 1


algorithm = SortingAlgorithm("insertion sort", insertion_sort, 9)
