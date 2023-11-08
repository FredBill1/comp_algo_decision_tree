from ..SortingAlgorithm import SortingAlgorithm


def heap_sort(arr: list) -> None:
    def push_down(i: int, n: int) -> None:
        k = i
        while 2 * k + 1 < n:
            j = 2 * k + 1
            if j + 1 < n and arr[j] < arr[j + 1]:
                j += 1
            if arr[k] < arr[j]:
                arr[k], arr[j] = arr[j], arr[k]
                k = j
            else:
                break

    for i in range(len(arr) // 2, -1, -1):
        push_down(i, len(arr))
    for i in range(len(arr) - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        push_down(0, i)


algorithm = SortingAlgorithm("heap sort", heap_sort, 9)
