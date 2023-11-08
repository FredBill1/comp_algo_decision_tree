from ..SortingAlgorithm import SortingAlgorithm


def hoare_quick_sort(arr: list) -> None:
    def partition(L: list, low: int, high: int) -> int:
        pivot = L[(high - low) // 2 + low]
        i = low - 1
        j = high + 1
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

    def impl(L: list, low: int, high: int) -> None:
        if low < high:
            p = partition(L, low, high)
            impl(L, low, p)
            impl(L, p + 1, high)

    impl(arr, 0, len(arr) - 1)


algorithm = SortingAlgorithm("Hoare quick sort", hoare_quick_sort, 9)
