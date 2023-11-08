from ..SortingAlgorithm import SortingAlgorithm


def merge_sort(arr: list) -> None:
    def merge(arr: list, l: int, m: int, r: int) -> None:
        result = []
        i, j = l, m + 1
        while i <= m and j <= r:
            if arr[i] < arr[j]:
                result.append(arr[i])
                i += 1
            else:
                result.append(arr[j])
                j += 1
        for i in range(i, m + 1):
            result.append(arr[i])
        for j in range(j, r + 1):
            result.append(arr[j])
        arr[l : r + 1] = result

    def impl(arr: list, l: int, r: int) -> None:
        if l < r:
            m = (l + r) // 2
            impl(arr, l, m)
            impl(arr, m + 1, r)
            merge(arr, l, m, r)

    impl(arr, 0, len(arr) - 1)


algorithm = SortingAlgorithm("merge sort", merge_sort, 9)
