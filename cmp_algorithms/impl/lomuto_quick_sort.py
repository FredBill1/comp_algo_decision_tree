from ..CmpAlgorithm import CmpAlgorithm


def lomuto_quick_sort(arr: list) -> None:
    def lomuto_partition(L: list, lo: int, hi: int) -> int:
        pivot = L[hi]
        i = lo
        for j in range(lo, hi):
            if L[j] <= pivot:
                L[i], L[j] = L[j], L[i]
                i += 1
        L[i], L[hi] = L[hi], L[i]
        return i

    def impl(L: list, lo: int, hi: int) -> None:
        if lo < hi:
            p = lomuto_partition(L, lo, hi)
            impl(L, lo, p - 1)
            impl(L, p + 1, hi)

    impl(arr, 0, len(arr) - 1)


algorithm = CmpAlgorithm("Lomuto quick sort", lomuto_quick_sort, 9)
