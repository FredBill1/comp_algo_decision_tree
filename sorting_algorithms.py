def bubble_sort(arr: list) -> None:
    for i in range(len(arr) - 1):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]


def quick_sort(arr: list) -> None:
    def impl(arr: list, l: int, r: int):
        if l >= r:
            return

        pivot = arr[l]
        L, R = l, r
        while l < r:
            while l < r and arr[r] >= pivot:
                r -= 1
            if l < r:
                arr[l] = arr[r]
            while l < r and arr[l] < pivot:
                l += 1
            if l < r:
                arr[r] = arr[l]
        if l != L:
            arr[l] = pivot

        impl(arr, L, l - 1)
        impl(arr, l + 1, R)

    impl(arr, 0, len(arr) - 1)


def LomutoQS(L: list) -> None:
    def Lomuto_partition(L: list, lo: int, hi: int) -> int:
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
            p = Lomuto_partition(L, lo, hi)
            impl(L, lo, p - 1)
            impl(L, p + 1, hi)

    impl(L, 0, len(L) - 1)


def merge_sort(arr: list) -> None:
    def merge(arr: list, l: int, m: int, r: int) -> None:
        L = arr[l : m + 1]
        R = arr[m + 1 : r + 1]
        i = j = 0
        k = l
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

    def impl(arr: list, l: int, r: int) -> None:
        if l < r:
            m = (l + r) // 2
            impl(arr, l, m)
            impl(arr, m + 1, r)
            merge(arr, l, m, r)

    impl(arr, 0, len(arr) - 1)
