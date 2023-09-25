def bubble_sort(arr: list) -> None:
    for i in range(len(arr) - 2, -1, -1):
        for j in range(i + 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def insertion_sort(arr: list) -> None:
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j - 1] > arr[j]:
            arr[j - 1], arr[j] = arr[j], arr[j - 1]
            j -= 1


def quick_sort(arr: list) -> None:
    def partition(L: list, low: int, high: int) -> int:
        pivot = L[low]
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


def LomutoQS(arr: list) -> None:
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

    impl(arr, 0, len(arr) - 1)


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


sorting_algorithms = [
    ("bubble sort", bubble_sort),
    ("insertion sort", insertion_sort),
    ("quick sort", quick_sort),
    ("Lomuto quick sort", LomutoQS),
    ("heap sort", heap_sort),
    ("merge sort", merge_sort),
]
