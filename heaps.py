"This module operates with Min-Heap"
from collections.abc import Generator, Sequence
from itertools import combinations
from random import randint, sample
from threading import Lock

import atomics


class _Cache:
    def __init__(self) -> None:
        self.heap: list[list[int]] = []
        self.lock = Lock()
        self.initialized = atomics.atomic(1, atomics.BYTES)

    __slots__ = ("heap", "lock", "initialized")


max_N = 12

# https://oeis.org/A056971
_totals = (1, 1, 1, 2, 3, 8, 20, 80, 210, 896, 3360, 19200, 79200, 506880, 2745600, 21964800, 108108000, 820019200, 5227622400, 48881664000)
_totals = _totals[: max_N + 1]
_heap_cache = [_Cache() for _ in range(len(_totals))]
_heap_cache[0].heap.append([])
_heap_cache[1].heap.append([0])


def heaps(N: int) -> list[list[int]]:
    cache = _heap_cache[N]
    if cache.initialized.load(atomics.MemoryOrder.RELAXED) != b"\x01":
        with cache.lock:
            if not cache.heap:
                cache.heap = _construct(N)
                cache.initialized.store(b"\x01", atomics.MemoryOrder.RELEASE)
    return cache.heap


def heaps_total(N: int) -> int | float:
    return _totals[N] if N < len(_totals) else float("inf")


def _sub_heap_sizes(N: int) -> tuple[int, int]:
    t = 1 << (N.bit_length() - 2)
    M = 1 + N - (t << 1)
    L = t - 1 + min(t, M)
    R = t - 1 + max(0, M - t)
    return L, R


def _construct(N: int) -> list[list[int]]:
    L, R = _sub_heap_sizes(N)
    res = []
    for map_l in combinations(range(1, N), L):
        map_r = tuple(filter(lambda x: x not in map_l, range(1, N)))
        for heap_l in heaps(L):
            heap_l = [map_l[x] for x in heap_l]
            for heap_r in heaps(R):
                heap_r = [map_r[x] for x in heap_r]
                res.append(_merge_heap(0, heap_l, heap_r))
    return res


def _heap_sample(N: int) -> list[int]:
    if N <= 1:
        return list(range(N))
    L, R = _sub_heap_sizes(N)
    map_l = sample(range(1, N), L)
    map_l.sort()
    map_r = tuple(filter(lambda x: x not in map_l, range(N)))
    heap_l, heap_r = [map_l[x] for x in _heap_sample(L)], [map_r[x] for x in _heap_sample(R)]
    return _merge_heap(0, heap_l, heap_r)


def is_heap(arr: Sequence[int]) -> bool:
    for i in range(1, len(arr)):
        if arr[(i - 1) >> 1] > arr[i]:
            return False
    return True


"The semi-heaps here will become real heaps after the root node being pushed down"


def _merge_heap(root: int, heap_l: list[int], heap_r: list[int]) -> list[int]:
    cur = [root]
    i = 1
    while i - 1 <= len(heap_l):
        cur += heap_l[i - 1 : min((i << 1) - 1, len(heap_l))]
        cur += heap_r[i - 1 : min((i << 1) - 1, len(heap_r))]
        i <<= 1
    return cur


def semi_heaps(N: int) -> Generator[list[int], None, None]:
    if N <= 1:
        yield list(range(N))
        return
    L, R = _sub_heap_sizes(N)
    for root in range(N):
        for map_l in combinations(filter(lambda x: x != root, range(N)), L):
            map_r = tuple(filter(lambda x: x != root and x not in map_l, range(N)))
            for heap_l in heaps(L):
                heap_l = [map_l[x] for x in heap_l]
                for heap_r in heaps(R):
                    heap_r = [map_r[x] for x in heap_r]
                    yield _merge_heap(root, heap_l, heap_r)


def _semi_heap_sample(N: int) -> list[int]:
    if N <= 1:
        return list(range(N))
    L, R = _sub_heap_sizes(N)
    root = randint(0, N - 1)
    map_l = sample(tuple(filter(lambda x: x != root, range(N))), L)
    map_l.sort()
    map_r = tuple(filter(lambda x: x != root and x not in map_l, range(N)))
    heap_l, heap_r = [map_l[x] for x in _heap_sample(L)], [map_r[x] for x in _heap_sample(R)]
    return _merge_heap(root, heap_l, heap_r)


def semi_heap_sampler(N: int) -> Generator[list[int], None, None]:
    while True:
        yield _semi_heap_sample(N)


def semi_heaps_total(N: int) -> int:
    return heaps_total(N) * N if N else 1


def push_down(arr: list) -> None:
    k, n = 0, len(arr)
    while 2 * k + 1 < n:
        j = 2 * k + 1
        if j + 1 < n and arr[j] > arr[j + 1]:
            j += 1
        if arr[k] > arr[j]:
            arr[k], arr[j] = arr[j], arr[k]
            k = j
        else:
            break


if __name__ == "__main__":
    print(heaps(5))
    for N in range(15):
        h = heaps(N)
        print(f"{N}: {heaps_total(N)}; {len(h)} heaps")
        if heaps_total(N) != len(h):
            print("Total number of heaps is incorrect!")
            quit(-1)
        if not all(map(is_heap, h)):
            print("Not all heaps are valid!")
            quit(-1)
    print("All heaps are valid!")
