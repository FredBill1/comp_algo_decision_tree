from threading import Lock

from ..CmpAlgorithm import CmpAlgorithm


def lomuto_partition(L: list) -> int:
    hi = len(L) - 1
    pivot = L[hi]
    i = 0
    for j in range(hi):
        if L[j] <= pivot:
            L[i], L[j] = L[j], L[i]
            i += 1
    L[i], L[hi] = L[hi], L[i]
    return i


class A356291:
    "https://oeis.org/A356291"

    def __init__(self):
        self.F = 1
        self.R = [0]
        self.C = [1]
        self._lock = Lock()

    def __call__(self, i: int) -> int:
        with self._lock:
            if i < len(self.R):
                return self.R[i]
            self.C += [0] * (i - len(self.C) + 1)
            for n in range(len(self.R), i + 1):
                self.F *= n
                for k in range(n, 0, -1):
                    self.C[k] = self.C[k - 1] * k
                self.C[0] = -sum(self.C[k] for k in range(1, n + 1))
                self.R.append(self.F + self.C[0])
            return self.R[i]


_a356291 = A356291()


def output_total(N: int) -> int:
    if N <= 1:
        return 1
    return _a356291(N)


def validator(L: list, i: int) -> bool:
    if not 0 <= i < len(L):
        return False
    pivot = L[i]
    return all(L[j] <= pivot for j in range(i)) and all(L[j] >= pivot for j in range(i + 1, len(L)))


algorithm = CmpAlgorithm(
    "Lomuto partition",
    lomuto_partition,
    9,
    output_total=output_total,
    validator=validator,
)
