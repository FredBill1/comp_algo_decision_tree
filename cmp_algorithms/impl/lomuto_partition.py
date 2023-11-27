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


def output_total(N: int) -> int:
    fact = [1] * (N)
    for i in range(1, N):
        fact[i] = fact[i - 1] * i
    return sum(a * b for a, b in zip(fact, reversed(fact)))


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
