from decimal import Decimal
from functools import cmp_to_key
from itertools import product
from math import log2, nan
from multiprocessing import Pool
from pathlib import Path
from random import Random
from time import thread_time
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .cmp_algorithms.cmp_algorithms import cmp_algorithms
from .cmp_algorithms.CmpAlgorithm import CmpAlgorithm, IdxVal
from .Config import *

RESULT_DIR = Path("logs/statistics.csv")


def to_displayable_int(x: int) -> str:
    return str(x) if x < 1e9 else f"{Decimal(x):.2e}"


def get_avg_operation_cnt(cmp_algorithm: CmpAlgorithm, N: int) -> tuple[int, int, float]:
    def cmp(x: IdxVal, y: IdxVal) -> int:
        if x.idx == y.idx:
            return 0
        if x.idx > y.idx:
            return -cmp(y, x)
        nonlocal operation_cnt, last_cmp
        if (cur_cmp := (x.idx, y.idx)) != last_cmp:
            last_cmp = cur_cmp
            operation_cnt += 1
        return x.val - y.val

    key = cmp_to_key(cmp)

    do_sample = N > cmp_algorithm.max_N
    total = 0
    best = float("inf")
    worst = 0
    avg_sum = 0
    if do_sample:
        start_time = thread_time()
    r = Random(SAMPLE_SEED)
    for val_array in cmp_algorithm.sampler(N, r) if do_sample else cmp_algorithm.generator(N):
        idx_array = cmp_algorithm.map_enumerate(key, val_array)
        last_cmp: Optional[tuple[int, int]] = None
        operation_cnt = 0
        cmp_algorithm.func(idx_array)
        avg_sum += operation_cnt
        total += 1
        best = min(best, operation_cnt)
        worst = max(worst, operation_cnt)
        if do_sample and int((thread_time() - start_time) * 1000) > MAX_SAMPLE_TIME_MS:
            break

    return best, worst, avg_sum / total


def _work(args: tuple[int, int]) -> str:
    cmp_algorithm_idx, N = args
    cmp_algorithm = cmp_algorithms[cmp_algorithm_idx]
    best, worst, avg = get_avg_operation_cnt(cmp_algorithm, N)
    input_total = cmp_algorithm.input_total(N)
    output_total = cmp_algorithm.output_total(N)
    lower_bound = log2(input_total) - log2(output_total)
    ratio = nan if input_total <= output_total else avg / lower_bound
    return ",".join(map(str, (cmp_algorithm.name, N, to_displayable_int(input_total), to_displayable_int(output_total), lower_bound, best, worst, avg, ratio)))


def generate_statistics() -> None:
    Ns = list(range(3, 10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100))
    tasks = list(product(range(len(cmp_algorithms)), Ns))
    RESULT_DIR.parent.mkdir(parents=True, exist_ok=True)
    with Pool() as pool, open(RESULT_DIR, "w") as f:
        f.write("name,N,input,output,lower bound,best,worst,avg,ratio\n")
        for result in tqdm(pool.imap_unordered(_work, tasks), total=len(tasks)):
            f.write(result + "\n")
            f.flush()


def sort_result() -> None:
    df = pd.read_csv(RESULT_DIR)
    df = df.sort_values(["name", "N"])
    df.to_csv(RESULT_DIR, index=False)
    for name, group in df.groupby("name"):
        group.drop(columns=["name"]).to_csv(RESULT_DIR.parent / f"{name}.csv", index=False)


if __name__ == "__main__":
    generate_statistics()
    sort_result()
