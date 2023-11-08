from importlib import import_module
from pathlib import Path

from .SortingAlgorithm import SortingAlgorithm

sorting_algorithms: list[SortingAlgorithm] = []
for file in (Path(__file__).parent / "impl").glob("*.py"):
    module = import_module(f".{file.stem}", package="sorting_algorithms.impl")
    sorting_algorithms.append(module.algorithm)
