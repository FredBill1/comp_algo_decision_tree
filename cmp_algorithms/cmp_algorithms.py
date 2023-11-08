from importlib import import_module
from pathlib import Path

from .CmpAlgorithm import CmpAlgorithm

cmp_algorithms: list[CmpAlgorithm] = []
for file in (Path(__file__).parent / "impl").glob("*.py"):
    module = import_module(f".{file.stem}", package="cmp_algorithms.impl")
    cmp_algorithms.append(module.algorithm)
