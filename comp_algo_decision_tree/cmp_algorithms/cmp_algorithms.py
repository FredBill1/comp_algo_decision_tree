from importlib import import_module
from pathlib import Path

from .CmpAlgorithm import CmpAlgorithm

cmp_algorithms: list[CmpAlgorithm] = []
for file in sorted((Path(__file__).parent / "impl").glob("*.py")):
    module = import_module(f".{file.stem}", package="comp_algo_decision_tree.cmp_algorithms.impl")
    cmp_algorithms.append(module.algorithm)
