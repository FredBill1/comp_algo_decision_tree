from collections.abc import Callable
from time import thread_time
from typing import Optional

from cmp_algorithms.CmpAlgorithm import CmpAlgorithm, IdxVal
from Config import *
from DecisionTreeNode import DecisionTreeNode


class NonDeterministicError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__("Non-deterministic cmp algorithm: " + msg)


class InvalidCmpAlgorithmError(Exception):
    def __init__(self) -> None:
        super().__init__("Invalid cmp algorithm")


# copy from functools.cmp_to_key
# since member "obj" is not present in the documentation, it is not guaranteed to be exist in the future
# fmt: off
def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function"""
    class K(object):
        __slots__ = ['obj']
        def __init__(self, obj):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        __hash__ = None
    return K
# fmt: on


def decision_tree(cmp_algorithm: CmpAlgorithm, N: int, callback: Optional[Callable[[int, int], None]] = None) -> tuple[list[DecisionTreeNode], list[int], int]:
    nodes = [DecisionTreeNode()]

    def convert_idx_array(idx_array: list[int]) -> list[int]:
        return cmp_algorithm.idx_converter(cmp_algorithm.map(lambda x: x.obj.idx, idx_array))

    def cmp(x: IdxVal, y: IdxVal) -> int:
        if x.idx == y.idx:
            return 0
        if x.idx > y.idx:
            return -cmp(y, x)
        cur_cmp_xy = (x.idx, y.idx, x.val < y.val)
        if not (idx_arrays and cur_cmp_xy == cmp_xys[-1]):
            nonlocal operation_cnt
            operation_cnt += 1
            idx_arrays.append(convert_idx_array(idx_array))
            cmp_xys.append(cur_cmp_xy)
        return 1 if x.val > y.val else -1 if x.val < y.val else 0

    key = cmp_to_key(cmp)

    do_sample = N > cmp_algorithm.max_N
    TOTAL = 1 if do_sample else cmp_algorithm.input_total(N)
    if callback is not None:
        if do_sample:
            callback(0, MAX_SAMPLE_TIME_MS)
        else:
            callback(0, TOTAL)
    operation_cnts = []
    start_time = thread_time()
    leaf_cnt = 0
    for I, val_array in enumerate(cmp_algorithm.sampler(N) if do_sample else cmp_algorithm.generator(N)):
        idx_arrays = []
        cmp_xys = []
        idx_array = cmp_algorithm.map_enumerate(key, val_array)
        operation_cnt = 0
        cmp_algorithm.func(idx_array)
        if not cmp_algorithm.validator(cmp_algorithm.map(lambda x: x.obj.val, idx_array)):
            raise InvalidCmpAlgorithmError
        operation_cnts.append(operation_cnt)
        idx_arrays.append(convert_idx_array(idx_array))
        cmp_xys.append(None)
        node = nodes[0]
        is_new_node = I == 0
        for J, (idx_array, cmp_xy) in enumerate(zip(idx_arrays, cmp_xys)):
            if node.idx_array is None:
                node.idx_array = idx_array
            elif node.idx_array != idx_array:
                raise NonDeterministicError("the new index array is not the same as the previous one at this node")
            if len(node.val_arrays) < ACTUALS_MAX_LENGTH:
                node.val_arrays.append(val_array)

            if J == len(idx_arrays) - 1:
                if node.cmp_xy is not None:
                    raise NonDeterministicError("should have stopped at a leaf node")
                if is_new_node:
                    leaf_cnt += 1
                break

            if node.cmp_xy is None:
                node.cmp_xy = cmp_xy[:2]
            elif node.cmp_xy != cmp_xy[:2]:
                raise NonDeterministicError(f"the new comparison({node.cmp_xy}) is not the same as the previous one({cmp_xy[:2]}) at this node")

            is_new_node = False
            if cmp_xy[2]:  # x < y
                if node.left is None:
                    node.left = DecisionTreeNode(node, True)
                    nodes.append(node.left)
                    is_new_node = True
                node = node.left
            else:
                if node.right is None:
                    node.right = DecisionTreeNode(node, False)
                    nodes.append(node.right)
                    is_new_node = True
                node = node.right
        if do_sample and (cur_time := int((thread_time() - start_time) * 1000)) >= MAX_SAMPLE_TIME_MS:
            callback(MAX_SAMPLE_TIME_MS, MAX_SAMPLE_TIME_MS)
            break
        if callback is not None:
            if do_sample:
                callback(cur_time, MAX_SAMPLE_TIME_MS)
            else:
                callback(I + 1, TOTAL)
    nodes.sort(key=lambda x: x.id)
    for i, node in enumerate(nodes):
        node.id = i
    return nodes, operation_cnts, leaf_cnt


if __name__ == "__main__":
    from cmp_algorithms.cmp_algorithms import cmp_algorithms

    N = 3
    nodes, operation_cnts, leaf_cnt = decision_tree(cmp_algorithms[0], N)
