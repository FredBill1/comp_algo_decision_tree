from typing import Generic, Optional, TypeVar
from weakref import proxy

Container = TypeVar("Container")


class DecisionTreeNode(Generic[Container]):
    def __init__(self, parent: Optional["DecisionTreeNode"] = None, is_left: bool = False) -> None:
        self.id = 0 if parent is None else parent.id * 2 + (2 - int(is_left))
        self.idx_array: Optional[Container] = None
        self.cmp_xy: Optional[tuple[int, int]] = None
        self.val_arrays: list[Container] = []
        self.left: Optional[DecisionTreeNode] = None
        self.right: Optional[DecisionTreeNode] = None
        self.parent = None if parent is None else proxy(parent)

    @property
    def is_left(self) -> bool:
        return self.parent.left is self

    def edge_data(self, use_letter: bool) -> None:
        x, y = [chr(ord("a") + x) if use_letter else f"[{x}]" for x in self.parent.cmp_xy[:2]]
        return {"data": dict(source=str(self.parent.id), target=str(self.id), cmp_op=f"{x}<{y}" if self.is_left else f"{x}>{y}")}

    __slots__ = ["id", "idx_array", "cmp_xy", "val_arrays", "left", "right", "parent", "__weakref__"]
