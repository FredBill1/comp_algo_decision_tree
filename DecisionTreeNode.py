from typing import Generic, Optional, TypeVar

Container = TypeVar("Container")


class DecisionTreeNode(Generic[Container]):
    def __init__(self, node_id: int, parent: Optional["DecisionTreeNode"] = None, use_letter: bool = True, is_left: bool = False) -> None:
        self.id = node_id
        self.idx_array: Optional[Container] = None
        self.cmp_xy: Optional[tuple[int, int]] = None
        self.val_arrays: list[Container] = []
        self.left: Optional[DecisionTreeNode] = None
        self.right: Optional[DecisionTreeNode] = None
        self.pos = 0.0 if parent is None else parent.pos * 2 + (2 - int(is_left))
        self.edge_data = None
        if parent is not None:
            x, y = [chr(ord("a") + x) if use_letter else f"[{x}]" for x in parent.cmp_xy[:2]]
            self.edge_data = {"data": dict(source=parent.id, target=self.id, cmp_op=f"{x}<{y}" if is_left else f"{x}>{y}")}

    __slots__ = ["id", "idx_array", "cmp_xy", "val_arrays", "left", "right", "pos", "edge_data"]
