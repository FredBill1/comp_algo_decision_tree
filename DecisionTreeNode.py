from typing import Optional

from Config import *


class DecisionTreeNode:
    def __init__(self, node_id: int, parent: Optional["DecisionTreeNode"] = None, use_letter: bool = True, is_left: bool = False) -> None:
        self.id = node_id
        self.arr: Optional[list[int]] = None
        self.cmp_xy: Optional[tuple[int, int]] = None
        self.actuals: list[tuple[int, ...]] = []
        self.left: Optional[DecisionTreeNode] = None
        self.right: Optional[DecisionTreeNode] = None
        self.pos = 0.0 if parent is None else parent.pos * 2 + (2 - int(is_left))
        self.edge_data = None
        if parent is not None:
            x, y = [chr(ord("a") + x) if use_letter else f"[{x}]" for x in parent.cmp_xy[:2]]
            self.edge_data = {"data": dict(source=parent.id, target=self.id, cmp_op=f"{x}<{y}" if is_left else f"{x}>{y}")}

    def get_arr(self) -> str:
        return "(" + ",".join(chr(ord("a") + x) if len(self.arr) <= 26 else str(x) for x in self.arr) + ")"

    def label(self, crop_length: Optional[int]) -> str:
        ret = [self.get_arr()]
        tot_len = len(ret[0])
        for actual in self.actuals:
            ret.append(f" ({','.join(str(x + 1) for x in actual)})")
            tot_len += len(ret[-1])
            if crop_length is not None and tot_len >= crop_length:
                return "".join(ret)[: crop_length - 3] + "..."
        return "".join(ret)

    def node_data(self, show_full_labels: bool, classes: str) -> dict:
        label = self.label(LABEL_MAX_LENGTH if show_full_labels else LABEL_CROP_LENGTH)
        return {"data": {"id": self.id, "label": label, "pos": self.pos}, "classes": classes}

    __slots__ = ["id", "arr", "cmp_xy", "actuals", "left", "right", "pos", "edge_data"]
