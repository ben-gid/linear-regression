from __future__ import annotations
import numpy as np
from typing import Optional

LEFT = 0
RIGHT = 1

def entropy(p: float) -> float:
    # Return 0 for degenerate p
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def information_gain(p_root: float, p_left: float, w_left: float, p_right: float, w_right: float) -> float:
    return entropy(p_root) - (w_left * entropy(p_left) + w_right * entropy(p_right))

class Node:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent: Optional["Node"] = None,
        depth: int = 0
    ) -> None:
        self.X = X
        self.y = y
        self.parent = parent
        self.depth = depth

        # set during fitting
        self.split_feature: Optional[int] = None
        self.child_l: Optional[Node] = None
        self.child_r: Optional[Node] = None
        self.is_leaf: bool = False
        self.prediction: Optional[int] = None   # discrete class
        self.prob: Optional[float] = None       # probability for class 1

    def make_leaf(self):
        self.is_leaf = True
        # If labels are 0/1, probability is mean
        self.prob = float(self.y.mean()) if len(self.y) > 0 else 0.0
        # class label: break ties by rounding (or choose majority)
        self.prediction = int(round(self.prob))

    def converged(self) -> bool:
        return np.unique(self.y).size <= 1

    def predict_single(self, x: np.ndarray) -> int:
        node = self
        while not node.is_leaf:
            if node.split_feature is None:
                # safety fallback
                return int(round(node.y.mean()))
            if x[node.split_feature] == LEFT:
                node = node.child_l
            else:
                node = node.child_r
        return int(node.prediction)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_single(x) for x in X])

class Tree:
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2) -> None:
        self.root: Optional[Node] = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = Node(X, y, depth=0)
        self._split_node(self.root)

    def _best_split_feature(self, X: np.ndarray, y: np.ndarray) -> Optional[int]:
        m, n = X.shape
        p_root = y.mean()
        best_ig = -np.inf
        best_j = None
        for j in range(n):
            feature = X[:, j]
            mask_left = (feature == LEFT)
            mask_right = ~mask_left
            if mask_left.sum() == 0 or mask_right.sum() == 0:
                continue  # skip splits that produce empty child

            y_left = y[mask_left]
            y_right = y[mask_right]

            p_left = float(y_left.mean())
            p_right = float(y_right.mean())
            w_left = len(y_left) / m
            w_right = len(y_right) / m

            ig = information_gain(p_root, p_left, w_left, p_right, w_right)
            if ig > best_ig:
                best_ig = ig
                best_j = j
        return best_j

    def _split_node(self, node: Node):
        # stopping rules
        if node.converged() or node.depth >= self.max_depth or len(node.y) < self.min_samples_split:
            node.make_leaf()
            return

        best_j = self._best_split_feature(node.X, node.y)
        if best_j is None:
            node.make_leaf()
            return

        node.split_feature = best_j
        feature = node.X[:, best_j]
        mask_left = (feature == LEFT)
        mask_right = ~mask_left

        X_left, y_left = node.X[mask_left], node.y[mask_left]
        X_right, y_right = node.X[mask_right], node.y[mask_right]

        node.child_l = Node(X_left, y_left, parent=node, depth=node.depth + 1)
        node.child_r = Node(X_right, y_right, parent=node, depth=node.depth + 1)

        # recursively split children
        self._split_node(node.child_l)
        self._split_node(node.child_r)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree not fitted")
        return self.root.predict(X)

# Pretty-print helper
def print_tree(node: Node, indent: str = ""):
    if node.is_leaf:
        counts = np.bincount(node.y.astype(int), minlength=2) if len(node.y) > 0 else np.array([0, 0])
        print(f"{indent}Leaf(depth={node.depth}) samples={len(node.y)} counts={counts.tolist()} pred={node.prediction} prob={node.prob:.3f}")
    else:
        print(f"{indent}Node(depth={node.depth}) split_feature={node.split_feature} samples={len(node.y)}")
        print(f"{indent} ├─ LEFT (feature[{node.split_feature}] == {LEFT}):")
        print_tree(node.child_l, indent + " │  ")
        print(f"{indent} └─ RIGHT (feature[{node.split_feature}] == {RIGHT}):")
        print_tree(node.child_r, indent + "    ")

if __name__ == "__main__":
    X = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
    ])
    y = np.array([0, 1, 0, 1, 1, 0])

    tree = Tree(max_depth=3, min_samples_split=1)
    tree.fit(X, y)
    print_tree(tree.root)
    preds = tree.predict(X)
    print("preds:", preds)
