from __future__ import annotations
import numpy as np

LEFT = 0 # left node value
RIGHT = 1 # right node value


class Node:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray, 
        parent: Node | None = None, 
        child_l: Node | None = None,
        child_r: Node | None = None,
    ) -> None:
        """creates a Node for a decision tree

        Args:
            x (np.ndarray): 2d array of data that will be in the node
            y (np.ndarray): 1d array of values (as 0 or 1) for each row of x
            parent (Node | None, optional): _description_. Defaults to None.
            child_l (Node | None, optional): _description_. Defaults to None.
            child_r (Node | None, optional): _description_. Defaults to None.
        """
        self.X = X
        self.y = y
        self.parent: Node | None = parent
        self.child_l: Node | None  = child_l
        self.child_r: Node | None = child_r
        
    def __str__(self) -> str:
        return f"""{self.child_l=}, {self.child_r=}"""
    
    def converged(self) -> bool:
        mean = self.y.mean()
        if mean == 1 or mean == 0:
            return True
        else:
            return False
        
    def set_children(self, child_l: Node, child_r: Node):
        self.child_l = child_l
        self.child_r = child_r
    
    def predict(self) -> int:
        return self.y.mean()

class Tree:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """creates a decision tree with a root node

        Args:
            X (np.ndarray): 2d array of all data
            y (np.ndarray): 1d array of all values
        """
        self.root_node = Node(X, y)

    def __str__(self) -> str:
        return f"""
        {self.root_node}
    """
        
    def split_node(self, feature_idx: np.intp, parent_node: Node) -> tuple[Node, Node]:
        """seperate by feature_idx.
        all 0 values go to the left. all 1 values go to the right

        Args:
            feature_idx (np.intp): _description_
            parent_node (Node): _description_

        Returns:
            tuple[Node, Node]: _description_
        """
        X = parent_node.X
        feature = X[:, feature_idx]
        y = parent_node.y
        
        # get inexes to split by requested feature
        left_f_idx = np.where(feature == LEFT)
        right_f_idx = np.where(feature == RIGHT)
        
        # get X and y values for each child node
        # print(f"{X=}, {left_f_idx=}, {right_f_idx=}")
        lx = X[left_f_idx]
        ly = y[left_f_idx]
        rx = X[right_f_idx]
        ry = y[right_f_idx]
        
        l_node = Node(lx, ly, parent=parent_node)
        r_node = Node(rx, ry, parent=parent_node)
        parent_node.set_children(l_node, r_node)
            
        return l_node, r_node
    
    
    

def main():
    X = np.array([[0, 1, 0],
                  [1, 1, 0],
                  [1, 0, 1],
                  [0, 0, 1],
                  [1, 1, 1],
                  [0, 0, 0],])
    
    y = np.array([0, 1, 0, 1, 1, 0])
    
    tree = create_tree(X, y)
    print(tree)

def p1(x: np.ndarray):
    return x.mean()

def entropy(p1: float) -> float:
    """calculates entropy/impurity of classified training data for
    decision tree. 

    Args:
        p1 (float | np.ndarray): mean of training data set examples as class 1.
        (# samples with class 1 in node) / (total samples in node)

    Returns:
        float | np.ndarray: entropy or impurity of p1
    """
    # (# samples with class 0 in node) / (total samples in node)
    p0 = 1 - p1 
    return -p1 * np.log2(p1) - p0 * np.log2(p0)

def information_gain(
    p1_root: float, 
    p1_left: float, 
    w_left: float, 
    p1_right: float, 
    w_right: float
) -> float:
    return entropy(p1_root) - (w_left * entropy(p1_left)) + (w_right * entropy(p1_right))

def best_seperation(X: np.ndarray) -> np.intp:
    """returns the feature with the highest information gain of X (X[:,j]) to seperate on.

    Args:
        X (np.ndarray): data as a 2d array

    Returns:
        np.intp: column index of X
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2d; {X=}, {X.ndim}")
    m_examples, n_features = X.shape
    igs = np.zeros(n_features)
    for j in range(n_features):
        feat_j = X[:, j]
        p1_root = feat_j.mean()
        p1_left = p1(feat_j[np.where(feat_j == LEFT)])
        w_left = np.sum(p1_left) / m_examples
        p1_right = p1(feat_j[np.where(feat_j == RIGHT)])
        w_right = np.sum(p1_right) / m_examples
        igs[j] = information_gain(p1_root, p1_left, w_left, p1_right, w_right)
    return igs.argmax()

def create_children(tree: Tree, parent_node: Node, depth_left: int):
    sep_index = best_seperation(parent_node.X)
    children = tree.split_node(sep_index, parent_node)
    for node in children:
        if node.converged() or depth_left < 0:
            continue
        depth_left -= 1
        create_children(tree, node, depth_left)

def create_tree(X: np.ndarray, y: np.ndarray, max_depth: int=3):
    tree = Tree(X, y) 
    parent_node = tree.root_node  
    
    if parent_node.converged() is not True:
        create_children(tree, parent_node, max_depth - 1)
    return tree
        
        
if __name__ == "__main__":
    main()