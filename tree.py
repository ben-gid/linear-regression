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
    
    
    