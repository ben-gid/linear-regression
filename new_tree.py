import numpy as np


LEFT = 1 # left node value
RIGHT = 0 # right node value

def main():
    X = np.array([[1, 1, 1],
              [2, 0, 1],
              [3, 1, 0],
              [1, 0, 1],
              [1, 1, 1],
              [1, 1, 0],
              [0, 0, 0],
              [1, 1, 0],
              [0, 1, 0],
              [0, 1, 0]])

    y = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    print(one_hot_encode(X))
    
def entropy(p1: float) -> float:
    """calculates entropy/impurity of classified training data for
    decision tree. 

    Args:
        p1 (float | np.ndarray): mean of training data set examples as class 1.
        (# samples with class 1 in node) / (total samples in node)

    Returns:
        float | np.ndarray: entropy or impurity of p1
    """
    if (p1 == 0) or (p1 == 1):
        return 0.
    # (# samples with class 0 in node) / (total samples in node)
    p0 = 1 - p1 
    return -p1 * np.log2(p1) - p0 * np.log2(p0)

def split_at_feature(X: np.ndarray, feature_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Splits X by chosen feature.

    Args:
        X (np.ndarray): training data as 2d array
        feature_idx (int): feature (column) index of X to split on

    Returns:
        tuple[np.ndarray, np.ndarray]: left_indeces, right_indeces
    """
    left_idcs = np.where(X[:, feature_idx] == LEFT)[0]
    right_idcs = np.where(X[:, feature_idx] == RIGHT)[0]
    return left_idcs, right_idcs

def weighted_entropy(X: np.ndarray, y: np.ndarray, left_idcs: np.ndarray, right_idcs: np.ndarray
                     ) -> float:
    """calculates entropy based on examples in new branch.

    Args:
        X (np.ndarray): data as 2d array
        y (np.ndarray): binary values as 1d array
        left_idcs (np.ndarray): indeces of examples in X to go to the left branch/node
        right_idcs (np.ndarray): indeces of examples in X to go to the right branch/node

    Returns:
        (float): weighted entropy as float
    """
    w_left = left_idcs.shape[0] / X.shape[0]
    w_right = right_idcs.shape[0] / X.shape[0]
    p_left = np.sum(y[left_idcs]) / left_idcs.shape[0]
    p_right = np.sum(y[right_idcs]) / right_idcs.shape[0]
    
    return w_left * entropy(p_left) + w_right * entropy(p_right)
    
def information_gain(X:np.ndarray, y:np.ndarray, left_idcs: np.ndarray, right_idcs: np.ndarray):
    """calculates information gained if X is split using left and right indeces.

    Args:
        X (np.ndarray): data as 2d array
        y (np.ndarray): binary values as 1d array
        left_idcs (np.ndarray): indeces of examples in X to go to the left branch/node
        right_idcs (np.ndarray): indeces of examples in X to go to the right branch/node

    Returns:
        (float): information gain as float
    """
    p = np.sum(y) / y.shape[0]
    return entropy(p) - weighted_entropy(X, y, left_idcs, right_idcs)

def best_seperation(X: np.ndarray, y: np.ndarray) -> int:
    """returns the feature of X (X[:,j]) with the highest information gain to seperate on.

    Args:
        X (np.ndarray): data as 2d array
        y (np.ndarray): binary values as 1d array

    Raises:
        ValueError: if X isnt a 2d array
        RuntimeError: if best_idx is never asigned a truethful value

    Returns:
        int: index of highest information gain
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2d; {X=}, {X.ndim}")
    
    n_features = X.shape[1]
    best_ig = -np.inf
    best_idx = None
    for idx in range(n_features):
        left_idcs, right_idcs = split_at_feature(X, idx)
        ig = information_gain(X, y, left_idcs, right_idcs)
        if ig > best_ig:
            best_ig = ig
            best_idx = idx
    if best_idx is None:
        raise RuntimeError(f"best_idx is {None}")
    return best_idx 

def one_hot_encode(X: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    """distributes non binary features to new on hot encoded binary features.

    Args:
        X (np.ndarray): data as 2d array

    Returns:
        tuple[list[int], list[np.ndarray]]: list of non_binary features 
        and list of new binary one hot encoded features 
    """
    m, n = X.shape
    nb_feats_and_args: dict[int, np.ndarray] = {}
    
    # get all non binary features
    for i in range(n):
        c = np.all(np.isin(X[:, i], [0, 1]))
        if not c:
            # add "feature_number: unique_features" to nb_feats_and_args
            nb_feats_and_args[i] = np.unique(X[:,i])
            
    new_binary_feats: list[np.ndarray] = []
    for feature, args in nb_feats_and_args.items():
        n_feats = args.shape[0] # loop for number of new one hot features
        for i in range(n_feats): 
            new_feats = np.zeros((m, n_feats)) # place holder for new features
            # get all places in X[:, feature] that is args[i]
            ones_places = np.where(X[:, feature] == args[i])[0]
            # set to 1
            new_feats[ones_places, i] = 1
            new_binary_feats.append(new_feats)
    
    old_non_binary_feats = list(nb_feats_and_args.keys())    
    return old_non_binary_feats, new_binary_feats
            

def generate_bagged_sample(X: np.ndarray, y: np.ndarray, B: int
                              ) -> tuple[np.ndarray, np.ndarray]:
    """generates a new X and y to train a decision tree on using sampling with replacement.

    Args:
        X (np.ndarray): data as 2d array
        y (np.ndarray): binary values as 1d array
        B (int): length of desired resulting X and y

    Returns:
        tuple[np.ndarray, np.ndarray]: sampled_X and sampled_y
    """
    len_X = X.shape[0]
    sampled_X = []
    sampled_y = []
    for _ in range(B):
        x_idx = np.random.randint(0, len_X)
        sampled_X.append(X[x_idx])
        sampled_y.append(y[x_idx])
    return np.array(sampled_X), np.array(sampled_y)

if __name__ == "__main__":
    main()