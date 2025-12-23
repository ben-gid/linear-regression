import pytest
from new_tree import *

X = np.array([[1, 1, 1],
              [0, 0, 1],
              [0, 1, 0],
              [1, 0, 1],
              [1, 1, 1],
              [1, 1, 0],
              [0, 0, 0],
              [1, 1, 0],
              [0, 1, 0],
              [0, 1, 0]])

y = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

def test_entropy():
    assert entropy(0.5) == 1.0
    assert entropy(0) == 0.
    assert entropy(1) == 0.
    
def test_split_at_feature():
    left_idx, right_idx = split_at_feature(X, 0)
    assert np.array_equal(left_idx, [0, 3, 4, 5, 7]) 
    assert np.array_equal(right_idx, [1, 2, 6, 8, 9])
    
    left_idx, right_idx = split_at_feature(X, 1)
    assert np.array_equal(left_idx, [0, 2, 4, 5, 7, 8, 9]) 
    assert np.array_equal(right_idx, [1, 3, 6])
    
    left_idx, right_idx = split_at_feature(X, 2)
    assert np.array_equal(left_idx, [0, 1, 3, 4]) 
    assert np.array_equal(right_idx, [2, 5, 6, 7, 8, 9])
    
def test_weighted_entropy():
    left_idc, right_idc = split_at_feature(X, 0)
    assert weighted_entropy(X, y, left_idc, right_idc) == 0.7219280948873623
    
    left_idc, right_idc = split_at_feature(X, 1)
    assert weighted_entropy(X, y, left_idc, right_idc) == 0.965148445440323
    
    left_idc, right_idc = split_at_feature(X, 2)
    assert weighted_entropy(X, y, left_idc, right_idc) == 0.8754887502163469
    
def test_information_gain():
    left_indices, right_indices = split_at_feature(X, 0)
    assert information_gain(X, y, left_indices, right_indices) == 0.2780719051126377
    
    left_indices, right_indices = split_at_feature(X, 1)
    assert information_gain(X, y, left_indices, right_indices) == 0.034851554559677034
    
    left_indices, right_indices = split_at_feature(X, 2)
    assert information_gain(X, y, left_indices, right_indices) == 0.12451124978365313
    
def test_best_seperation():
    assert best_seperation(X, y) == 0
    
def test_one_hot_encode():
    assert one_hot_encode(X) == ([], [])
    import numpy as np

def test_one_hot_encode_standard():
    X = np.array([
        [0, 1, 10],
        [1, 0, 20],
        [0, 1, 10],
        [1, 1, 30],
    ])
    
    old_feats, new_feats = one_hot_encode(X)
    
    # Check detected non-binary columns
    assert old_feats == [2]
    
    # Column 2 has 3 unique values → 3 new arrays
    assert len(new_feats) == 3
    
    # Expected one-hot arrays for the non-binary feature
    expected_0 = np.array([
        [1, 0, 0],  # 10
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    expected_1 = np.array([
        [0, 0, 0],
        [0, 1, 0],  # 20
        [0, 0, 0],
        [0, 0, 0],
    ])
    expected_2 = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],  # 30
    ])
    
    # Compare using np.array_equal
    assert np.array_equal(new_feats[0], expected_0)
    assert np.array_equal(new_feats[1], expected_1)
    assert np.array_equal(new_feats[2], expected_2)