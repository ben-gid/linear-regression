import numpy as np
from numpy.typing import NDArray

def scale(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scales each feature by dividing by its maximum value (feature-wise)"""
    denom = np.max(features, axis=0) # devide by denom only if denom != 0
    denom = np.where(denom==0, 1, denom)
    return features / denom

def mean_normalization(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Applies mean normalization feature-wise."""
    denom = (np.max(features, axis=0) - np.min(features, axis=0))
    denom = np.where(denom == 0, 1, denom) 
    return (features - np.mean(features, axis=0)) / denom

def z_score_normalization(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Applies Z-score normalization feature-wise."""

    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0) # np.sqrt(np.mean((features - mu)**2, axis=0))
    sigma = np.where(sigma==0, 1, sigma)
    return (features - mu) / sigma