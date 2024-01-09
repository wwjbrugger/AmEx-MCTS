import numpy as np


def tie_breaking_argmax(a: np.ndarray, eps: float = 1e-8) -> int:
    return np.argmax(a + np.random.random(a.shape) * eps)
