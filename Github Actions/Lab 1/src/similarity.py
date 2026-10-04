"""Similarity and distance measures for vectors, sets and distributions.

These are the measures used across ML systems: comparing embeddings
(cosine), clustering and nearest-neighbour search (Euclidean, Manhattan),
comparing tag or token sets (Jaccard), correlating features (Pearson), and
monitoring data drift between a training sample and live data (Wasserstein).
"""

import math


# ---------- input validation ----------

def _validate_vector(v, name="vector"):
    if not isinstance(v, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of numbers.")
    if len(v) == 0:
        raise ValueError(f"{name} must not be empty.")
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError(f"{name} must contain only numbers.")
        if not math.isfinite(x):
            raise ValueError(f"{name} must contain only finite numbers.")


def _validate_pair(a, b):
    _validate_vector(a, "a")
    _validate_vector(b, "b")
    if len(a) != len(b):
        raise ValueError("a and b must have the same length.")


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


# ---------- vector measures ----------

def cosine_similarity(a, b):
    """Cosine of the angle between a and b, in [-1, 1].

    1 means same direction, 0 means orthogonal, -1 means opposite.
    Ignores vector length, which is why it is the default for embeddings.
    """
    _validate_pair(a, b)
    norm_a, norm_b = _norm(a), _norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine similarity is undefined for a zero vector.")
    dot = sum(x * y for x, y in zip(a, b))
    # Clamp to [-1, 1] to absorb floating-point error.
    return max(-1.0, min(1.0, dot / (norm_a + norm_b)))


def euclidean_distance(a, b):
    """Straight-line (L2) distance between a and b."""
    _validate_pair(a, b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a, b):
    """Sum of absolute coordinate differences (L1 distance)."""
    _validate_pair(a, b)
    return sum(abs(x - y) for x, y in zip(a, b))


def pearson_correlation(a, b):
    """Linear correlation between a and b, in [-1, 1]."""
    _validate_pair(a, b)
    if len(a) < 2:
        raise ValueError("Pearson correlation needs at least two values.")
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    da = [x - mean_a for x in a]
    db = [y - mean_b for y in b]
    denom = _norm(da) * _norm(db)
    if denom == 0:
        raise ValueError("Pearson correlation is undefined for a constant vector.")
    r = sum(x * y for x, y in zip(da, db)) / denom
    return max(-1.0, min(1.0, r))


# ---------- set measure ----------

def jaccard_similarity(a, b):
    """Size of the intersection divided by size of the union, in [0, 1]."""
    if isinstance(a, str) or isinstance(b, str):
        raise TypeError("pass collections of items, not strings.")
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        raise ValueError("Jaccard similarity is undefined for two empty sets.")
    return len(set_a & set_b) / len(union)


# ---------- distribution measure ----------

def _validate_distribution(values, weights, name):
    _validate_vector(values, name)
    if weights is None:
        return list(values), [1.0] * len(values)
    _validate_vector(weights, f"{name}_weights")
    if len(weights) != len(values):
        raise ValueError(f"{name}_weights must have the same length as {name}.")
    if any(w < 0 for w in weights):
        raise ValueError(f"{name}_weights must be non-negative.")
    if sum(weights) == 0:
        raise ValueError(f"{name}_weights must not all be zero.")
    return list(values), list(weights)


def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    """1-D Wasserstein (earth mover's) distance between two samples.

    Think of each sample as a pile of earth along a number line: this is the
    minimum total "earth x distance" needed to reshape one pile into the
    other. Computed as the area between the two cumulative distribution
    functions. Weights are optional and are normalized to sum to 1.
    """
    u_values, u_weights = _validate_distribution(u_values, u_weights, "u_values")
    v_values, v_weights = _validate_distribution(v_values, v_weights, "v_values")
    u_total, v_total = sum(u_weights), sum(v_weights)

    points = sorted(set(u_values) | set(v_values))
    distance = 0.0
    for left, right in zip(points, points[1:]):
        cdf_u = sum(w for x, w in zip(u_values, u_weights) if x <= left) / u_total
        cdf_v = sum(w for x, w in zip(v_values, v_weights) if x <= left) / v_total
        distance += abs(cdf_u - cdf_v) * (right - left)
    return distance
