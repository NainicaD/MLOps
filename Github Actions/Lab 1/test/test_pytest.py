import json
import math
from pathlib import Path

import pytest

from src import similarity as sim

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_data.json"


@pytest.fixture
def data():
    with open(DATA_FILE) as f:
        return json.load(f)


@pytest.fixture
def docs(data):
    return data["documents"]


@pytest.fixture
def samples(data):
    return data["feature_samples"]


# ---------- cosine similarity ----------

@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([1, 0], [1, 0], 1.0),      # same direction
        ([1, 0], [0, 1], 0.0),      # orthogonal
        ([1, 2], [-1, -2], -1.0),   # opposite
        ([1, 1], [1, 0], 1 / math.sqrt(2)),
    ],
)
def test_cosine_known_values(a, b, expected):
    assert sim.cosine_similarity(a, b) == pytest.approx(expected)


def test_cosine_hand_computed_on_documents(docs):
    # dot = 4, both norms = sqrt(30), so cosine = 4 / 30.
    assert sim.cosine_similarity(docs["ml_intro"], docs["baking"]) == pytest.approx(4 / 30)


def test_cosine_ranks_related_documents_higher(docs):
    related = sim.cosine_similarity(docs["ml_intro"], docs["ml_pipeline"])
    unrelated = sim.cosine_similarity(docs["ml_intro"], docs["baking"])
    assert related > unrelated


@pytest.mark.parametrize("scale", [0.5, 2, 100])
def test_cosine_ignores_vector_length(docs, scale):
    a = docs["ml_intro"]
    scaled = [x * scale for x in a]
    assert sim.cosine_similarity(a, scaled) == pytest.approx(1.0)


def test_cosine_rejects_zero_vector():
    with pytest.raises(ValueError):
        sim.cosine_similarity([0, 0], [1, 2])


# ---------- Euclidean and Manhattan distance ----------

@pytest.mark.parametrize(
    "a, b, euclid, manhattan",
    [
        ([0, 0], [3, 4], 5.0, 7.0),
        ([1, 2, 3], [1, 2, 3], 0.0, 0.0),
        ([-1, -1], [1, 1], math.sqrt(8), 4.0),
    ],
)
def test_distances_known_values(a, b, euclid, manhattan):
    assert sim.euclidean_distance(a, b) == pytest.approx(euclid)
    assert sim.manhattan_distance(a, b) == pytest.approx(manhattan)


def test_euclidean_triangle_inequality(docs):
    a, b, c = docs["ml_intro"], docs["ml_pipeline"], docs["baking"]
    assert sim.euclidean_distance(a, c) <= sim.euclidean_distance(a, b) + sim.euclidean_distance(b, c)


def test_manhattan_never_smaller_than_euclidean(docs):
    a, b = docs["ml_intro"], docs["baking"]
    assert sim.manhattan_distance(a, b) >= sim.euclidean_distance(a, b)


# ---------- Pearson correlation ----------

@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([1, 2, 3], [2, 4, 6], 1.0),
        ([1, 2, 3], [6, 4, 2], -1.0),
        ([1, 2, 3], [10, 20, 30], 1.0),
    ],
)
def test_pearson_known_values(a, b, expected):
    assert sim.pearson_correlation(a, b) == pytest.approx(expected)


def test_pearson_ignores_shift_and_scale():
    a = [1, 4, 2, 8, 5]
    b = [3, 1, 4, 1, 5]
    transformed = [10 * x + 7 for x in b]
    assert sim.pearson_correlation(a, b) == pytest.approx(sim.pearson_correlation(a, transformed))


def test_pearson_rejects_constant_vector():
    with pytest.raises(ValueError):
        sim.pearson_correlation([1, 2, 3], [5, 5, 5])


# ---------- Jaccard similarity ----------

@pytest.mark.parametrize(
    "a, b, expected",
    [
        ({"a", "b"}, {"a", "b"}, 1.0),
        ({"a", "b"}, {"c", "d"}, 0.0),
        ({"a", "b", "c"}, {"b", "c", "d"}, 0.5),
        (["a", "a", "b"], ["b"], 0.5),  # duplicates count once
    ],
)
def test_jaccard_known_values(a, b, expected):
    assert sim.jaccard_similarity(a, b) == pytest.approx(expected)


def test_jaccard_on_vocabulary_overlap(data):
    vocab = data["vocabulary"]
    used = lambda doc: {w for w, count in zip(vocab, data["documents"][doc]) if count > 0}
    # ml_intro uses 4 words, baking uses 4, they share only "data": 1 / 7.
    assert sim.jaccard_similarity(used("ml_intro"), used("baking")) == pytest.approx(1 / 7)


def test_jaccard_rejects_two_empty_sets():
    with pytest.raises(ValueError):
        sim.jaccard_similarity(set(), set())


def test_jaccard_rejects_strings():
    with pytest.raises(TypeError):
        sim.jaccard_similarity("abc", "abd")


# ---------- Wasserstein distance ----------

@pytest.mark.parametrize(
    "u, v, u_w, v_w, expected",
    [
        ([0, 1, 3], [5, 6, 8], None, None, 5.0),
        ([0, 1], [0, 1], [3, 1], [2, 2], 0.25),
        ([3.4, 3.9, 7.5, 7.8], [4.5, 1.4], [1.4, 0.9, 3.1, 7.2], [3.2, 3.5], 4.0781331438047861),
    ],
)
def test_wasserstein_reference_values(u, v, u_w, v_w, expected):
    # Reference values from the SciPy documentation for scipy.stats.wasserstein_distance.
    assert sim.wasserstein_distance(u, v, u_w, v_w) == pytest.approx(expected)


@pytest.mark.parametrize("shift", [-3, 0.5, 10])
def test_wasserstein_of_shifted_sample_equals_shift(samples, shift):
    base = samples["training"]
    moved = [x + shift for x in base]
    assert sim.wasserstein_distance(base, moved) == pytest.approx(abs(shift))


def test_wasserstein_detects_drift(samples):
    ok = sim.wasserstein_distance(samples["training"], samples["live_ok"])
    drift = sim.wasserstein_distance(samples["training"], samples["live_drift"])
    assert ok == pytest.approx(0.125)
    assert drift == pytest.approx(5.0)
    assert drift > ok


def test_wasserstein_ignores_sample_order(samples):
    a = samples["training"]
    assert sim.wasserstein_distance(a, samples["live_ok"]) == pytest.approx(
        sim.wasserstein_distance(list(reversed(a)), samples["live_ok"])
    )


@pytest.mark.parametrize("bad_weights", [[-1, 2], [0, 0], [1]])
def test_wasserstein_rejects_bad_weights(bad_weights):
    with pytest.raises(ValueError):
        sim.wasserstein_distance([1, 2], [1, 2], u_weights=bad_weights)


# ---------- properties shared by every measure ----------

SYMMETRIC = [
    sim.cosine_similarity,
    sim.euclidean_distance,
    sim.manhattan_distance,
    sim.pearson_correlation,
    sim.wasserstein_distance,
]


@pytest.mark.parametrize("measure", SYMMETRIC, ids=lambda f: f.__name__)
def test_symmetry(docs, measure):
    a, b = docs["ml_intro"], docs["baking"]
    assert measure(a, b) == pytest.approx(measure(b, a))


@pytest.mark.parametrize(
    "measure, expected",
    [
        (sim.euclidean_distance, 0.0),
        (sim.manhattan_distance, 0.0),
        (sim.wasserstein_distance, 0.0),
        (sim.cosine_similarity, 1.0),
        (sim.pearson_correlation, 1.0),
    ],
    ids=lambda x: getattr(x, "__name__", str(x)),
)
def test_identical_inputs(docs, measure, expected):
    a = docs["ml_pipeline"]
    assert measure(a, a) == pytest.approx(expected)


# ---------- input validation ----------

@pytest.mark.parametrize(
    "a, b, error",
    [
        ([1, 2], [1, 2, 3], ValueError),      # length mismatch
        ([], [], ValueError),                 # empty
        ([1, "2"], [1, 2], TypeError),        # non-numeric
        ([True, 2], [1, 2], TypeError),       # bools are rejected
        ([1, float("nan")], [1, 2], ValueError),
        ("12", [1, 2], TypeError),            # not a list or tuple
    ],
)
def test_vector_input_validation(a, b, error):
    with pytest.raises(error):
        sim.euclidean_distance(a, b)
