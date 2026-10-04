import json
import math
import unittest
from pathlib import Path

from src import similarity as sim

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_data.json"


class TestSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(DATA_FILE) as f:
            data = json.load(f)
        cls.docs = data["documents"]
        cls.samples = data["feature_samples"]

    def test_cosine_similarity(self):
        self.assertAlmostEqual(sim.cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(sim.cosine_similarity([1, 2], [-1, -2]), -1.0)
        self.assertAlmostEqual(
            sim.cosine_similarity(self.docs["ml_intro"], self.docs["baking"]), 4 / 30
        )

    def test_cosine_prefers_related_documents(self):
        related = sim.cosine_similarity(self.docs["ml_intro"], self.docs["ml_pipeline"])
        unrelated = sim.cosine_similarity(self.docs["ml_intro"], self.docs["baking"])
        self.assertGreater(related, unrelated)

    def test_distances(self):
        self.assertAlmostEqual(sim.euclidean_distance([0, 0], [3, 4]), 5.0)
        self.assertAlmostEqual(sim.manhattan_distance([0, 0], [3, 4]), 7.0)

    def test_pearson_correlation(self):
        self.assertAlmostEqual(sim.pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(sim.pearson_correlation([1, 2, 3], [6, 4, 2]), -1.0)

    def test_jaccard_similarity(self):
        self.assertAlmostEqual(sim.jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}), 0.5)
        self.assertEqual(sim.jaccard_similarity({"a"}, {"b"}), 0.0)

    def test_wasserstein_reference_values(self):
        self.assertAlmostEqual(sim.wasserstein_distance([0, 1, 3], [5, 6, 8]), 5.0)
        self.assertAlmostEqual(sim.wasserstein_distance([0, 1], [0, 1], [3, 1], [2, 2]), 0.25)

    def test_wasserstein_detects_drift(self):
        ok = sim.wasserstein_distance(self.samples["training"], self.samples["live_ok"])
        drift = sim.wasserstein_distance(self.samples["training"], self.samples["live_drift"])
        self.assertAlmostEqual(ok, 0.125)
        self.assertAlmostEqual(drift, 5.0)

    def test_symmetry(self):
        a, b = self.docs["ml_intro"], self.docs["baking"]
        for measure in (sim.cosine_similarity, sim.euclidean_distance,
                        sim.manhattan_distance, sim.pearson_correlation,
                        sim.wasserstein_distance):
            with self.subTest(measure=measure.__name__):
                self.assertAlmostEqual(measure(a, b), measure(b, a))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            sim.euclidean_distance([1, 2], [1, 2, 3])
        with self.assertRaises(TypeError):
            sim.euclidean_distance([1, "2"], [1, 2])
        with self.assertRaises(ValueError):
            sim.cosine_similarity([0, 0], [1, 1])
        with self.assertRaises(ValueError):
            sim.pearson_correlation([1, 2, 3], [5, 5, 5])
        with self.assertRaises(ValueError):
            sim.jaccard_similarity(set(), set())
        with self.assertRaises(ValueError):
            sim.wasserstein_distance([1, 2], [1, 2], u_weights=[-1, 2])
        with self.assertRaises(ValueError):
            sim.euclidean_distance([1, math.inf], [1, 2])


if __name__ == "__main__":
    unittest.main()
