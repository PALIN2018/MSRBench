import json
import math
import tempfile
import unittest
from pathlib import Path

from LVLM4Rec.evaluation import (
    conservative_alias,
    evaluate_file,
    map_to_candidate,
    normalize_title,
)


def example(target, candidates, recommendations):
    prompt = f"There are {len(candidates)} candidate items in the item pool: {candidates}.\nRank them."
    return {
        "prompt": prompt,
        "target": {"titles": [target]},
        "api_response": json.dumps({"recommendations": recommendations}),
    }


class TitleMappingTests(unittest.TestCase):
    def test_format_normalization(self):
        self.assertEqual(normalize_title("Men’s  Shirt"), normalize_title("Men's Shirt"))

    def test_exact_other_candidate_stays_other_candidate(self):
        candidates = [
            "Fisher-Price Nickelodeon Bubble Guppies: Fin-Tastic Guitar",
            "Fisher-Price Little People Wheelies Zig The Big Rig",
        ]
        self.assertEqual(map_to_candidate(candidates[1], candidates), (1, "exact"))

    def test_punctuation_variant_is_accepted(self):
        candidates = ["Discovery Kids Indoor/Outdoor Play Tent.", "Other Item"]
        self.assertEqual(
            map_to_candidate("Discovery Kids Indoor/Outdoor Play Tent", candidates),
            (0, "conservative"),
        )

    def test_long_title_truncation_is_accepted(self):
        short = "Watts Beauty 2% Retinol Vitamin A Hyaluronic Acid Moisturizer Cream"
        long = short + " Proudly Made in the USA 97% Natural 2oz"
        self.assertTrue(conservative_alias(short, long))

    def test_different_edition_is_rejected(self):
        self.assertFalse(
            conservative_alias(
                "Cards Against Humanity: First Expansion",
                "Cards Against Humanity: Second Expansion",
            )
        )

    def test_different_model_number_is_rejected(self):
        self.assertFalse(
            conservative_alias(
                "Syma S111G 3.5 Channel RC Helicopter with Gyro",
                "Syma S102G 3.5 Channel RC Helicopter with Gyro",
            )
        )

    def test_short_generic_prefix_is_rejected(self):
        self.assertFalse(conservative_alias("Munchkin", "Munchkin Clerical Errors"))


class EvaluationTests(unittest.TestCase):
    def test_fixed_denominator_and_rank_slots(self):
        candidates = ["Target Item", "Other Item", "Third Item"]
        data = {
            "u1": example("Target Item", candidates, ["Unknown Product", "Target Item"]),
            "u2": example("Target Item", candidates, ["Other Item", "Third Item"]),
            "u3": example(
                "Discovery Kids Indoor/Outdoor Play Tent.",
                ["Discovery Kids Indoor/Outdoor Play Tent.", "Other Item"],
                ["Discovery Kids Indoor/Outdoor Play Tent"],
            ),
            "u4": {
                "prompt": "unused",
                "target": {"titles": ["Target Item"]},
                "api_response": "not-json",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed_data.json"
            path.write_text(json.dumps(data))
            result = evaluate_file(path, cohort_size=4, topks=(1, 3))

        self.assertEqual(result["hits_at_1_pct"], 25.0)
        self.assertEqual(result["hits_at_3_pct"], 50.0)
        expected_ndcg = 100 * (1.0 + 1.0 / math.log2(3)) / 4
        self.assertAlmostEqual(result["ndcg_at_3_pct"], expected_ndcg)
        self.assertEqual(result["status_json_error"], 1)

    def test_duplicate_output_does_not_shift_target(self):
        candidates = ["Target Item", "Other Item"]
        data = {
            "u1": example(
                "Target Item",
                candidates,
                ["Other Item", "Other Item", "Target Item"],
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed_data.json"
            path.write_text(json.dumps(data))
            result = evaluate_file(path, cohort_size=1, topks=(1, 3))
        self.assertEqual(result["hits_at_1_pct"], 0.0)
        self.assertEqual(result["hits_at_3_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
