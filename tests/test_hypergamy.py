"""Regression and edge-case checks for the HYPERGAMY engine.

Run with:

    python -m unittest discover -s tests -v

No pytest.  The suite deliberately exercises the same `gui.compute` function the
browser calls, so a broken JSON payload, a NaN or an out-of-range probability is
caught before the page sees it.
"""

import json
import math
import os
import random
import unittest

import gui
from profile import DEFAULTS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# The exact profile from the earlier conversation: an 18-year-old man earning
# $4,500, 185 cm, average build, serious-dating goal, very high self-ratings on
# sincerity/intelligence/fun and low on ambition.
MY_PROFILE = {
    "mode": "adult",
    "gender": "man",
    "partner_gender": "women",
    "age": 18,
    "income": 4500,
    "goal": 4,
    "self": [6, 8, 7, 7, 3],
    "want": [6, 7, 6, 6, 5, 5],
    "importance": [30, 10, 15, 15, 20, 10],
    "height_cm": 185,
    "body_type": "average",
    "ethnicity": "white",
    "age_gap": 3,
    "same_race": True,
    "kink_match": 3,
    "libido_match": 3,
    "size_cm": 12.5,
    "own_cup": "C",
    "cup": "C",
    "partner_size_cm": 13.1,
    "partner_body": "average",
    "experience": 0,
    "orientation_match": True,
}

AVERAGE_GUY = {
    **MY_PROFILE,
    "age": 30,
    "income": 50000,
    "self": [5, 5, 5, 5, 3],
    "want": [5, 5, 5, 5, 5, 5],
    "height_cm": 175,
    "ethnicity": "not specified",
    "kink_match": 3,
    "libido_match": 3,
    "size_cm": 13.1,
    "own_cup": "C",
    "cup": "C",
    "partner_size_cm": 13.1,
    "experience": 3,
}

AVERAGE_WOMAN = {
    **AVERAGE_GUY,
    "gender": "woman",
    "partner_gender": "men",
    "size_cm": 13.1,
    "own_cup": "C",
    "cup": "C",
    "partner_size_cm": 13.1,
}


def run(payload):
    """Compute and also prove the result is strict JSON (no NaN/Infinity)."""
    result = gui.compute(payload)
    json.dumps(result, allow_nan=False)
    return result


class UserProfileTest(unittest.TestCase):
    def test_user_profile_has_all_sections(self):
        result = run(MY_PROFILE)
        for key in ("p_them", "p_you", "mutual", "map", "likes", "person",
                    "market_context", "research", "adjustments"):
            self.assertIn(key, result)
        self.assertGreaterEqual(result["p_them"], 0.0)
        self.assertLessEqual(result["p_them"], 1.0)
        self.assertGreaterEqual(result["p_you"], 0.0)
        self.assertLessEqual(result["p_you"], 1.0)
        self.assertAlmostEqual(result["mutual"], result["p_them"] * result["p_you"], places=9)
        self.assertIn("same race", [a["label"] for a in result["adjustments"]])
        self.assertEqual(result["market_context"]["cell"]["n"], 589)
        self.assertTrue(result["research"]["sources"])

    def test_user_profile_package_rows(self):
        result = run(MY_PROFILE)
        self.assertEqual(len(result["map"]["their_yes_by_package"]), 10)
        self.assertEqual(len(result["map"]["your_yes_by_package"]), 10)
        self.assertEqual(len(result["map"]["matrix"]), 10)
        self.assertTrue(all(len(row) == 10 for row in result["map"]["matrix"]))
        best = result["map"]["best_level"] - 1
        expected = (result["map"]["their_yes_by_package"][best]
                    * result["map"]["your_yes_by_package"][best])
        # the GUI rounds each package array to four decimals; the raw chance is not
        self.assertAlmostEqual(result["map"]["chance"], expected, places=3)
        self.assertEqual(result["map"]["best_level"],
                         max(range(1, 11),
                             key=lambda k: result["map"]["their_yes_by_package"][k - 1]
                             * result["map"]["your_yes_by_package"][k - 1]))

    def test_same_race_toggle(self):
        with_race = run({**MY_PROFILE, "same_race": True})
        without = run({**MY_PROFILE, "same_race": False})
        self.assertGreaterEqual(with_race["p_them"], without["p_them"])
        self.assertGreater(with_race["p_them"], without["p_them"])


class AverageAndRandomTest(unittest.TestCase):
    def test_average_guy_and_woman(self):
        for payload in (AVERAGE_GUY, AVERAGE_WOMAN):
            result = run(payload)
            self.assertTrue(math.isfinite(result["mutual"]))
            self.assertGreater(result["mutual"], 0.0)
            self.assertLess(result["mutual"], 1.0)
            self.assertTrue(result["person"]["package"] in range(1, 11))

    def test_many_random_profiles(self):
        rng = random.Random(7)
        for _ in range(20):
            gender = rng.choice(["man", "woman"])
            partner = rng.choice(["men", "women"])
            payload = {
                "mode": rng.choice(["regular", "advanced", "adult"]),
                "gender": gender,
                "partner_gender": partner,
                "age": rng.choice([18, 22, 25, 26, 35, 50, 80]),
                "income": rng.choice([0, 1, 4500, 32000, 75000, 10 ** 7]),
                "goal": rng.randint(1, 6),
                "self": [rng.randint(0, 10) for _ in range(5)],
                "want": [rng.randint(0, 10) for _ in range(6)],
                "importance": [rng.randint(0, 100) for _ in range(6)],
                "height_cm": rng.choice([140, 160, 178, 210]),
                "body_type": rng.choice(["thin", "athletic", "average", "curvy",
                                         "full figured", "a little extra"]),
                "ethnicity": rng.choice(["not specified", "white", "black", "latino",
                                         "asian", "pacific", "other"]),
                "age_gap": rng.randint(0, 20),
                "same_race": rng.choice([True, False]),
                "kink_match": rng.randint(0, 5),
                "libido_match": rng.randint(0, 5),
                "size_cm": rng.choice([8, 13.1, 18]),
                "own_cup": rng.choice(["A", "B", "C", "D", "DD/E", "F or larger"]),
                "cup": rng.choice(["A", "B", "C", "D", "DD/E", "F or larger"]),
                "partner_size_cm": rng.choice([8, 13.1, 18]),
                "partner_body": rng.choice(["thin", "athletic", "average", "curvy",
                                            "full figured", "a little extra"]),
                "experience": rng.randint(0, 5),
                "orientation_match": rng.choice([True, False]),
            }
            result = run(payload)
            self.assertGreaterEqual(result["p_them"], 0.0)
            self.assertLessEqual(result["p_them"], 1.0)
            self.assertGreaterEqual(result["p_you"], 0.0)
            self.assertLessEqual(result["p_you"], 1.0)
            self.assertLessEqual(result["mutual"], 1.0)
            self.assertEqual(len(result["map"]["matrix"]), 10)
            for row in result["map"]["matrix"]:
                self.assertEqual(len(row), 10)
                self.assertTrue(all(0.0 <= v <= 1.0 for v in row))
            self.assertIn(result["map"]["best_level"], range(1, 11))
            self.assertEqual(len(result["likes"]["their_yes"]), 10)
            self.assertEqual(len(result["likes"]["your_yes"]), 10)
            self.assertEqual(len(result["likes"]["mutual"]), 10)

    def test_edge_values(self):
        cases = [
            {**AVERAGE_GUY, "age": 18, "income": 0, "height_cm": 140,
             "self": [0, 0, 0, 0, 0], "want": [10] * 6, "importance": [0] * 6,
             "goal": 1, "age_gap": 0, "same_race": False, "orientation_match": False},
            {**AVERAGE_GUY, "age": 80, "income": 10 ** 7, "height_cm": 210,
             "self": [10] * 5, "want": [0] * 6, "importance": [100] * 6,
             "goal": 6, "age_gap": 20, "same_race": True, "orientation_match": True},
            {**AVERAGE_WOMAN, "age": 18, "income": 0, "height_cm": 140,
             "self": [0] * 5, "want": [10] * 6, "importance": [0] * 6,
             "goal": 1, "age_gap": 0, "same_race": False, "orientation_match": False},
            {**AVERAGE_WOMAN, "age": 80, "income": 10 ** 7, "height_cm": 210,
             "self": [10] * 5, "want": [0] * 6, "importance": [100] * 6,
             "goal": 6, "age_gap": 20, "same_race": True, "orientation_match": True},
            {**AVERAGE_GUY, "gender": "man", "partner_gender": "men"},
            {**AVERAGE_WOMAN, "gender": "woman", "partner_gender": "women"},
            {**AVERAGE_GUY, "mode": "regular"},
            {**AVERAGE_WOMAN, "mode": "advanced"},
        ]
        for payload in cases:
            result = run(payload)
            self.assertTrue(all(math.isfinite(v) for v in
                                (result["p_them"], result["p_you"], result["mutual"])))
            self.assertIn(result["map"]["same_gender"], (True, False))
            if payload["gender"] == payload["partner_gender"]:
                self.assertTrue(result["map"]["same_gender"])

    def test_missing_payload_uses_defaults(self):
        result = run({})
        self.assertIn("p_them", result)
        # The default profile is a woman interested in women, so the ACS
        # opposite-sex benchmark correctly says it is not available.
        self.assertFalse(result["market_context"]["available"])
        self.assertTrue(result["market_context"]["same_gender"])


class AnatomyTest(unittest.TestCase):
    def test_man_size_is_his_side_only(self):
        small = run({**MY_PROFILE, "size_cm": 8})
        large = run({**MY_PROFILE, "size_cm": 18})
        self.assertGreater(large["p_them"], small["p_them"])
        self.assertAlmostEqual(large["p_you"], small["p_you"], places=9)

    def test_man_partner_size_is_not_used_when_dating_women(self):
        small = run({**MY_PROFILE, "partner_size_cm": 8})
        large = run({**MY_PROFILE, "partner_size_cm": 18})
        self.assertAlmostEqual(large["p_them"], small["p_them"], places=9)
        self.assertAlmostEqual(large["p_you"], small["p_you"], places=9)

    def test_man_partner_chest_is_his_side(self):
        # Use the average profile: the user profile is already near the 1% floor,
        # where a small research prior is deliberately clipped.
        small = run({**AVERAGE_GUY, "cup": "A"})
        large = run({**AVERAGE_GUY, "cup": "F or larger"})
        self.assertGreater(large["p_you"], small["p_you"])
        self.assertAlmostEqual(large["p_them"], small["p_them"], places=9)

    def test_woman_own_chest_is_their_side(self):
        small = run({**AVERAGE_WOMAN, "own_cup": "A"})
        large = run({**AVERAGE_WOMAN, "own_cup": "F or larger"})
        self.assertGreater(large["p_them"], small["p_them"])
        self.assertAlmostEqual(large["p_you"], small["p_you"], places=9)

    def test_woman_preferred_size_is_her_side(self):
        small = run({**AVERAGE_WOMAN, "partner_size_cm": 8})
        large = run({**AVERAGE_WOMAN, "partner_size_cm": 18})
        self.assertGreater(large["p_you"], small["p_you"])
        self.assertAlmostEqual(large["p_them"], small["p_them"], places=9)

    def test_woman_own_size_is_ignored(self):
        small = run({**AVERAGE_WOMAN, "size_cm": 8})
        large = run({**AVERAGE_WOMAN, "size_cm": 18})
        self.assertAlmostEqual(large["p_them"], small["p_them"], places=9)
        self.assertAlmostEqual(large["p_you"], small["p_you"], places=9)


class AccuracyMetadataTest(unittest.TestCase):
    def test_committed_cv_values(self):
        expected = {
            "weights.json": 0.7739,
            "weights_deep_men.json": 0.8112,
            "weights_deep_women.json": 0.8031,
            "weights_market_men.json": 0.6607,
            "weights_market_women.json": 0.6386,
            "weights_young.json": 0.7975,
            "weights_likes_men.json": 0.8095,
            "weights_likes_women.json": 0.7949,
        }
        for name, cv in expected.items():
            with open(os.path.join(HERE, name), encoding="utf-8") as f:
                model = json.load(f)
            self.assertAlmostEqual(model["accuracy_cv"], cv, places=4, msg=name)
            self.assertGreater(model["n_rows"], 3000)

    def test_html_has_dynamic_anatomy_and_accessible_controls(self):
        self.assertIn("partner_size_cm", gui.PAGE)
        self.assertIn("own_cup", gui.PAGE)
        self.assertIn("preferred penis size", gui.PAGE)
        self.assertIn("their chest (cup size)", gui.PAGE)
        self.assertIn('id="size_group"', gui.PAGE)
        self.assertIn('id="cup_group"', gui.PAGE)
        self.assertIn('id="partner_size_group"', gui.PAGE)
        self.assertIn("aria-live", gui.PAGE)
        self.assertIn("@media (max-width: 980px)", gui.PAGE)



class PartnerAndCalibrationTest(unittest.TestCase):
    """The measured "who you end up with" panel and the honesty fixes."""

    def test_measured_partner_panel(self):
        result = run(MY_PROFILE)
        panel = result["partner"]
        self.assertTrue(panel["available"])
        self.assertGreater(panel["n_couples"], 200_000)
        self.assertEqual(len(panel["candidates"]), 9)
        self.assertTrue(panel["most_typical"]["most_typical"])
        self.assertGreaterEqual(panel["typical"]["same_education_pct"], 0.0)
        self.assertLessEqual(panel["typical"]["same_education_pct"], 100.0)
        self.assertIn("realised couples", panel["note"])
        self.assertNotIn("mutual", json.dumps(panel["candidates"]))

    def test_partner_ladder_is_measured(self):
        panel = run(MY_PROFILE)["partner"]
        self.assertIsNotNone(panel["ladder_level"])
        self.assertGreaterEqual(panel["ladder_level"], 1.0)
        self.assertLessEqual(panel["ladder_level"], 10.0)
        self.assertTrue(panel["self_median"])

    def test_same_gender_partner_is_unavailable(self):
        result = run({**MY_PROFILE, "partner_gender": "men"})
        self.assertFalse(result["partner"]["available"])
        self.assertTrue(result["partner"]["same_gender"])

    def test_out_of_support_profile_warns(self):
        result = run({**MY_PROFILE, "age": 80, "income": 1_000_000})
        self.assertTrue(result["warnings"])
        self.assertTrue(any("extrapolation" in w or "reliable" in w
                            for w in result["warnings"]))

    def test_calibration_block(self):
        result = run(MY_PROFILE)
        cal = result["calibration"]
        self.assertEqual(cal["scope"], "in-sample")
        self.assertGreater(cal["n"], 3000)
        self.assertTrue(cal["taste"])
        self.assertTrue(cal["market"])

    def test_education_uses_the_acs_table(self):
        result = run({**MY_PROFILE, "education": "graduate"})
        cell = result["market_context"]["education_cell"]
        self.assertIsNotNone(cell)
        self.assertGreater(cell["n"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
