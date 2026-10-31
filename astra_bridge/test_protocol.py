import unittest
from protocol import validate_step


class ProtocolTests(unittest.TestCase):
    def test_sparse_actions_keep_previous_targets(self):
        self.assertEqual(validate_step({"actions": {"b": -0.4}}, ["a", "b"], [0.3, 0.2]), ([0.3, -0.4], 4))

    def test_invalid_actions_never_mutate_previous(self):
        previous = [0.3, 0.2]
        bad = [
            {"actions": {"missing": 0}}, {"actions": {"a": float("nan")}},
            {"actions": {"a": float("inf")}}, {"actions": {"a": True}},
            {"actions": {"a": 1.01}}, {"steps": 0}, {"steps": 11},
            {"steps": True}, {"steps": 2.5}, {"teleport": True},
        ]
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_step(payload, ["a", "b"], previous)
            self.assertEqual(previous, [0.3, 0.2])


if __name__ == "__main__":
    unittest.main()
