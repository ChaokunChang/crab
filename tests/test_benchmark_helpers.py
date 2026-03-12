from __future__ import annotations

import unittest

from benchmarks.bench_tree_search import choose_replay_steps
from benchmarks.real_host_scenario_base import compute_summary, total_actions


class BenchmarkHelperTests(unittest.TestCase):
    def test_choose_replay_steps_is_deterministic(self) -> None:
        self.assertEqual(choose_replay_steps(6, 2), [1, 3])
        self.assertEqual(choose_replay_steps(4, 10), [1, 2, 3])

    def test_compute_summary_averages_metrics(self) -> None:
        rows = [
            {"checkpoint_ms": 10.0, "restore_ms": 20.0},
            {"checkpoint_ms": 30.0, "restore_ms": 40.0},
        ]
        self.assertEqual(
            compute_summary(rows, ["checkpoint_ms", "restore_ms"]),
            {"checkpoint_ms": 20.0, "restore_ms": 30.0},
        )

    def test_total_actions_reads_payload(self) -> None:
        self.assertEqual(total_actions({"total_actions": 7}), 7)


if __name__ == "__main__":
    unittest.main()
