import json
from pathlib import Path
import sys
import tempfile
import unittest

from orbitweaver import Plan, Planner
from orbitweaver.backends import CommandBackend, CriticalPathBackend, EventReplayBackend
from orbitweaver.data import build_problem
from orbitweaver.demo import example_graph


class FixedBackend:
    def __init__(self, value):
        self.value = value

    def evaluate(self, problem, plan):
        return {"mean_jct_s": self.value, "jct_s": [self.value] * len(problem.allocations),
                "audits": {"complete_graph": True}}


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.problem = build_problem(example_graph(), jobs=2, condition="slow-links")
        self.plan = Plan(self.problem.initial_placement)

    def planner(self, fast=None, reference=None, **kwargs):
        return Planner(self.problem, fast or CriticalPathBackend(), reference or EventReplayBackend(),
                       self.temp.name, **kwargs)

    def test_reference_is_authoritative(self):
        planner = self.planner(FixedBackend(1.0), FixedBackend(10.0))
        planner.probe(self.plan.to_dict(), "fast")
        self.assertEqual(planner.best()["result"]["mean_jct_s"], 10.0)

    def test_invalid_request_is_charged(self):
        planner = self.planner()
        before = planner.budget()["fast"]
        bad = self.plan.to_dict()
        bad["placement"][0], bad["placement"][8] = bad["placement"][8], bad["placement"][0]
        with self.assertRaises(ValueError):
            planner.probe(bad, "fast")
        self.assertEqual(planner.budget()["fast"], before - 1)
        self.assertEqual(planner.records[-1]["status"], "error")

    def test_cache_hit_is_charged(self):
        planner = self.planner()
        planner.probe(self.plan.to_dict(), "fast")
        before = planner.budget()["fast"]
        planner.probe(self.plan.to_dict(), "fast")
        self.assertTrue(planner.records[-1]["cache_hit"])
        self.assertEqual(planner.budget()["fast"], before - 1)

    def test_seed_uses_reference_budget(self):
        planner = self.planner(reference_budget=1)
        self.assertEqual(planner.budget()["reference"], 0)
        with self.assertRaises(ValueError):
            planner.probe(self.plan.to_dict(), "reference")

    def test_fresh_session_preserves_existing_evidence(self):
        self.planner()
        with self.assertRaises(FileExistsError):
            self.planner()

    def test_bad_backend_result_retains_incumbent(self):
        planner = self.planner(FixedBackend(float("nan")), FixedBackend(10.0))
        with self.assertRaises(ValueError):
            planner.probe(self.plan.to_dict(), "fast")
        self.assertEqual(planner.best()["result"]["mean_jct_s"], 10.0)

    def test_complete_resource_replay_and_waits(self):
        result = EventReplayBackend().evaluate(self.problem, self.plan)
        self.assertTrue(all(result["audits"].values()))
        self.assertTrue(any(c["wait_s"] > 0 for c in result["diagnostics"]["collectives"]))
        for finish, arrival, jct in zip(result["completion_s"], self.problem.arrivals_s, result["jct_s"]):
            self.assertAlmostEqual(finish - arrival, jct)

    def test_fast_estimate_and_reference_cover_all_jobs(self):
        fast = CriticalPathBackend().evaluate(self.problem, self.plan)
        reference = EventReplayBackend().evaluate(self.problem, self.plan)
        self.assertTrue(all(a <= b + 1e-10 for a, b in zip(fast["jct_s"], reference["jct_s"])))

    def test_command_backend_roundtrip(self):
        backend = CommandBackend([sys.executable, "-m", "orbitweaver.bridge"])
        direct = EventReplayBackend().evaluate(self.problem, self.plan)
        self.assertEqual(backend.evaluate(self.problem, self.plan)["jct_s"], direct["jct_s"])

    def test_tool_client_uses_live_controller(self):
        from orbitweaver.agent import tool_server
        from orbitweaver.client import Client
        planner = self.planner()
        with tool_server(planner, self.temp.name) as address:
            client = Client(address)
            before = client.budget()["fast"]
            result = client.probe(self.plan.placement)
            self.assertIn("mean_jct_s", result)
            self.assertEqual(client.budget()["fast"], before - 1)
            self.assertEqual(client.best()["plan"], planner.best()["plan"])

    def test_cost_ledger_covers_requests(self):
        planner = self.planner()
        planner.probe(self.plan.to_dict(), "fast")
        records = [json.loads(line) for line in (Path(self.temp.name) / "costs.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["wall_s"] >= 0 for r in records))


if __name__ == "__main__":
    unittest.main()
