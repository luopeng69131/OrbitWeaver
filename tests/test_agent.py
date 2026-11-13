from pathlib import Path
import json
import sys
import tempfile
import unittest

from orbitweaver import Planner
from orbitweaver.agent import run_codex
from orbitweaver.backends import CriticalPathBackend, EventReplayBackend
from orbitweaver.data import build_problem
from orbitweaver.demo import example_graph


class AgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def planner(self):
        return Planner(build_problem(example_graph(), jobs=2), CriticalPathBackend(),
                       EventReplayBackend(), self.root / "session")

    def executable(self, body):
        path = self.root / "agent-fixture"
        path.write_text("#!" + sys.executable + "\n" + body)
        path.chmod(0o755)
        return str(path)

    def test_agent_tools_and_usage_accounting(self):
        body = """import json, sys
from pathlib import Path
folder = Path(sys.argv[sys.argv.index("--cd") + 1])
sys.path.insert(0, str(folder))
import lab
plan = lab.best()["plan"]
lab.probe_plan(plan, "fast")
lab.probe_plan(plan, "reference")
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}))
"""
        planner = self.planner()
        result = run_codex(planner, self.root / "session", executable=self.executable(body))
        self.assertEqual(result["agent"]["exit_code"], 0)
        self.assertEqual(result["evaluation_attempts"], 3)
        self.assertEqual(result["agent"]["usage"]["output_tokens"], 5)
        costs = [json.loads(line) for line in (self.root / "session/costs.jsonl").read_text().splitlines()]
        self.assertEqual(costs[-1]["kind"], "model_run")
        self.assertEqual(costs[-1]["usage"]["input_tokens"], 10)

    def test_failed_agent_retains_validated_seed(self):
        planner = self.planner()
        seed = planner.best()["result"]["mean_jct_s"]
        result = run_codex(planner, self.root / "session", executable=self.executable("raise SystemExit(7)\n"))
        self.assertEqual(result["agent"]["exit_code"], 7)
        self.assertEqual(result["best"]["result"]["mean_jct_s"], seed)

    def test_timeout_retains_validated_seed(self):
        planner = self.planner()
        result = run_codex(planner, self.root / "session",
                           executable=self.executable("import time\ntime.sleep(30)\n"), wall_limit_s=0.05)
        self.assertTrue(result["agent"]["timed_out"])
        self.assertIsNotNone(result["best"])


if __name__ == "__main__":
    unittest.main()
