from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

from orbitweaver import Plan
from orbitweaver.data import load_a100, build_problem
from orbitweaver.native.backend import NativeBackend, export_compact

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/a100"


class DatasetTests(unittest.TestCase):
    def test_all_released_rank_files_match_manifest(self):
        manifest = json.loads((DATA / "manifest.json").read_text())
        self.assertEqual(len(manifest["files"]), 16)
        for record in manifest["files"]:
            path = DATA / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
            self.assertEqual(path.stat().st_size, record["bytes"])

    def test_released_traces_have_stable_preprocessing_fingerprints(self):
        manifest = json.loads((DATA / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            for model in ("transformer", "resnet18"):
                graph = load_a100(DATA / model)
                result = export_compact(graph, Path(temp) / (model + ".orb"))
                self.assertEqual(result["sha256"], manifest["workloads"][model]["native_graph"]["sha256"])


@unittest.skipUnless(shutil.which("g++"), "A C++17 compiler builds the native simulator")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.build.name) / "bin"
        subprocess.run([sys.executable, "-m", "orbitweaver", "build-simulator", "--output", str(cls.binary)],
                       capture_output=True, text=True, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.graph = load_a100(DATA / "resnet18")
        self.problem = build_problem(self.graph, jobs=2, condition="slow-links")
        self.plan = Plan(self.problem.initial_placement, "progress")

    def backend(self, diagnostics=False):
        return NativeBackend(self.binary, Path(self.temp.name) / "native", diagnostics)

    def test_complete_measured_workload_replay(self):
        result = self.backend(True).evaluate(self.problem, self.plan)
        self.assertTrue(all(result["audits"].values()))
        self.assertEqual(result["original_nodes_completed"], 1032 * 3)
        self.assertEqual(len(result["diagnostics"]["collectives"]), 32 * 3)
        self.assertTrue(all(v >= 0 for v in result["jct_s"]))

    def test_native_replay_is_deterministic(self):
        backend = self.backend()
        first = backend.evaluate(self.problem, self.plan)
        second = backend.evaluate(self.problem, self.plan)
        self.assertEqual(first["jct_s"], second["jct_s"])

    def test_custom_priorities_support_new_schedulers(self):
        scores = {n.id: 0.0 for n in self.problem.graph.nodes if n.kind == "collective"}
        result = self.backend().evaluate(self.problem, self.plan, scores=scores)
        self.assertTrue(all(result["audits"].values()))

    def test_incomplete_custom_priorities_are_detected(self):
        with self.assertRaises(ValueError):
            self.backend().evaluate(self.problem, self.plan, scores={})


if __name__ == "__main__":
    unittest.main()
