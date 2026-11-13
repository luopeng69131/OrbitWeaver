from pathlib import Path
import io
import json
import tarfile
import tempfile
import unittest
import zipfile

from orbitweaver import Graph, Node, Problem
from orbitweaver.data import build_problem, load_a100, load_chakra, repeat_jobs
from orbitweaver.data.chakra import messages
from orbitweaver.data.h200 import extract_mixtral
from orbitweaver._vendor import et_def_pb2 as et


def framed(message):
    raw = message.SerializeToString()
    size = len(raw)
    prefix = bytearray()
    while size >= 128:
        prefix.append((size & 127) | 128)
        size >>= 7
    prefix.append(size)
    return bytes(prefix) + raw


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def capture(self, ranks=2):
        for rank in range(ranks):
            payload = {"rank": rank, "world_size": ranks, "model": "generated",
                       "records": [{"kind": "compute", "sequence": None, "bytes": 0,
                                    "duration_us": 100 + rank},
                                   {"kind": "collective", "sequence": 0, "bytes": 1024,
                                    "duration_us": 10}]}
            (self.root / ("rank_%d.json" % rank)).write_text(json.dumps(payload))
        return load_a100(self.root)

    def test_segment_pairing_preserves_compute(self):
        graph = self.capture()
        self.assertEqual(len(graph.nodes), 5)
        coll = next(n for n in graph.nodes if n.kind == "collective")
        self.assertEqual(coll.members, (0, 1))
        durations = [n.duration_s for n in graph.nodes if n.kind == "compute"]
        for actual, expected in zip(durations, [100e-6, 101e-6]):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_capture_mismatch_is_detected(self):
        self.capture()
        p = self.root / "rank_1.json"
        payload = json.loads(p.read_text())
        payload["records"][-1]["sequence"] = 9
        p.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            load_a100(self.root)

    def test_repetition_adds_rank_local_bridges(self):
        graph = self.capture()
        repeated = repeat_jobs(graph, jobs=2)
        self.assertEqual(repeated.ranks, 4)
        self.assertEqual(len(repeated.nodes), 3 * len(graph.nodes))
        self.assertEqual(repeated.metadata["window_repeats"], [2, 1])
        roots = [n for n in repeated.nodes if n.id.startswith("j0:w1:") and n.kind == "compute"]
        self.assertTrue(all(any(p.startswith("j0:w0:") for p in n.deps) for n in roots))
        repeated.topology()

    def test_scenario_roundtrip_and_directional_units(self):
        problem = build_problem(self.capture(), jobs=2, condition="staggered", arrival_gap_s=0.2)
        decoded = Problem.from_dict(problem.to_dict())
        self.assertEqual(problem.fingerprint, decoded.fingerprint)
        self.assertEqual(problem.links[0].bandwidth_Bps, 6.25e9)
        self.assertEqual(problem.arrivals_s, [0.0, 0.2])

    def test_cycle_is_detected(self):
        graph = Graph(1, [Node("a", "compute", rank=0, deps=("b",)),
                          Node("b", "compute", rank=0, deps=("a",))])
        with self.assertRaises(ValueError):
            graph.topology()

    def test_chakra_normalization(self):
        for rank in range(2):
            compute = et.Node(id=1, type=et.COMP_NODE, duration_micros=20)
            compute.attr.add(name="is_cpu_op", bool_val=False)
            collective = et.Node(id=2, type=et.COMM_COLL_NODE, data_deps=[1])
            collective.attr.add(name="pg_name", string_val="0")
            collective.attr.add(name="comm_tag", uint32_val=0)
            collective.attr.add(name="comm_type", int64_val=0)
            collective.attr.add(name="comm_size", int64_val=128)
            raw = framed(et.GlobalMetadata(version="1.0.3")) + framed(compute) + framed(collective)
            (self.root / ("trace.%d.et" % rank)).write_bytes(raw)
        graph = load_chakra(self.root)
        self.assertEqual(len(graph.nodes), 3)
        collective = next(n for n in graph.nodes if n.kind == "collective")
        self.assertEqual(len(collective.deps), 2)
        self.assertEqual(len(collective.members), 2)

    def test_incomplete_chakra_collective_is_detected(self):
        for rank in range(2):
            records = [et.GlobalMetadata(version="1.0.3"), et.Node(id=1, type=et.COMP_NODE)]
            if rank == 0:
                coll = et.Node(id=2, type=et.COMM_COLL_NODE)
                for name, value in [("comm_type", 0), ("comm_size", 128)]:
                    coll.attr.add(name=name, int64_val=value)
                coll.attr.add(name="comm_tag", uint32_val=0)
                records.append(coll)
            (self.root / ("trace.%d.et" % rank)).write_bytes(b"".join(framed(n) for n in records))
        with self.assertRaises(ValueError):
            load_chakra(self.root)

    def test_truncated_protobuf_is_detected(self):
        p = self.root / "trace.0.et"
        p.write_bytes(framed(et.GlobalMetadata(version="1")) + b"\x80")
        with self.assertRaises(ValueError):
            list(messages(p))

    def test_zip_and_targz_content_detection(self):
        expected = {kind + "_" + str(rank) + ".json" for kind in ("host", "device") for rank in range(8)}
        for encoding in ("zip", "tar"):
            archive = self.root / (encoding + ".zip")
            if encoding == "zip":
                with zipfile.ZipFile(archive, "w") as target:
                    for name in expected:
                        target.writestr("records/" + name, "{}")
                    target.writestr("../outside.txt", "ignored")
            else:
                with tarfile.open(archive, "w:gz") as target:
                    for name in expected:
                        raw = b"{}"
                        info = tarfile.TarInfo("records/" + name)
                        info.size = len(raw)
                        target.addfile(info, io.BytesIO(raw))
            output = self.root / encoding
            record = extract_mixtral(archive, output, expected_md5=None)
            self.assertEqual(set(record["files"]), expected)
        self.assertFalse((self.root.parent / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
