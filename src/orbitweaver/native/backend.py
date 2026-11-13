"""Build and run the native execution-graph simulator used by OrbitWeaver."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import time
import uuid
from ..analysis import priorities
from ..backends import validate_result

COMM_TYPES = {"allreduce": 0, "allgather": 2, "broadcast": 5,
              "alltoall": 6, "reducescatter": 7, "barrier": 9}
NODE_STRUCT = struct.Struct("<dQQIHBB")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_simulator(output, compiler="g++"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(compiler)
    if executable is None:
        raise RuntimeError("Install a C++17 compiler or provide --compiler")
    source = Path(__file__).parent
    records = []
    for filename, target in (("fast.cpp", "orbit_fast"), ("features.cpp", "orbit_features")):
        temp = output / (target + ".tmp")
        command = [executable, "-O3", "-std=c++17", str(source / filename), "-o", str(temp)]
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True)
        (output / (target + ".build.log")).write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(process.stderr[-4000:])
        temp.replace(output / target)
        records.append({"target": target, "command": command,
                        "wall_s": time.perf_counter() - started,
                        "sha256": file_sha(output / target)})
    manifest = {"compiler": subprocess.check_output([executable, "--version"], text=True).splitlines()[0],
                "sources": {name: file_sha(source / name) for name in ("fast.cpp", "features.cpp")},
                "binaries": records}
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_compact(graph, path):
    graph.topology()
    if not 1 <= graph.ranks <= 64:
        raise ValueError("The native compact format supports one to 64 logical ranks")
    lookup = {node.id: i for i, node in enumerate(graph.nodes)}
    edges = sorted({(lookup[parent], lookup[node.id]) for node in graph.nodes for parent in node.deps})
    indegree = [0] * len(graph.nodes)
    children = defaultdict(list)
    for parent, child in edges:
        indegree[child] += 1
        children[parent].append(child)
    offsets = [0]
    for i in range(len(graph.nodes)):
        offsets.append(offsets[-1] + len(children[i]))
    original = sum(len(n.members) if n.kind == "collective" else 1 for n in graph.nodes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sIIII", b"ORBET001", len(graph.nodes), len(edges), graph.ranks, original))
        for node in graph.nodes:
            if node.kind == "collective":
                members = sum(1 << rank for rank in node.members)
                record = (0.0, node.bytes, members, len(node.members),
                          COMM_TYPES[node.operation], 2, min(node.members))
            else:
                record = (node.duration_s, 0, 1 << node.rank, 1, 0,
                          {"cpu": 0, "compute": 1, "metadata": 3}[node.kind], node.rank)
            stream.write(NODE_STRUCT.pack(*record))
        if indegree:
            stream.write(struct.pack("<%dI" % len(indegree), *indegree))
        stream.write(struct.pack("<%dQ" % len(offsets), *offsets))
        if edges:
            stream.write(struct.pack("<%dI" % len(edges), *(child for _, child in edges)))
    return {"nodes": len(graph.nodes), "edges": len(edges), "ranks": graph.ranks,
            "original_nodes": original, "sha256": file_sha(path), "node_index": lookup}


class NativeBackend:
    """Run routed collective rounds, dependency events and resource admission."""
    def __init__(self, binary_directory, output, diagnostics=False, timeout_s=3600):
        self.binary_directory = Path(binary_directory).resolve()
        self.output = Path(output).resolve()
        self.diagnostics, self.timeout_s = diagnostics, timeout_s
        self.output.mkdir(parents=True, exist_ok=True)
        self.graphs = {}
        for name in ("orbit_fast", "orbit_features"):
            if not (self.binary_directory / name).is_file():
                raise ValueError("Build the native simulator first: " + name)

    def evaluate(self, problem, plan, scores=None):
        problem.validate()
        plan.validate(problem)
        graph_key = hashlib.sha256(json.dumps(problem.graph.to_dict(), sort_keys=True).encode()).hexdigest()
        graph_file = self.output / "graphs" / (graph_key + ".orb")
        if graph_key not in self.graphs:
            self.graphs[graph_key] = export_compact(problem.graph, graph_file)
        graph = self.graphs[graph_key]
        directory = self.output / ("evaluation-" + uuid.uuid4().hex[:12])
        directory.mkdir()
        network = directory / "network.txt"
        network.write_text(
            "%d %d\n" % (problem.graph.ranks, len(problem.links))
            + " ".join(map(str, plan.placement)) + "\n"
            + "".join("%d %d %.17g %.17g\n" % (e.src, e.dst, e.bandwidth_Bps, e.latency_s) for e in problem.links))
        environment = {k: v for k, v in os.environ.items() if not k.startswith(("ORBIT_", "ASTRA_SIM_"))}
        processes = []
        def run(command, role):
            started = time.perf_counter()
            row = {"role": role, "command": list(map(str, command))}
            processes.append(row)
            try:
                process = subprocess.run(list(map(str, command)), env=environment,
                                         capture_output=True, text=True, timeout=self.timeout_s)
                row.update(exit_code=process.returncode)
                (directory / (role + ".stdout.log")).write_text(process.stdout)
                (directory / (role + ".stderr.log")).write_text(process.stderr)
                if process.returncode:
                    raise RuntimeError("Native %s failed: %s" % (role, process.stderr[-1500:]))
            except Exception as error:
                row["error"] = str(error)
                raise
            finally:
                row["wall_s"] = time.perf_counter() - started
                (directory / "processes.json").write_text(json.dumps(processes, indent=2) + "\n")
        started = time.perf_counter()
        feature_file = directory / "features.tsv"
        run([self.binary_directory / "orbit_features", graph_file, network, feature_file], "features")
        native_features = {}
        by_index = {i: node for i, node in enumerate(problem.graph.nodes)}
        for line in feature_file.read_text().splitlines():
            index, service, tail = line.split()
            node = by_index[int(index)]
            native_features[node.id] = {"job": node.job, "service_s": float(service),
                                        "tail_s": float(tail), "source_order": node.source_order}
        expected = {n.id for n in problem.graph.nodes if n.kind == "collective"}
        if set(native_features) != expected:
            raise ValueError("Native features must cover every collective")
        if scores is None:
            scores = priorities(problem, plan, native_features)
        else:
            scores = dict(scores)
            if set(scores) != expected or any(type(v) not in (int, float) or not math.isfinite(v) for v in scores.values()):
                raise ValueError("Provide one finite priority for every collective")
        priority_file = directory / "priorities.tsv"
        priority_file.write_text("".join("%d %.17g\n" % (graph["node_index"][nid], score) for nid, score in scores.items()))
        environment["ORBIT_PRIORITY_FILE"] = str(priority_file)
        width = len(problem.allocations[0])
        environment["ORBIT_RANK_RELEASE_NS"] = ",".join(str(round(problem.arrivals_s[r // width] * 1e9))
                                                         for r in range(problem.graph.ranks))
        if self.diagnostics:
            environment["ORBIT_EVENT_LOG"] = str(directory / "events.jsonl")
        raw_path = directory / "native_result.json"
        run([self.binary_directory / "orbit_fast", graph_file, network, raw_path, "ring"], "simulation")
        raw = json.loads(raw_path.read_text())
        completion = [max(raw["rank_completion_s"][j * width:(j + 1) * width])
                      for j in range(len(problem.allocations))]
        jct = [finish - arrival for finish, arrival in zip(completion, problem.arrivals_s)]
        audits = {"complete_graph": raw["complete"] and raw["nodes_completed"] == graph["nodes"],
                  "original_nodes": raw["original_nodes_completed"] == graph["original_nodes"],
                  "collective_coverage": raw["collectives_completed"] == len(expected),
                  "rank_coverage": len(raw["rank_completion_s"]) == problem.graph.ranks}
        result = {"mean_jct_s": sum(jct) / len(jct), "jct_s": jct, "completion_s": completion,
                  "backend": "native_round_replay", "audits": audits,
                  "wall_s": time.perf_counter() - started, "processes": processes,
                  "nodes_completed": raw["nodes_completed"], "original_nodes_completed": raw["original_nodes_completed"],
                  "graph_sha256": graph["sha256"], "output": str(directory)}
        if self.diagnostics:
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            timing = {}
            for event in events:
                if event["event"] in {"ready", "start", "end"}:
                    timing.setdefault(event["node"], {})[event["event"]] = event["time_s"]
            timeline = []
            for index, row in timing.items():
                if {"ready", "start", "end"} <= set(row):
                    node = by_index[index]
                    timeline.append({"node": node.id, "job": node.job, "ready_s": row["ready"],
                                     "start_s": row["start"], "finish_s": row["end"],
                                     "wait_s": max(0.0, row["start"] - row["ready"])})
            result["diagnostics"] = {"collectives": timeline}
        validate_result(result, len(problem.allocations))
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
