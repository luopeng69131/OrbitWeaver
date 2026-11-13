"""A generated workload illustrating structural tools and candidate validation."""
from pathlib import Path
import json
from .model import Graph, Node, Plan
from .data.scenarios import build_problem
from .backends import CriticalPathBackend, EventReplayBackend
from .planner import Planner


def example_graph(ranks=8):
    nodes = []
    for rank in range(ranks):
        nodes.append(Node("cpu%d" % rank, "cpu", rank=rank, duration_s=0.0002))
        nodes.append(Node("compute%d" % rank, "compute", rank=rank, duration_s=0.003 + rank * 0.0001,
                          deps=("cpu%d" % rank,)))
    halves = [tuple(range(ranks // 2)), tuple(range(ranks // 2, ranks))]
    for index, members in enumerate(halves):
        nodes.append(Node("group%d" % index, "collective", members=members, bytes=16 * 1024**2,
                          deps=tuple("compute%d" % r for r in members), source_order=1))
        for rank in members:
            nodes.append(Node("middle%d" % rank, "compute", rank=rank, duration_s=0.002,
                              deps=("group%d" % index,)))
    nodes.append(Node("global", "collective", members=tuple(range(ranks)), bytes=32 * 1024**2,
                      deps=tuple("middle%d" % r for r in range(ranks)), source_order=2))
    for rank in range(ranks):
        nodes.append(Node("finish%d" % rank, "compute", rank=rank, duration_s=0.001,
                          deps=("global",)))
    return Graph(ranks, nodes, {"source": "procedurally generated example"})


def run_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    problem = build_problem(example_graph(), jobs=2, condition="slow-links")
    planner = Planner(problem, CriticalPathBackend(), EventReplayBackend(), output)
    (output / "problem.json").write_text(json.dumps(problem.to_dict(), indent=2) + "\n")
    initial = planner.best()["result"]["mean_jct_s"]
    candidates = []
    for pattern in range(4):
        placement = []
        for job, slots in enumerate(problem.allocations):
            p = list(slots)
            if pattern == 1:
                p.reverse()
            elif pattern == 2:
                p = p[::2] + p[1::2]
            elif pattern == 3:
                shift = (job + 1) % len(p)
                p = p[shift:] + p[:shift]
            placement.extend(p)
        candidates.append(placement)
    planner.analyze(candidates)
    ranked = []
    for placement in candidates:
        for rule in ("progress", "shortest"):
            plan = Plan(tuple(placement), rule)
            result = planner.probe(plan.to_dict(), "fast")
            ranked.append((result["mean_jct_s"], plan))
    for _, plan in sorted(ranked, key=lambda row: row[0])[:4]:
        planner.probe(plan.to_dict(), "reference")
    result = planner.summary()
    result["initial_mean_jct_s"] = initial
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
