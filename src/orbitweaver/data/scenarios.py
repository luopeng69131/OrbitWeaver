"""Replay complete windows and build disjoint-allocation orbital scenarios."""
from dataclasses import asdict
import json
import random
from ..model import Graph, Node, Problem
from ..network import ring_links


def repeat_jobs(graph, jobs, repeats=None):
    if jobs < 1:
        raise ValueError("Provide at least one job")
    lookup, children, _, _ = graph.topology()
    repeats = list(repeats if repeats is not None else (2 if j % 2 == 0 else 1 for j in range(jobs)))
    if len(repeats) != jobs or any(type(n) is not int or n < 1 for n in repeats):
        raise ValueError("Each job needs a positive whole-window repetition count")
    roots, sinks = {}, {}
    for rank in range(graph.ranks):
        selected = {n.id for n in graph.nodes if rank in (n.members if n.kind == "collective" else (n.rank,))}
        roots[rank] = {nid for nid in selected if not (set(lookup[nid].deps) & selected)}
        sinks[rank] = {nid for nid in selected if not (set(children[nid]) & selected)}
    nodes = []
    for job, count in enumerate(repeats):
        for repeat in range(count):
            prefix = "j%d:w%d:" % (job, repeat)
            previous = "j%d:w%d:" % (job, repeat - 1)
            for node in graph.nodes:
                data = asdict(node)
                dependencies = {prefix + p for p in node.deps}
                if repeat:
                    for rank in range(graph.ranks):
                        if node.id in roots[rank]:
                            dependencies.update(previous + p for p in sinks[rank])
                data.update(id=prefix + node.id, job=job, deps=tuple(sorted(dependencies)),
                            rank=None if node.rank is None else node.rank + job * graph.ranks,
                            members=tuple(r + job * graph.ranks for r in node.members))
                nodes.append(Node(**data))
    result = Graph(graph.ranks * jobs, nodes, {"source": graph.metadata, "window_repeats": repeats})
    result.topology()
    return result


def build_problem(graph, jobs=2, condition="fragmented", bandwidth_Gbps=50, arrival_gap_s=0.01, seed=131000):
    if condition not in {"contiguous", "fragmented", "staggered", "slow-links"}:
        raise ValueError("Unknown operating condition")
    repeated = repeat_jobs(graph, jobs)
    slots = list(range(repeated.ranks))
    if condition != "contiguous":
        random.Random(seed).shuffle(slots)
    allocations = [sorted(slots[j * graph.ranks:(j + 1) * graph.ranks]) for j in range(jobs)]
    arrivals = [j * arrival_gap_s if condition in {"staggered", "slow-links"} else 0.0 for j in range(jobs)]
    links = ring_links(repeated.ranks, bandwidth_Gbps)
    if condition == "slow-links":
        from ..model import Link
        for block in range(0, repeated.ranks, 16):
            for src, fraction in ((block, 0.4), ((block + 8) % repeated.ranks, 0.6)):
                for i, edge in enumerate(links):
                    if edge.src == src and edge.dst == (src + 1) % repeated.ranks:
                        links[i] = Link(edge.src, edge.dst, edge.bandwidth_Bps * fraction, edge.latency_s)
    return Problem(repeated, allocations, arrivals, links, condition).validate()
