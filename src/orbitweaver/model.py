"""Execution graphs, optical links and allocation-preserving joint plans."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from collections import deque
import hashlib
import json
import math

RULES = {"progress", "shortest", "tail", "normalized_tail",
         "job_asc", "job_desc", "source_order", "zero"}


@dataclass
class Node:
    id: str
    kind: str
    job: int = 0
    rank: int | None = None
    duration_s: float = 0.0
    deps: tuple[str, ...] = ()
    members: tuple[int, ...] = ()
    bytes: int = 0
    operation: str = "allreduce"
    source_order: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.id = str(self.id)
        self.deps = tuple(map(str, self.deps))
        self.members = tuple(self.members)


@dataclass
class Graph:
    ranks: int
    nodes: list[Node]
    metadata: dict = field(default_factory=dict)

    def topology(self):
        lookup = {n.id: n for n in self.nodes}
        if len(lookup) != len(self.nodes) or not self.nodes or self.ranks < 1:
            raise ValueError("A nonempty graph with unique node IDs and positive rank count is required")
        children = {n.id: [] for n in self.nodes}
        indegree = {}
        for node in self.nodes:
            if node.kind not in {"compute", "cpu", "metadata", "collective"}:
                raise ValueError("Unknown node kind: " + node.kind)
            if not math.isfinite(node.duration_s) or node.duration_s < 0:
                raise ValueError("Node durations must be finite and nonnegative")
            if type(node.bytes) is not int or node.bytes < 0:
                raise ValueError("Communication bytes must be nonnegative integers")
            ranks = node.members if node.kind == "collective" else (node.rank,)
            if not ranks or len(set(ranks)) != len(ranks) or any(type(r) is not int or not 0 <= r < self.ranks for r in ranks):
                raise ValueError("Invalid rank membership: " + node.id)
            if len(set(node.deps)) != len(node.deps):
                raise ValueError("Duplicate dependency: " + node.id)
            for parent in node.deps:
                if parent not in lookup or parent == node.id:
                    raise ValueError("Missing or self dependency: " + node.id)
                children[parent].append(node.id)
            indegree[node.id] = len(node.deps)
        pending = indegree.copy()
        ready = deque(n.id for n in self.nodes if pending[n.id] == 0)
        order = []
        while ready:
            nid = ready.popleft()
            order.append(nid)
            for child in children[nid]:
                pending[child] -= 1
                if pending[child] == 0:
                    ready.append(child)
        if len(order) != len(self.nodes):
            raise ValueError("Execution dependencies contain a cycle")
        return lookup, children, indegree, order

    def to_dict(self):
        return {"schema_version": 1, "ranks": self.ranks,
                "nodes": [asdict(n) for n in self.nodes], "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data):
        graph = cls(data["ranks"], [Node(**n) for n in data["nodes"]], data.get("metadata", {}))
        graph.topology()
        return graph


@dataclass(frozen=True)
class Link:
    src: int
    dst: int
    bandwidth_Bps: float
    latency_s: float


@dataclass
class Problem:
    graph: Graph
    allocations: list[list[int]]
    arrivals_s: list[float]
    links: list[Link]
    name: str = "orbital-cluster"

    def validate(self):
        self.graph.topology()
        if not self.allocations or len(self.allocations) != len(self.arrivals_s):
            raise ValueError("Each job needs an allocation and arrival time")
        width = len(self.allocations[0])
        flat = [r for group in self.allocations for r in group]
        if width < 1 or any(len(g) != width for g in self.allocations):
            raise ValueError("Jobs must have equal nonempty allocations")
        if sorted(flat) != list(range(self.graph.ranks)):
            raise ValueError("Allocations must partition the physical endpoints")
        if any(not math.isfinite(t) or t < 0 for t in self.arrivals_s):
            raise ValueError("Arrival times must be finite and nonnegative")
        if len({(e.src, e.dst) for e in self.links}) != len(self.links):
            raise ValueError("Directed links must be unique")
        for link in self.links:
            if not (0 <= link.src < len(flat) and 0 <= link.dst < len(flat)) or link.src == link.dst:
                raise ValueError("Invalid link endpoints")
            if not math.isfinite(link.bandwidth_Bps) or link.bandwidth_Bps <= 0 or not math.isfinite(link.latency_s) or link.latency_s < 0:
                raise ValueError("Invalid link capacity or latency")
        for n in self.graph.nodes:
            ranks = n.members if n.kind == "collective" else (n.rank,)
            if any(r // width != n.job for r in ranks):
                raise ValueError("Node job and logical ranks disagree: " + n.id)
        return self

    @property
    def initial_placement(self):
        return tuple(r for group in self.allocations for r in group)

    def to_dict(self):
        return {"schema_version": 1, "name": self.name, "graph": self.graph.to_dict(),
                "allocations": self.allocations, "arrivals_s": self.arrivals_s,
                "links": [asdict(e) for e in self.links]}

    @classmethod
    def from_dict(cls, data):
        return cls(Graph.from_dict(data["graph"]), data["allocations"], data["arrivals_s"],
                   [Link(**e) for e in data["links"]], data.get("name", "orbital-cluster")).validate()

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Plan:
    placement: tuple[int, ...]
    rule: str = "progress"
    alpha: float = 0.0
    beta: float = 0.0

    def to_dict(self):
        return {"placement": list(self.placement),
                "priority": {"rule": self.rule, "alpha": self.alpha, "beta": self.beta}}

    @classmethod
    def from_dict(cls, data):
        spec = data.get("priority", {})
        if set(data) - {"placement", "priority"} or set(spec) - {"rule", "alpha", "beta"}:
            raise ValueError("Unknown plan fields")
        return cls(tuple(data["placement"]), spec.get("rule", "progress"),
                   spec.get("alpha", 0.0), spec.get("beta", 0.0))

    def validate(self, problem):
        if len(self.placement) != problem.graph.ranks or any(type(v) is not int for v in self.placement):
            raise ValueError("One integer endpoint is required for each logical rank")
        width = len(problem.allocations[0])
        for j, slots in enumerate(problem.allocations):
            if sorted(self.placement[j * width:(j + 1) * width]) != sorted(slots):
                raise ValueError("A placement must preserve each job's GPU allocation")
        if self.rule not in RULES or any(type(v) not in (int, float) or not math.isfinite(v) for v in (self.alpha, self.beta)):
            raise ValueError("A supported rule and finite parameters are required")
        return self
