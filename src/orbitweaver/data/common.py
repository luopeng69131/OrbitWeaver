"""Provenance and dependency-preserving collective normalization."""
from pathlib import Path
from collections import defaultdict
import hashlib
from ..model import Graph, Node


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize(records, ranks, metadata=None):
    nodes = {}
    mappings = {}
    pending = []
    expected = {}
    seen_rank_ids = set()
    for row in records:
        rank, original_id = row["rank"], str(row["id"])
        if (rank, original_id) in seen_rank_ids:
            raise ValueError("Duplicate source node ID within a rank")
        seen_rank_ids.add((rank, original_id))
        if row["kind"] == "collective":
            identity = row["identity"]
            if identity not in nodes:
                nodes[identity] = Node(identity, "collective", members=(), bytes=row["bytes"],
                                       operation=row["operation"], source_order=row["source_order"],
                                       metadata={"rank_node_ids": [], "recorded_durations_s": []})
            node = nodes[identity]
            if rank in node.members or node.bytes != row["bytes"] or node.operation != row["operation"]:
                raise ValueError("Inconsistent collective participant records")
            node.members += (rank,)
            node.metadata["rank_node_ids"].append([rank, original_id])
            node.metadata["recorded_durations_s"].append(row["duration_s"])
            if identity in expected and expected[identity] != row["group_size"]:
                raise ValueError("Conflicting collective group sizes")
            expected[identity] = row["group_size"]
            nid = identity
        else:
            nid = "r%d:%s" % (rank, original_id)
            nodes[nid] = Node(nid, row["kind"], rank=rank, duration_s=row["duration_s"],
                              source_order=row["source_order"], metadata=row.get("metadata", {}))
        mappings[rank, original_id] = nid
        pending.append((nid, rank, row["deps"]))
    dependencies = defaultdict(set)
    for nid, rank, deps in pending:
        for parent in deps:
            key = (rank, str(parent))
            if key not in mappings:
                raise ValueError("Missing source dependency: " + repr(key))
            dependencies[nid].add(mappings[key])
    for nid, node in nodes.items():
        node.deps = tuple(sorted(dependencies[nid]))
        if node.kind == "collective":
            node.members = tuple(sorted(node.members))
            if len(node.members) != expected[nid]:
                raise ValueError("Incomplete collective participant coverage: " + nid)
    graph = Graph(ranks, list(nodes.values()), metadata or {})
    graph.topology()
    return graph
