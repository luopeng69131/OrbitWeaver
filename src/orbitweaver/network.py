"""Directed optical routes and whole-collective service demand."""
from collections import deque
from .model import Link


def ring_links(endpoints, bandwidth_Gbps=50.0, spacing_m=300.0, terminal_delay_s=1e-6):
    if endpoints < 2 or bandwidth_Gbps <= 0 or spacing_m < 0 or terminal_delay_s < 0:
        raise ValueError("Positive capacity and a connected ring are required")
    return [Link(i, j, bandwidth_Gbps * 1e9 / 8, spacing_m / 299792458.0 + terminal_delay_s)
            for i in range(endpoints) for j in dict.fromkeys(((i + 1) % endpoints, (i - 1) % endpoints))]


def routes(problem):
    adjacency = {r: [] for r in range(problem.graph.ranks)}
    for index, edge in enumerate(problem.links):
        adjacency[edge.src].append((edge.dst, index))
    result = {}
    for source in adjacency:
        pred = {source: None}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for target, edge in adjacency[current]:
                if target not in pred:
                    pred[target] = (current, edge)
                    queue.append(target)
        if len(pred) != len(adjacency):
            raise ValueError("The physical interconnect must be strongly connected")
        for target in adjacency:
            path = []
            current = target
            while current != source:
                current, edge = pred[current]
                path.append(edge)
            result[source, target] = tuple(reversed(path))
    return result


def collective_service(node, plan, problem, paths):
    members = sorted(node.members)
    k = len(members)
    if k < 2 or node.bytes == 0:
        return 0.0, (), {}
    if node.operation == "allreduce":
        volume, rounds = node.bytes * 2 * (k - 1) / k, 2 * (k - 1)
    elif node.operation == "allgather":
        volume, rounds = node.bytes * (k - 1), k - 1
    elif node.operation == "reducescatter":
        volume, rounds = node.bytes * (k - 1) / k, k - 1
    elif node.operation == "alltoall":
        volume, rounds = node.bytes * (k - 1) / 2, k * (k - 1) // 2
    elif node.operation == "broadcast":
        volume, rounds = node.bytes, k - 1
    elif node.operation == "barrier":
        return 0.0, (), {}
    else:
        raise ValueError("Unsupported collective operation: " + node.operation)
    pairs = list(zip(members, members[1:])) if node.operation == "broadcast" else list(zip(members, members[1:] + members[:1]))
    loads = {}
    longest_latency = 0.0
    for a, b in pairs:
        path = paths[plan.placement[a], plan.placement[b]]
        longest_latency = max(longest_latency, sum(problem.links[e].latency_s for e in path))
        for edge in path:
            loads[edge] = loads.get(edge, 0.0) + volume
    service = max((v / problem.links[e].bandwidth_Bps for e, v in loads.items()), default=0.0)
    return service + rounds * longest_latency, tuple(sorted(loads)), loads
