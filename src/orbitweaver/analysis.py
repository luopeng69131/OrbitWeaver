"""Expose path sharing, communication subgroups and dependency progress."""
from collections import defaultdict
from .model import Plan
from .network import routes, collective_service


def features(problem, plan):
    plan.validate(problem)
    lookup, children, _, order = problem.graph.topology()
    paths = routes(problem)
    result = {}
    weight = {}
    for n in problem.graph.nodes:
        if n.kind == "collective":
            service, links, demand = collective_service(n, plan, problem, paths)
            result[n.id] = {"node": n.id, "job": n.job, "members": list(n.members),
                            "reserved_links": list(links), "link_bytes": demand,
                            "service_s": service, "bytes": n.bytes, "source_order": n.source_order}
            weight[n.id] = service
        else:
            weight[n.id] = n.duration_s
    tail = {}
    for nid in reversed(order):
        tail[nid] = weight[nid] + max((tail[c] for c in children[nid]), default=0.0)
    for nid, row in result.items():
        row["tail_s"] = tail[nid]
    return result, weight, tail


def summarize(problem, plan):
    feat, _, _ = features(problem, plan)
    grouped = {}
    loads = defaultdict(float)
    for row in feat.values():
        key = (row["job"], tuple(row["members"]), tuple(row["reserved_links"]))
        item = grouped.setdefault(key, {"job": row["job"], "members": row["members"],
                                       "links": row["reserved_links"], "calls": 0,
                                       "modeled_service_s": 0.0, "bytes": 0, "max_tail_s": 0.0})
        item["calls"] += 1
        item["modeled_service_s"] += row["service_s"]
        item["bytes"] += row["bytes"]
        item["max_tail_s"] = max(item["max_tail_s"], row["tail_s"])
        for edge, amount in row["link_bytes"].items():
            loads[edge] += amount
    return {"placement": list(plan.placement),
            "groups": sorted(grouped.values(), key=lambda g: -g["modeled_service_s"]),
            "link_demand_bytes": dict(loads)}


def priorities(problem, plan, feat):
    spans = {j: max((r["tail_s"] for r in feat.values() if r["job"] == j), default=0.0)
             for j in range(len(problem.allocations))}
    repeats = problem.graph.metadata.get("window_repeats", [1] * len(spans))
    scale = sum(spans[j] / repeats[j] for j in spans) / len(spans)
    max_order = max((r["source_order"] for r in feat.values()), default=0) or 1
    scores = {}
    for nid, row in feat.items():
        j, tail = row["job"], row["tail_s"]
        if plan.rule == "progress":
            score = ((1 - plan.beta) * (tail - spans[j])
                     - plan.beta * row["source_order"] / max_order * scale
                     - plan.alpha * problem.arrivals_s[j])
        elif plan.rule == "shortest":
            score = -row["service_s"]
        elif plan.rule == "tail":
            score = tail
        elif plan.rule == "normalized_tail":
            score = tail / repeats[j]
        elif plan.rule == "job_asc":
            score = tail - 1e6 * j
        elif plan.rule == "job_desc":
            score = tail + 1e6 * j
        elif plan.rule == "source_order":
            score = -row["source_order"]
        else:
            score = 0.0
        scores[nid] = score
    return scores
