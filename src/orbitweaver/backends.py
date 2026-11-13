"""Fast estimates, dependency replay, and a JSON command adapter."""
from __future__ import annotations
from typing import Protocol
import heapq
import json
import math
import subprocess
from .analysis import features, priorities


class Backend(Protocol):
    def evaluate(self, problem, plan) -> dict: ...


class CriticalPathBackend:
    def evaluate(self, problem, plan):
        _, weight, _ = features(problem, plan)
        lookup, _, _, order = problem.graph.topology()
        finish = {}
        for nid in order:
            node = lookup[nid]
            finish[nid] = weight[nid] + max((finish[p] for p in node.deps), default=0.0)
        jct = [max(finish[n.id] for n in problem.graph.nodes if n.job == j)
               for j in range(len(problem.allocations))]
        return {"mean_jct_s": sum(jct) / len(jct), "jct_s": jct,
                "backend": "critical_path", "audits": {"complete_graph": True}}


class EventReplayBackend:
    """Dispatch ready work against GPU, CPU, communication-channel and link state."""
    def evaluate(self, problem, plan):
        feat, weight, _ = features(problem, plan)
        scores = priorities(problem, plan, feat)
        lookup, children, indegree, _ = problem.graph.topology()
        ready = {nid for nid, count in indegree.items() if count == 0}
        ready_at = {nid: problem.arrivals_s[lookup[nid].job] for nid in ready}
        events, occupied, completed, timeline = [], set(), {}, []
        now = min(problem.arrivals_s)
        serial = 0
        def resources(node):
            if node.kind == "collective":
                return {("comm", plan.placement[r]) for r in node.members} | {("link", e) for e in feat[node.id]["reserved_links"]}
            if node.kind == "metadata":
                return set()
            return {(node.kind, plan.placement[node.rank])}
        while len(completed) < len(lookup):
            while events and events[0][0] <= now:
                finish, _, nid, held = heapq.heappop(events)
                completed[nid] = finish
                occupied.difference_update(held)
                for child in children[nid]:
                    indegree[child] -= 1
                    ready_at[child] = max(ready_at.get(child, problem.arrivals_s[lookup[child].job]), finish)
                    if indegree[child] == 0:
                        ready.add(child)
            candidates = sorted(ready, key=lambda nid: (lookup[nid].kind == "collective", -scores.get(nid, 0.0), nid))
            for nid in candidates:
                if ready_at[nid] > now:
                    continue
                held = resources(lookup[nid])
                if held & occupied:
                    continue
                ready.remove(nid)
                occupied.update(held)
                serial += 1
                heapq.heappush(events, (now + weight[nid], serial, nid, held))
                if lookup[nid].kind == "collective":
                    timeline.append({"node": nid, "job": lookup[nid].job, "ready_s": ready_at[nid],
                                     "start_s": now, "finish_s": now + weight[nid],
                                     "wait_s": max(0.0, now - ready_at[nid]),
                                     "links": feat[nid]["reserved_links"]})
            if len(completed) == len(lookup):
                break
            future = ([events[0][0]] if events else []) + [ready_at[nid] for nid in ready if ready_at[nid] > now]
            if not future:
                raise RuntimeError("Dependency replay stalled")
            now = min(future)
        completions = [max(completed[n.id] for n in problem.graph.nodes if n.job == j)
                       for j in range(len(problem.allocations))]
        jct = [t - a for t, a in zip(completions, problem.arrivals_s)]
        return {"mean_jct_s": sum(jct) / len(jct), "jct_s": jct, "completion_s": completions,
                "backend": "event_replay", "diagnostics": {"collectives": timeline},
                "audits": {"complete_graph": len(completed) == len(lookup),
                           "resources_released": not occupied, "events_drained": not events}}


class CommandBackend:
    """Send a problem and plan to an external evaluator through stdin/stdout JSON."""
    def __init__(self, command, timeout_s=3600):
        if not command or not all(isinstance(s, str) for s in command):
            raise ValueError("Provide an evaluator argument list")
        self.command, self.timeout_s = command, timeout_s

    def evaluate(self, problem, plan):
        proc = subprocess.run(self.command, input=json.dumps({"problem": problem.to_dict(), "plan": plan.to_dict()}),
                              capture_output=True, text=True, timeout=self.timeout_s)
        if proc.returncode:
            raise RuntimeError("Evaluator failed: " + proc.stderr[-1500:])
        return json.loads(proc.stdout)


def make_backend(spec):
    if spec["type"] == "critical_path":
        return CriticalPathBackend()
    if spec["type"] == "event_replay":
        return EventReplayBackend()
    if spec["type"] == "command":
        return CommandBackend(spec["command"], spec.get("timeout_s", 3600))
    raise ValueError("Unknown backend type")


def validate_result(result, jobs):
    if len(result.get("jct_s", [])) != jobs:
        raise ValueError("Evaluator must return one JCT per job")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in result["jct_s"]):
        raise ValueError("JCT values must be finite and nonnegative")
    mean = result.get("mean_jct_s")
    if type(mean) not in (int, float) or not math.isfinite(mean) or abs(mean - sum(result["jct_s"]) / jobs) > 1e-8:
        raise ValueError("Mean JCT must match the per-job values")
    if not result.get("audits") or not all(result["audits"].values()):
        raise ValueError("Execution audits must pass")
    return result
