"""Budgeted tool calls and reference-validated plan selection."""
import copy
import json
import math
from pathlib import Path
import threading
import time
from .analysis import summarize
from .backends import validate_result
from .model import Plan


class Planner:
    def __init__(self, problem, fast, reference, output, fast_budget=64, reference_budget=8):
        if type(fast_budget) is not int or type(reference_budget) is not int or fast_budget < 0 or reference_budget < 1:
            raise ValueError("Provide nonnegative fast and positive reference request budgets")
        self.problem = copy.deepcopy(problem).validate()
        self.backends = {"fast": fast, "reference": reference}
        self.remaining = {"fast": fast_budget, "reference": reference_budget}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "session.json").exists() or (self.output / "costs.jsonl").exists():
            raise FileExistsError("Choose a fresh planning output directory")
        self.incumbent = None
        self.records = []
        self.cache = {}
        self.lock = threading.RLock()
        self.started = time.perf_counter()
        self.probe(Plan(self.problem.initial_placement).to_dict(), "reference")

    def _save(self):
        target = self.output / "session.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps({"problem_sha256": self.problem.fingerprint, "remaining": self.remaining,
                                    "best": self.incumbent, "requests": self.records}, indent=2) + "\n")
        temp.replace(target)

    def _ledger(self, record):
        with (self.output / "costs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def probe(self, data, kernel="fast"):
        with self.lock:
            if kernel not in self.remaining or self.remaining[kernel] <= 0:
                raise ValueError("Requested evaluation budget is exhausted")
            self.remaining[kernel] -= 1
            record = {"index": len(self.records), "kind": "evaluation", "kernel": kernel,
                      "plan": data, "status": "running"}
            self.records.append(record)
            start = time.perf_counter()
            self._save()
            try:
                plan = Plan.from_dict(data).validate(self.problem)
                key = (kernel, json.dumps(plan.to_dict(), sort_keys=True))
                record["cache_hit"] = key in self.cache
                result = copy.deepcopy(self.cache[key]) if key in self.cache else self.backends[kernel].evaluate(self.problem, plan)
                validate_result(result, len(self.problem.allocations))
                self.cache[key] = copy.deepcopy(result)
                record.update(status="ok", result=result)
                if kernel == "reference" and (self.incumbent is None or result["mean_jct_s"] < self.incumbent["result"]["mean_jct_s"]):
                    self.incumbent = {"plan": plan.to_dict(), "result": result, "request_index": record["index"]}
                return copy.deepcopy(result)
            except Exception as exc:
                record.update(status="error", error=str(exc))
                raise
            finally:
                record["wall_s"] = time.perf_counter() - start
                self._ledger(record)
                self._save()

    def analyze(self, placements):
        start = time.perf_counter()
        try:
            if not 1 <= len(placements) <= 128:
                raise ValueError("Analyze between one and 128 placements per request")
            return [summarize(self.problem, Plan(tuple(p))) for p in placements]
        finally:
            self._ledger({"kind": "analysis", "placements": len(placements), "wall_s": time.perf_counter() - start})

    def budget(self):
        with self.lock:
            return dict(self.remaining)

    def best(self):
        with self.lock:
            return copy.deepcopy(self.incumbent)

    def record_model_run(self, **data):
        self._ledger({"kind": "model_run", **data})

    def summary(self):
        return {"best": self.best(), "remaining": self.budget(),
                "evaluation_attempts": len(self.records),
                "failed_requests": sum(r["status"] == "error" for r in self.records),
                "cache_hits": sum(r.get("cache_hit", False) for r in self.records),
                "planning_wall_s": time.perf_counter() - self.started}
