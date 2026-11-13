"""Expose tools to a Codex session and retain execution and model costs."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from .analysis import summarize
from .model import Plan

TASK = """# Joint planning with OrbitWeaver

Minimize reference-evaluated mean job completion time (JCT).
Read case_input.json, then import lab to inspect and improve the current plan.
Use the Python interpreter specified below.

Tools:
- lab.budget(): remaining fast and reference requests.
- lab.best(): current reference-validated incumbent.
- lab.analyze([placement, ...]): subgroup, path-demand and dependency features.
- lab.probe(placement, rule="progress", kernel="fast", alpha=0, beta=0).
- lab.probe_plan({"placement": [...], "priority": {...}}, kernel="reference").

Permute ranks within their assigned GPU sets. Keep execution graphs, recorded
durations, job arrivals and physical interconnect parameters fixed.
Higher priority scores dispatch first. Available rules: progress, shortest,
tail, normalized_tail, job_asc, job_desc, source_order, zero.
For progress, alpha weights arrival offsets and beta mixes remaining-tail
progress with source-node order. Choose finite alpha and beta.

Use structural observations to propose joint placement/priority changes.
Evaluate promising candidates quickly and reference-check them while budget
remains. Retain successful hypotheses and explain the effect of path sharing
and dependency waits in rationale.md. Every evaluation attempt, including
invalid requests and cache hits, consumes its corresponding request budget.
The controller delivers its best reference-validated plan.
"""


def public_result(result):
    output = {k: result[k] for k in ("mean_jct_s", "jct_s", "backend", "audits") if k in result}
    diagnostics = result.get("diagnostics", {})
    if "collectives" in diagnostics:
        rows = diagnostics["collectives"]
        output["diagnostics"] = {"collective_count": len(rows),
                                 "highest_wait_collectives": sorted(rows, key=lambda r: -r["wait_s"])[:20]}
    elif diagnostics:
        output["diagnostics"] = diagnostics
    return output


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            request = json.loads(self.rfile.readline(2 << 20))
            planner = self.server.planner
            operation = request["op"]
            if operation == "budget":
                result = planner.budget()
            elif operation == "best":
                best = planner.best()
                result = {**best, "result": public_result(best["result"])}
            elif operation == "analyze":
                result = planner.analyze(request["placements"])
            elif operation == "probe":
                result = public_result(planner.probe(request["plan"], request.get("kernel", "fast")))
            else:
                raise ValueError("Unknown tool operation")
            response = {"ok": True, "value": result}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        self.wfile.write((json.dumps(response) + "\n").encode())


@contextmanager
def tool_server(planner, output):
    output = Path(output)
    address = str(Path(tempfile.gettempdir()) / ("orbitweaver-" + uuid.uuid4().hex[:12] + ".sock"))
    if len(address.encode()) >= 100:
        address = "/tmp/orbitweaver-" + uuid.uuid4().hex[:12] + ".sock"
    server = socketserver.ThreadingUnixStreamServer(address, Handler)
    server.daemon_threads = True
    server.planner = planner
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (output / "lab.py").write_text(
        "from orbitweaver.client import Client\n"
        + "_client = Client(" + repr(address) + ")\n"
        + "budget = _client.budget\nbest = _client.best\nanalyze = _client.analyze\n"
        + "probe = _client.probe\nprobe_plan = _client.probe_plan\n")
    problem = planner.problem
    analysis = summarize(problem, Plan(problem.initial_placement))
    context = {"name": problem.name, "ranks": problem.graph.ranks,
               "nodes": len(problem.graph.nodes), "allocations": problem.allocations,
               "arrivals_s": problem.arrivals_s,
               "links": problem.to_dict()["links"],
               "initial": {"plan": planner.best()["plan"], "result": public_result(planner.best()["result"])},
               "communication_groups": analysis["groups"][:32]}
    (output / "case_input.json").write_text(json.dumps(context, indent=2) + "\n")
    (output / "TASK.md").write_text(TASK + "\nPython: " + sys.executable + "\n")
    try:
        yield address
    finally:
        server.shutdown()
        server.server_close()
        Path(address).unlink(missing_ok=True)
        thread.join()


def run_codex(planner, output, model="gpt-6.1-sol", effort="xhigh", wall_limit_s=1800, executable="codex"):
    output = Path(output).resolve()
    executable = shutil.which(executable) or executable
    command = [executable, "exec", "--cd", str(output), "--sandbox", "workspace-write",
               "--model", model, "-c", 'model_reasoning_effort="' + effort + '"',
               "--json", "--output-last-message", str(output / "answer.txt"), "-"]
    started, timed_out, returncode, error = time.perf_counter(), False, None, None
    with tool_server(planner, output):
        try:
            with (output / "agent_events.jsonl").open("w") as stdout, (output / "agent_stderr.log").open("w") as stderr:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                           text=True, start_new_session=True)
                try:
                    process.communicate("Read TASK.md and case_input.json. Use lab to improve and reference-validate the joint plan.", timeout=wall_limit_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                returncode = process.returncode
        except Exception as exc:
            error = str(exc)
    usages = []
    events = output / "agent_events.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed" and "usage" in event:
                usages.append(event["usage"])
    usage = {k: sum(u.get(k, 0) for u in usages) for k in set().union(*(u.keys() for u in usages))} if usages else None
    planner.record_model_run(model=model, reasoning_effort=effort, wall_s=time.perf_counter() - started,
                             exit_code=returncode, timed_out=timed_out, error=error, usage=usage)
    result = planner.summary()
    result["agent"] = {"exit_code": returncode, "timed_out": timed_out, "error": error, "usage": usage}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
